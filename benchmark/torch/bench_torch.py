# torchrun --nproc_per_node 2 --master_port 26543 ./benchmark/bench_torch.py

import logging
import argparse
import os
import sys
from functools import partial
from contextlib import nullcontext
import ctypes

import torch
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tensor import DeviceMesh

from easydist import easydist_setup, mdconfig
from easydist.torch.api import easydist_compile, set_device_mesh
from easydist.utils.timer import EDTimer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benchmark.torch.model import GPT, GATLayer, wresnet50
from benchmark.bench_case import GPTCase, ResNetCase, GATCase

torch.manual_seed(0)

def get_gpt_case():
    case = GPTCase()
    model = GPT(depth=case.num_layers, dim=case.hidden_dim, num_heads=case.num_heads)
    data_in = torch.ones(case.batch_size, case.seq_size, case.hidden_dim)
    return model, data_in


def get_resnet_case():
    case = ResNetCase()
    model = wresnet50()
    data_in = torch.ones(case.batch_size, 3, 224, 224)
    return model, data_in


def get_gat_case():
    case = GATCase()
    model = GATLayer(case.in_feature, case.out_feature)
    data_in = torch.ones(case.num_node, case.in_feature)
    adj = torch.ones(case.num_node, case.num_node)
    return model, [data_in, adj]


def bench_ddp(model, data_in):

    if not isinstance(data_in, list):
        data_in = [data_in]

    world_size = torch.distributed.get_world_size()
    for i in range(len(data_in)):
        data_in[i] = torch.chunk(data_in[i], world_size)[0]

    ddp_model = DDP(model)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    def train_step():
        optimizer.zero_grad()
        out = ddp_model(*data_in)
        out_grad = torch.ones_like(out)
        out.backward(out_grad)
        optimizer.step()

    torch.cuda.reset_peak_memory_stats()

    timer = EDTimer(train_step, in_ms=False)

    elaps_time = timer.time()
    peak_memory = torch.cuda.max_memory_allocated()

    print(f"Memory: {peak_memory / 1024 / 1024 / 1024} GB")
    print(f"Time: {elaps_time}")


def bench_fsdp(model, data_in):

    if not isinstance(data_in, list):
        data_in = [data_in]

    if not isinstance(model, GATLayer):
        world_size = torch.distributed.get_world_size()
        for i in range(len(data_in)):
            data_in[i] = torch.chunk(data_in[i], world_size)[0]

    ddp_model = FSDP(model)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    def train_step():
        optimizer.zero_grad()
        out = ddp_model(*data_in)
        out_grad = torch.ones_like(out)
        out.backward(out_grad)
        optimizer.step()

    torch.cuda.reset_peak_memory_stats()

    timer = EDTimer(train_step, in_ms=False)

    elaps_time = timer.time()
    peak_memory = torch.cuda.max_memory_allocated()

    print(f"Memory: {peak_memory / 1024 / 1024 / 1024} GB")
    print(f"Time: {elaps_time}")


def to_meta(node_output):
    if type(node_output) is torch.Tensor:
        return node_output.to(device="meta")
    elif type(node_output) is torch.nn.parameter.Parameter:
        return node_output.to(device="meta")
    else:
        return node_output


def bench_easydist(model, data_in):

    if not isinstance(data_in, list):
        data_in = [data_in]

    optimizer = optim.SGD(model.parameters(), lr=0.001)

    @easydist_compile(cuda_graph=False)
    def train_step(model, optimizer, data_in):
        output_ = model(*data_in)
        output_grad = torch.ones_like(output_)
        output_.backward(output_grad)
        optimizer.step()
        optimizer.zero_grad()
        return output_

    train_step_partial = partial(train_step, model, optimizer, data_in)
    train_step_partial()

    torch.cuda.empty_cache()

    # TODO(wuhao): temporarily banned torch.cuda.reset_peak_memory_stats(), 
    # because CUDAPluggableAllocator didn't support this method

    # torch.cuda.reset_peak_memory_stats()

    timer = EDTimer(train_step_partial, in_ms=False)

    elaps_time = timer.time()

    # TODO(wuhao): temporarily banned torch.cuda.max_memory_allocated(), 
    # because CUDAPluggableAllocator didn't support this method
 
    # peak_memory = torch.cuda.max_memory_allocated()
    peak_memory = 0

    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"[{local_rank}] Memory: {peak_memory / 1024 / 1024 / 1024} GB")
    print(f"[{local_rank}] Time: {elaps_time}")

def main():

    parser = argparse.ArgumentParser(description="Simple example of parallelize model.")

    parser.add_argument("--model",
                        type=str,
                        default=None,
                        choices=["gpt", "resnet", "gat"],
                        required=True)
    parser.add_argument("--fake-init", action="store_true")
    parser.add_argument("--spmd0", type=int, default=1)
    parser.add_argument("--spmd1", type=int, default=1)

    args = parser.parse_args()

    spmd0, spmd1 = args.spmd0, args.spmd1

    # setup easydist
    easydist_setup(backend="torch", device="cuda")

    # setup distributed
    torch.distributed.init_process_group(backend="nccl")
    world_size = int(os.environ["WORLD_SIZE"])
    if spmd0 * spmd1 != world_size:
        raise ValueError(f"spmd0({spmd0}) * spmd1({spmd1}) should be equal to world_size({world_size})")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device_mesh = DeviceMesh('cuda', torch.arange(world_size).reshape(spmd0, spmd1), mesh_dim_names=['spmd0', 'spmd1'])
    set_device_mesh(device_mesh)

    fake_mode = FakeTensorMode()
    # (NOTE) initialize cuda context first see https://github.com/pytorch/pytorch/issues/92627
    torch.ones(1).cuda()
    with torch.device('cuda'), fake_mode if args.fake_init else nullcontext():
        if args.model == "gpt":
            model, data_in = get_gpt_case()
        elif args.model == "resnet":
            model, data_in = get_resnet_case()
        elif args.model == "gat":
            model, data_in = get_gat_case()

    bench_easydist(model, data_in)


if __name__ == '__main__':
    main()
