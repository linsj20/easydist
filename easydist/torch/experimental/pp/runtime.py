# Copyright (c) 2023, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import operator
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed._tensor import Replicate

from easydist.torch.device_mesh import get_device_mesh
from easydist.torch.experimental.pp.compile_pipeline import (
    CompiledMeta,
    CompiledStage,
    StateType,
)
from easydist.torch.experimental.pp.microbatch import (
    DEFAULT_CHUNK_DIM,
    CustomReducer,
    TensorChunkSpec,
    merge_chunks,
    split_args_kwargs_into_chunks,
)
from easydist.torch.init_helper import materialize_zero
from easydist.torch.utils import do_spmd_comm

logger = logging.getLogger(__name__)


def maybe_batch_isend_irecv(p2p_op_list):
    '''
    note: this might lead to hang when the first collective call in the group
    see dist.batch_isend_irecv for more details
    '''
    if len(p2p_op_list) == 0:
        return []
    return dist.batch_isend_irecv(p2p_op_list)


class Placeholder:

    def __init__(self, input_name: str):
        self.input_name = input_name

    def __repr__(self):
        return f"{type(self).__class__}({self.input_name})"


class StageKwargPlaceholder(Placeholder):

    def __init__(self, input_name: str):
        super().__init__(input_name)


class RecevPlaceholder(Placeholder):

    def __init__(self, input_name: str, source: int, example_tensor: FakeTensor,
                 device: torch.device):
        super().__init__(input_name)
        self.source = source
        self.buffer = materialize_zero(example_tensor, device)


class RuntimeMixin(ABC):
    @abstractmethod
    def forward_send_one_chunk(self) -> List[dist.Work]:
        raise NotImplementedError

    @abstractmethod
    def forward_compute_one_chunk(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward_recv_one_chunk(self, wait=True) -> List[dist.Work]:
        raise NotImplementedError

    @abstractmethod
    def backward_recv_one_chunk(self, wait=True) -> List[dist.Work]:
        raise NotImplementedError

    @abstractmethod
    def backward_compute_one_chunk(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def backward_send_one_chunk(self) -> List[dist.Work]:
        raise NotImplementedError

    @abstractmethod
    def merge_and_assign_chunked_grads(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError


class PipelineStage(RuntimeMixin):

    def __init__(self, schedule_cls: Type['Schedule'], local_gm: fx.GraphModule, stage_idx: int,
                 compiled_meta: CompiledMeta, compiled_stage: CompiledStage,
                 node_metas: Dict[str, Dict[str, FakeTensor]], num_chunks: int,
                 args_chunk_spec: Optional[Tuple[TensorChunkSpec]],
                 kwargs_chunk_spec: Optional[Dict[str, TensorChunkSpec]],
                 returns_chunk_spec: Optional[Tuple[Union[TensorChunkSpec, CustomReducer]]],
                 device: torch.device, pp_group: dist.ProcessGroup, sharded_graph: fx.GraphModule,
                 return_to_all_stages: bool, accumulate_grads_inplace: bool):
        # meta info
        self.name = f'stage_{stage_idx}'
        self._init_fw_bw_step_nodes(local_gm)
        assert issubclass(schedule_cls, Schedule), "schedule_cls must be the Schedule class"
        self.stage_idx = stage_idx
        self.compiled_meta = compiled_meta
        self.compiled_stage = compiled_stage
        self.num_chunks = num_chunks
        self._init_inputs_nodes_spec(compiled_meta, args_chunk_spec, kwargs_chunk_spec)
        self._init_returns_nodes_spec(compiled_meta, returns_chunk_spec)
        self.device = device
        self.pp_group = pp_group

        self.pp_rank = dist.get_rank(pp_group)
        self.num_stages = compiled_meta.nstages
        self.graph = sharded_graph
        self.return_to_all_stages = return_to_all_stages
        self.accumulate_grads_inplace = accumulate_grads_inplace

        if dist.get_world_size(self.pp_group) > self.num_stages:
            raise RuntimeError(
                "Number of ranks is larger than number of stages, some ranks are unused")

        # communication infra
        self._init_communication(node_metas)

        # runtime states
        self._init_runtime_states()

        # post init here (schedule_cls requires PipelineStage initialized)
        self.schedule = schedule_cls(self)

    def _init_fw_bw_step_nodes(self, local_gm):
        # Find stage forward node in graph
        self.fw_node = None
        for node in local_gm.graph.nodes:
            if node.name == f'{self.name}_fw':
                assert self.fw_node is None, "Multiple forward nodes found"
                self.fw_node = node
        if not self.fw_node:
            raise AssertionError(f"Cannot find {self.name} in graph")

        # Find stage backward node in graph
        self.bw_node = None
        for node in local_gm.graph.nodes:
            if node.name == f'{self.name}_bw':
                assert self.bw_node is None, "Multiple backward nodes found"
                self.bw_node = node

        # Find stage step node in graph
        self.step_node = None
        for node in local_gm.graph.nodes:
            if node.name == f'{self.name}_step':
                assert self.step_node is None, "Multiple step nodes found"
                self.step_node = node

    def _init_inputs_nodes_spec(self, compiled_meta: CompiledMeta, args_chunk_spec,
                                kwargs_chunk_spec):
        node_val_chunk_spec = {}
        args_nodes_flatten, _ = pytree.tree_flatten(compiled_meta.args_nodes_unflatten)
        args_chunk_spec = args_chunk_spec or [None] * len(
            args_nodes_flatten)  # input could be non tensor, use None instead of TensorChunkSpec
        args_chunk_spec_flatten, _ = pytree.tree_flatten(args_chunk_spec)
        assert len(args_chunk_spec_flatten) == len(args_nodes_flatten)
        for node_name, arg_chunk_spec in zip(compiled_meta.args_nodes_unflatten,
                                             args_chunk_spec_flatten):
            node_val_chunk_spec[node_name] = arg_chunk_spec

        kwargs_nodes_flatten, spec1 = pytree.tree_flatten(compiled_meta.kwargs_nodes_unflatten)
        kwargs_chunk_spec = kwargs_chunk_spec or {
            node_name: TensorChunkSpec(DEFAULT_CHUNK_DIM)
            for node_name in kwargs_nodes_flatten
        }
        kwargs_chunk_spec_flatten, spec2 = pytree.tree_flatten(kwargs_chunk_spec)
        assert spec1 == spec2
        for node_name, kwarg_chunk_spec in zip(kwargs_nodes_flatten, kwargs_chunk_spec_flatten):
            node_val_chunk_spec[node_name] = kwarg_chunk_spec

        self.inputs_nodes_chunk_spec = node_val_chunk_spec

    def _init_returns_nodes_spec(self, compiled_meta, returns_chunk_spec):
        returns_nodes_chunk_spec = {}
        returns_chunk_spec = returns_chunk_spec or [TensorChunkSpec(DEFAULT_CHUNK_DIM)] * len(
            compiled_meta.return_nodes_flatten)
        returns_chunk_spec_flatten, _ = pytree.tree_flatten(returns_chunk_spec)
        assert len(returns_chunk_spec_flatten) == len(compiled_meta.return_nodes_flatten)
        for name, spec in zip(compiled_meta.return_nodes_flatten, returns_chunk_spec_flatten):
            returns_nodes_chunk_spec[name] = spec

        self.returns_nodes_chunk_spec = returns_nodes_chunk_spec

    def _init_communication(self, node_metas):
        """
        Create send/recv infrastructures for activations (during forward) and gradients (during backward)
        """
        # Create stage id to group rank mapping
        # In interleaved case, `group_rank` is stage index % group size.
        stage_index_to_pp_rank: Dict[int, int] = {}
        pg_world_size = dist.get_world_size(self.pp_group)
        assert pg_world_size == self.num_stages, "Currently only support one stage per rank"  # TODO @botbw
        for i in range(self.num_stages):
            # We only support wrapped-around interleaving
            peer_rank = i % pg_world_size
            stage_index_to_pp_rank.setdefault(i, peer_rank)
        self.stage_index_to_group_rank = stage_index_to_pp_rank

        # chunk : Dict of kwarg buffers
        self.fw_kwargs_recv_info = self._create_recv_info(node_metas,
                                                          self.fw_node,
                                                          is_forward=True)
        self.fw_act_send_info = self._create_send_info(self.fw_node, is_forward=True)

        if self.bw_node is not None:
            self.bw_kwargs_recv_info = self._create_recv_info(node_metas,
                                                              self.bw_node,
                                                              is_forward=False)
            self.bw_grad_send_info = self._create_send_info(self.bw_node, is_forward=False)

    def _init_runtime_states(self):
        self.cur_fw_send_chunk = None
        self.cur_bw_send_chunk = None
        self.cur_fw_chunk_id = 0
        self.cur_bw_chunk_id = 0
        self.kwargs_chunks = [{} for _ in range(self.num_chunks)]
        self.saved_tensors_bw_chunks: List[Dict[str, Any]] = [{} for _ in range(self.num_chunks)]
        self.saved_params_step: Dict[str, Any] = {}
        self.saved_grads_step_chunks: List[Dict[str, Any]] = [{} for _ in range(self.num_chunks)]
        self.output_grads_chunks: List[Dict[str, Any]] = [{} for _ in range(self.num_chunks)]
        self.returns_chunks: List[Dict[str, Any]] = [{} for _ in range(self.num_chunks)]

        if self.accumulate_grads_inplace:
            self.output_grads_reduced = {}
            self.step_grads_reduced = {}

    def reset_and_check_runtime_states(self):

        def clear_single(chunks_list):
            for chunk in chunks_list:
                chunk.clear()
        def check_single(chunks_list):
            for chunk in chunks_list:
                assert len(chunk) == 0

        self.cur_fw_send_chunk = None
        self.cur_bw_send_chunk = None
        self.cur_fw_chunk_id = 0
        self.cur_bw_chunk_id = 0

        clear_single(self.kwargs_chunks)

        check_single(self.returns_chunks)
        check_single(self.saved_tensors_bw_chunks)
        check_single(self.saved_grads_step_chunks)
        check_single(self.output_grads_chunks)

        if self.accumulate_grads_inplace:
            assert len(self.output_grads_reduced) == 0
            assert len(self.step_grads_reduced) == 0

    def _create_send_info(self, node: fx.Node,
                          is_forward: bool) -> Dict[int, List[str]]:
        to_sort = []
        for user in node.users:
            assert user.target is operator.getitem, "Output must be a dict"
            out_str = user.args[-1]
            assert isinstance(out_str, str)
            for gi_user in user.users:
                dst_rank = gi_user.meta['stage_idx']
                to_sort.append((dst_rank, out_str))
        if is_forward:
            to_sort.sort(key=lambda x:
                         (x[0], x[1]))  # send lower to rank first and in alphabetical order
        else:
            to_sort.sort(key=lambda x:
                         (-x[0], x[1]))  # send higher to rank first and in alphabetical order

        send_info_by_stage = defaultdict(list)
        for dst_rank, out_str in to_sort:
            send_info_by_stage[dst_rank].append(out_str)

        return send_info_by_stage

    def _create_send_ops(self, send_info: Dict[int, List[str]], output_dict: Dict[str,
                                                                                  torch.Tensor]) -> List[List[dist.Work]]:
        # Send requests of a chunk
        send_ops_by_dst = defaultdict(list)
        for dst, nodes in send_info.items():
            peer_rank = self.stage_index_to_group_rank[dst]
            peer_global_rank = peer_rank if self.pp_group is None else dist.get_global_rank(
                self.pp_group, peer_rank)
            for node in nodes:
                val = output_dict[node]
                send_ops_by_dst[dst].append(
                    dist.P2POp(dist.isend, val, peer_global_rank, self.pp_group))

        return [send_ops_by_dst[dst] for dst in sorted(send_info.keys())]

    def _create_recv_info(
        self,
        node_metas: Dict[str, Dict],
        node: fx.Node,
        is_forward: bool,
    ) -> Dict[int, List[Placeholder]]:
        to_sort = []
        for gi_input in node.kwargs.values():
            if gi_input.op == "placeholder":
                to_sort.append((-1, gi_input.name))
                continue
            example_value = node_metas[gi_input.name]["val"]
            src_rank = gi_input.args[0].meta['stage_idx']
            global_src_rank = src_rank if self.pp_group is None else dist.get_global_rank(
                self.pp_group, src_rank)
            to_sort.append((global_src_rank, gi_input.name, example_value))

        if is_forward:
            # receive lower rank first and in alphabetical order
            to_sort.sort(key=lambda x: (x[0], x[1]))
        else:
            # receive higer rank first and in alphabetical order
            to_sort.sort(key=lambda x: (-x[0], x[1]))
        kwargs_recv_info = defaultdict(list)
        for x in to_sort:
            if x[0] == -1:
                assert is_forward
                kwargs_recv_info[0].append(StageKwargPlaceholder(
                    x[1]))  # args recev with rank 0 (lowest rank)
            else:
                global_src_rank, name, example_value = x
                kwargs_recv_info[global_src_rank].append(
                    RecevPlaceholder(name, global_src_rank, example_value, self.device))

        return kwargs_recv_info

    def _create_recv_ops(self, recv_info: Dict[int, List[Placeholder]]) -> List[List[dist.Work]]:
        recv_ops_by_src = []
        for src, ph_list in recv_info.items():
            rec_ops = []
            for ph in ph_list:
                if isinstance(ph, RecevPlaceholder):
                    rec_ops.append(dist.P2POp(dist.irecv, ph.buffer, src, self.pp_group))
            recv_ops_by_src.append(rec_ops)
        return recv_ops_by_src

    def collect_kwargs(
        self,
        recv_info: Dict[int, List[Placeholder]],
        chunk: int,
    ):
        chunk_kwargs = self.kwargs_chunks[chunk]

        composite_kwargs = {}
        for rank, ph_list in recv_info.items():
            for ph in ph_list:
                if isinstance(ph, RecevPlaceholder):
                    composite_kwargs[ph.input_name] = ph.buffer.clone()  # NOTE: need clone here so that all micro-batches use different memory
                else:
                    composite_kwargs[ph.input_name] = chunk_kwargs[ph.input_name]

        return composite_kwargs

    def forward_recv_one_chunk(self, wait=True) -> List[dist.Work]:
        # Receive activations
        recv_ops_by_src = self._create_recv_ops(self.fw_kwargs_recv_info)
        recv_reqs = []
        for ops in recv_ops_by_src:
            recv_reqs += maybe_batch_isend_irecv(ops)
        if wait:
            for req in recv_reqs:
                req.wait()
        return recv_reqs

    def forward_compute_one_chunk(self):
        # Collect activations and kwargs
        composite_kwargs_chunk = self.collect_kwargs(self.fw_kwargs_recv_info,
                                                     self.cur_fw_chunk_id)

        # Compute forward
        self.cur_fw_send_chunk = self.compiled_stage.forward(
            self.saved_tensors_bw_chunks[self.cur_fw_chunk_id],
            self.saved_params_step,
            self.returns_chunks[self.cur_fw_chunk_id],
            **composite_kwargs_chunk
        )
        # Update runtime states
        self.cur_fw_chunk_id += 1

    def forward_send_one_chunk(self) -> List[dist.Work]:
        # Send activations
        send_ops_by_dst = self._create_send_ops(self.fw_act_send_info, self.cur_fw_send_chunk)
        reqs = []
        for ops in send_ops_by_dst:
            reqs += maybe_batch_isend_irecv(ops)
        return reqs

    def backward_recv_one_chunk(self, wait=True) -> List[dist.Work]:
        # Receive grads
        recv_ops_by_src = self._create_recv_ops(self.bw_kwargs_recv_info)
        recv_reqs = []
        for ops in recv_ops_by_src:
            recv_reqs += maybe_batch_isend_irecv(ops)
        if wait:
            for req in recv_reqs:
                req.wait()
        return recv_reqs

    def backward_compute_one_chunk(self):
        # Collect grads and kwargs
        composite_kwargs_chunk = self.collect_kwargs(self.bw_kwargs_recv_info,
                                                     self.cur_bw_chunk_id)

        # Compute backward
        self.cur_bw_send_chunk = self.compiled_stage.backward(
            self.saved_tensors_bw_chunks[self.cur_bw_chunk_id],
            self.saved_grads_step_chunks[self.cur_bw_chunk_id],
            self.output_grads_chunks[self.cur_bw_chunk_id],
            **composite_kwargs_chunk
        )

        if self.accumulate_grads_inplace:
            for grad_node, grad in self.saved_grads_step_chunks[self.cur_bw_chunk_id].items():
                if grad_node in self.step_grads_reduced:
                    self.step_grads_reduced[grad_node].add_(grad)
                else:
                    self.step_grads_reduced[grad_node] = grad
            self.saved_grads_step_chunks[self.cur_bw_chunk_id].clear()

            for grad_node, grad in self.output_grads_chunks[self.cur_bw_chunk_id].items():
                if grad_node in self.output_grads_reduced:
                    self.output_grads_reduced[grad_node].add_(grad)
                else:
                    self.output_grads_reduced[grad_node] = grad
            self.output_grads_chunks[self.cur_bw_chunk_id].clear()

        # Update runtime states
        self.cur_bw_chunk_id += 1

    def backward_send_one_chunk(self) -> List[dist.Work]:
        # Send grads
        send_ops_by_dst = self._create_send_ops(self.bw_grad_send_info, self.cur_bw_send_chunk)
        reqs = []
        for ops in send_ops_by_dst:
            reqs += maybe_batch_isend_irecv(ops)
        return reqs

    def step(self):
        self.compiled_stage.step(self.saved_params_step, self.step_grads_reduced)

    def split_input_kwargs(self, kwargs):
        return split_args_kwargs_into_chunks(
            (),
            kwargs,
            self.num_chunks,
            None,
            self.inputs_nodes_chunk_spec,
        )[1]

    @torch.no_grad
    def merge_and_assign_chunked_grads(self) -> None:
        if not self.accumulate_grads_inplace:
            self.step_grads_reduced = reduce(lambda a, b: {k: torch.add(a[k], b[k]) for k in a}, self.saved_grads_step_chunks)
            for chunk in self.saved_grads_step_chunks:
                chunk.clear()
            self.output_grads_reduced = reduce(lambda a, b: {k: torch.add(a[k], b[k]) for k in a}, self.output_grads_chunks)
            for chunk in self.output_grads_chunks:
                chunk.clear()

        if self.step_node is None:
            for grad_name, grad in self.output_grads_reduced.items():
                param_torch_name = self.compiled_meta.output_grads_map.inv_get(grad_name)
                param_node_name = self.compiled_meta.input_params_map.get(param_torch_name)
                self.compiled_stage.fw_gm.node_states[StateType.PARAMS][param_node_name].grad = grad

        self.output_grads_reduced.clear()

    @torch.no_grad
    def merge_chunked_returns(self) -> Tuple[Any, ...]:
        for chunk in self.returns_chunks:
            for node_name in self.returns_nodes_chunk_spec:
                chunk[node_name] = chunk.get(node_name, None)

        returns = merge_chunks(self.returns_chunks, self.returns_nodes_chunk_spec)

        for chunk in self.returns_chunks:
            chunk.clear()

        returns_all_gather = [None for _ in range(self.num_stages)]
        dist.all_gather_object(returns_all_gather, returns, group=self.pp_group)

        returns_dict = {}
        for returns in returns_all_gather:
            for k, v, in returns.items():
                if v is not None:
                    returns_dict[k] = v
        return_tensors = [returns_dict[node_name] for node_name in self.compiled_meta.return_nodes_flatten]

        return pytree.tree_unflatten(return_tensors, self.compiled_meta.return_nodes_spec)

    def _all_gather_inner(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        state_dicts = [None for _ in range(self.num_stages)]
        dist.all_gather_object(state_dicts,
                            state_dict,
                            group=self.pp_group)
        return reduce(lambda a, b: {**a, **b}, state_dicts)

    def _optimizer_state_dict(self, all_gather=True) -> Dict[str, Any]:
        state_dict = self.compiled_stage._optimizer_state_dict()
        if all_gather:
            state_dict = self._all_gather_inner(state_dict)
        return state_dict

    def state_dict(self, all_gather=True) -> Dict[str, Any]:
        state_dict = self.compiled_stage.state_dict()
        if all_gather:
            state_dict = self._all_gather_inner(state_dict)
        return state_dict

    def named_parameters(self, all_gather=True) -> Dict[str, Any]:
        state_dict = self.compiled_stage.named_parameters()
        if all_gather:
            state_dict = self._all_gather_inner(state_dict)
        return state_dict

    def named_buffers(self, all_gather=True) -> Dict[str, Any]:
        state_dict = self.compiled_stage.named_buffers()
        if all_gather:
            state_dict = self._all_gather_inner(state_dict)
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        self.compiled_stage.load_state_dict(state_dict, strict=strict)

    def load_optimizer_state_dict(self, state_dict, strict=True):
        self.compiled_stage.load_optimizer_state_dict(state_dict, strict=strict)


    def __call__(self, *args, **kwargs) -> None:
        # Clean per iteration
        self.reset_and_check_runtime_states()

        # TODO @botbw: check TODO in compiled.py: compiled_func
        args_kwargs_vals_flatten, spec_val = pytree.tree_flatten((args, kwargs))
        args_kwargs_nodes_flatten, spec_node = pytree.tree_flatten(
            (self.compiled_meta.args_nodes_unflatten, self.compiled_meta.kwargs_nodes_unflatten))
        if self.compiled_meta.tensors_spmd_strategies:
            device_mesh = get_device_mesh('spmd')
            for i, (node_name, val) in enumerate(zip(args_kwargs_nodes_flatten, args_kwargs_vals_flatten)):
                if isinstance(val, torch.Tensor):
                    src_specs = [Replicate()] * device_mesh.mesh.dim()
                    tgt_specs = self.compiled_meta.tensors_spmd_strategies[node_name]
                    args_kwargs_vals_flatten[i] = do_spmd_comm(val, src_specs, tgt_specs)
            args, kwargs = pytree.tree_unflatten(args_kwargs_vals_flatten, spec_val)
        assert spec_val == spec_node, "Mismatched args/kwargs"

        input_node_vals = {}
        for node, val in zip(args_kwargs_nodes_flatten, args_kwargs_vals_flatten):
            if isinstance(val, torch.Tensor):
                val = val.to(self.device)
            input_node_vals[node] = val

        # Split inputs into chunks
        self.kwargs_chunks = self.split_input_kwargs(input_node_vals)

        self.schedule()

        return self.merge_chunked_returns()

    def run_with_graph(self, graph, *args,
                       **kwargs):  # TODO @botbw: could construct a partial graph here
        return self(*args, **kwargs)


def print_tensor_dict(chunk, di):
    print(f'Chunk {chunk}')
    for k, v in di.items():
        print(f'{k} size {v.size()} mean {v.float().mean()}')


class Schedule(RuntimeMixin):

    def __init__(self, pipeline_stage: PipelineStage):
        assert isinstance(pipeline_stage, PipelineStage)
        self.pipeline_stage = pipeline_stage

    @property
    def num_chunks(self):
        return self.pipeline_stage.num_chunks

    @property
    def fw_node(self):
        return self.pipeline_stage.fw_node

    @property
    def bw_node(self):
        return self.pipeline_stage.bw_node

    @property
    def step_node(self):
        return self.pipeline_stage.step_node

    @abstractmethod
    def __call__(self) -> None:
        raise NotImplementedError

    def forward_send_one_chunk(self) -> List[dist.Work]:
        return self.pipeline_stage.forward_send_one_chunk()

    def forward_compute_one_chunk(self):
        return self.pipeline_stage.forward_compute_one_chunk()

    def forward_recv_one_chunk(self, wait=True) -> List[dist.Work]:
        return self.pipeline_stage.forward_recv_one_chunk(wait=wait)

    def backward_recv_one_chunk(self, wait=True) -> List[dist.Work]:
        return self.pipeline_stage.backward_recv_one_chunk(wait=wait)

    def backward_compute_one_chunk(self):
        return self.pipeline_stage.backward_compute_one_chunk()

    def backward_send_one_chunk(self) -> List[dist.Work]:
        return self.pipeline_stage.backward_send_one_chunk()

    def merge_and_assign_chunked_grads(self) -> Dict[str, Any]:
        return self.pipeline_stage.merge_and_assign_chunked_grads()

    def step(self):
        return self.pipeline_stage.step()

class ScheduleGPipe(Schedule):

    def __call__(self) -> None:

        all_send_reqs: List[dist.Work] = []

        # Forward all chunks
        for fw_chunk in range(self.num_chunks):
            self.forward_recv_one_chunk()
            self.forward_compute_one_chunk()
            all_send_reqs += self.forward_send_one_chunk()

        # Backward all chunks
        if self.bw_node is not None:
            for bwd_chunk in range(self.num_chunks):
                self.backward_recv_one_chunk()
                self.backward_compute_one_chunk()
                all_send_reqs += self.backward_send_one_chunk()

        for work in all_send_reqs:
            work.wait()

        self.merge_and_assign_chunked_grads()

        if self.step_node is not None:
            self.step()


class ScheduleDAPPLE(Schedule):

    def __init__(self, pipeline_stage: PipelineStage):
        super().__init__(pipeline_stage)
        assert pipeline_stage.bw_node is not None, f"{type(self).__name__} requires backward node"
        num_warmup = self.pipeline_stage.num_stages - self.pipeline_stage.stage_idx
        self.num_warmup = min(num_warmup, self.pipeline_stage.num_chunks)

    def __call__(self) -> None:
        all_send_reqs: List[dist.Work] = []

        # Warm-up phase: forward number of chunks equal to pipeline depth.
        for chunk in range(self.num_warmup):
            self.forward_recv_one_chunk()
            self.forward_compute_one_chunk()
            all_send_reqs += self.forward_send_one_chunk()

        # 1F1B phase
        for fwd_chunk in range(self.num_warmup, self.num_chunks):
            # recv backward first
            self.backward_recv_one_chunk()
            # IMPORTANT: recv forward after recv backward and before send backward
            reqs = self.forward_recv_one_chunk(wait=False)
            self.backward_compute_one_chunk()
            all_send_reqs += self.backward_send_one_chunk()
            for req in reqs:
                req.wait()
            self.forward_compute_one_chunk()
            all_send_reqs += self.forward_send_one_chunk()

        # Cool-down phase: backward for the rest of the chunks
        for bwd_chunk in range(self.num_chunks - self.num_warmup, self.num_chunks):
            self.backward_recv_one_chunk()
            self.backward_compute_one_chunk()
            all_send_reqs += self.backward_send_one_chunk()

        for work in all_send_reqs:
            work.wait()

        self.merge_and_assign_chunked_grads()

        if self.step_node is not None:
            self.step()
