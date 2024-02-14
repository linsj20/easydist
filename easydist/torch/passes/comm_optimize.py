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
import time
from functools import reduce

import torch
from torch.fx.node import _get_qualified_name

import easydist
import easydist.config as mdconfig
import easydist.torch.schedule.rcpsp as rcpsp
from easydist.torch.passes.sharding import create_meta_from_node
from easydist.torch.utils import EDInfo, EDNodeType

logger = logging.getLogger(__name__)


def bandwidth_profile():
    '''
    Currently get maximum bandwidth through communicating a large tensor(4096 * 1024 * 16)
    '''
    iter_time = 10
    comm_v = iter_time * 2 * 4096 * 1024 * 16 * 4.0
    res_t = 0.0
    for _ in range(0, iter_time):
        with torch.device('cuda'):
            t = torch.randn(4096, 1024, 16)
        torch.distributed.barrier()
        start_t = time.perf_counter()
        torch.distributed.all_reduce(t)
        torch.distributed.barrier()
        res_t += time.perf_counter() - start_t
    return comm_v / res_t


def rcpsp_schedule(fx_module: torch.fx.GraphModule, mem_constrain: bool):
    '''
    This function returns the best schedule executing given graph under rcpsp

    Args:
    fx_module: fx graph to be optimized
    mem_constrain: flag to turn on the memory constrain in scpsp model 

    Returns:
    An ordering of nodes
    '''

    # prepare RCPSP input
    task_data = []
    available_resources = {'comm': 1, 'comp': 1}
    if mem_constrain is True:
        available_resources['mem'] = int(0.95 * mdconfig.available_mem)

    # whether resource release only until all nodes depended on it have finished
    resource_dep_mask = [0, 0, 1]
    precedence_relations = []

    arg_num = 0
    arg_list = []
    for node in fx_module.graph.nodes:
        duration = node.ed_info.normalized_int_runtime_ms
        assert (duration > 0)
        if node.name.__contains__('arg'):
            arg_list.append(node)
            arg_num += 1
            continue

        resource = []

        priority = 0
        if node.ed_info.is_communication():
            priority = 1
            resource.append(('comm', 1))
            if mem_constrain is True:
                mem_req = int(node.ed_info.comm_meta['comm_vol'] / 1024)
        else:
            resource.append(('comp', 1))
            if mem_constrain is True:
                output_shapes = node.meta['val'].shape
                if isinstance(output_shapes, tuple):
                    output_shapes = list(output_shapes)
                elif not isinstance(output_shapes, list):
                    output_shapes = [output_shapes]
                mem_req = 0
                for output_shape in output_shapes:
                    if output_shape.get('shape') is not None:
                        mem_req += int(
                            reduce(lambda x, y: x * y, output_shape['shape'], 1) / 1024) # unit: dtype
        if mem_constrain is True:
            resource.append(('mem', mem_req))

        precedence = []
        for pre in node.all_input_nodes:
            if not pre.name.__contains__('arg'):
                precedence.append(pre)
        precedence_relations.append(precedence)

        task_data.append((node, duration, precedence, resource, priority))

    assert (len(task_data) == len(fx_module.graph.nodes) - arg_num)

    # only rank 0 process do the calculation
    if torch.distributed.get_rank() == 0:
        logger.info('enter rcpsp')
        logger.info(f'task cnt: {len(task_data)}')
        if mem_constrain is True:
            logger.info(f'[RCPSP]: Scheduling with Memory Constraint.')
        else:
            logger.info(f'[RCPSP]: Scheduling without Memory Constraint.')
        start_t = time.perf_counter()
        raw_sche = rcpsp.rcpsp(task_data, available_resources,
                               resource_dep_mask, mdconfig.rcpsp_method)
        logger.info(f"[RCPSP.time]:\t {time.perf_counter() - start_t} s.")
        logger.info('exit rcpsp')

        assert (len(raw_sche) == len(fx_module.graph.nodes) - arg_num)
    else:
        raw_sche = [None] * (len(fx_module.graph.nodes) - arg_num)
    torch.distributed.broadcast_object_list(raw_sche, src=0, device="cuda")

    node_sche = [task_data[i][0] for i in raw_sche]

    sche = arg_list + node_sche

    assert (len(sche) == len(fx_module.graph.nodes))

    return sche


group_ops = [
    'easydist.torch.passes.sharding.all_reduce_start',
    'easydist.torch.passes.sharding.all_gather_start',
    'easydist.torch.passes.sharding.reduce_scatter_start'
]


def comm_nodes_group(fx_module, node_list):
    '''
    Group the nodes in node_list
    '''

    if len(node_list) <= 1:
        return False
    # group the node and add proper decouple node to both the graph and the schedule
    sche = [node for node in fx_module.graph.nodes]
    from_nodes = [node.all_input_nodes[0] for node in node_list]
    to_nodes = [node.ed_info.comm_meta['to_node'] for node in node_list]
    total_size = 0

    comm_op = _get_qualified_name(node_list[0].target)
    comm_args = list(node_list[0].args[1:])
    if comm_op == 'easydist.torch.passes.sharding.all_reduce_start':
        retrive_points = []
        retrive_shapes = []
        for node in node_list:
            comm_vol = node.ed_info.comm_meta['comm_vol']
            comm_shape = node.ed_info.comm_meta['comm_shape']
            retrive_points.append(comm_vol)
            retrive_shapes.append(comm_shape)
            total_size += comm_vol
        decouple_args = [tuple(retrive_points), tuple(retrive_shapes)]

        def comm_couple(*tensor_list):
            flattened_tensor_list = [t.flatten() for t in tensor_list]
            return torch.cat(tuple(flattened_tensor_list))

        def comm_decouple(tensor, retrive_points, retrive_shapes):
            tensor_list = torch.split(tensor, retrive_points)
            return [tensor.reshape(shape) for tensor, shape in zip(tensor_list, retrive_shapes)]
    elif comm_op == 'easydist.torch.passes.sharding.all_gather_start':
        retrive_points = []
        retrive_shapes = []
        for node in node_list:
            comm_vol = node.ed_info.comm_meta['comm_vol']
            comm_shape = node.ed_info.comm_meta['comm_shape']
            retrive_points.append(comm_vol)
            retrive_shapes.append(comm_shape)
            total_size += comm_vol

        # force to gather on dim 0
        org_dim = comm_args[0]
        comm_args[0] = 0

        decouple_args = [total_size, org_dim, tuple(retrive_points), tuple(retrive_shapes)]

        def comm_couple(*tensor_list):
            flattened_tensor_list = [t.flatten() for t in tensor_list]
            return torch.cat(tuple(flattened_tensor_list))

        def comm_decouple(tensor, chunk_size, gather_dim, retrive_points, retrive_shapes):
            chunk_list = torch.split(tensor, chunk_size)
            tensor1d_lists = [torch.split(chunk, retrive_points) for chunk in chunk_list]
            tensor_lists = [[tensor1d.reshape(shape) for tensor1d, shape in 
                            zip(tensor1d_list, retrive_shapes)] for tensor1d_list in tensor1d_lists]
            return [torch.cat([tensor_list[i] for tensor_list in 
                               tensor_lists], gather_dim) for i in range(len(tensor_lists[0]))]

    elif comm_op == 'easydist.torch.passes.sharding.reduce_scatter_start':
        # transpose + view
        # TODO how to pass pp process group?
        split_size = torch.distributed.get_world_size()
        scatter_dim = comm_args[1]
        
        retrive_points = []
        retrive_shapes = []
        for node in node_list:
            comm_vol = node.ed_info.comm_meta['comm_vol']
            comm_shape = list(node.ed_info.comm_meta['comm_shape'])
            retrive_points.append(int(comm_vol / split_size))
            comm_shape[scatter_dim] = int(comm_shape[scatter_dim] / split_size)
            retrive_shapes.append(torch.Size(comm_shape))

        decouple_args = [tuple(retrive_points), tuple(retrive_shapes)]

        def comm_couple(*tensor_list):
            chunked_tensor = [t.chunk(split_size, dim=scatter_dim) for t in tensor_list]
            flattened_tensor_lists = [[t.flatten() for t in tensor_list] for 
                                      tensor_list in chunked_tensor]
            new_chunked_tensor = [torch.cat([flattened_tensor_list[i] for flattened_tensor_list in
                                              flattened_tensor_lists]) for i in 
                                              range(len(flattened_tensor_lists[0]))]
            return torch.cat(new_chunked_tensor)

        def comm_decouple(tensor, retrive_points, retrive_shapes):
            tensor_list = torch.split(tensor, retrive_points)
            return [tensor.reshape(shape) for tensor, shape in zip(tensor_list, retrive_shapes)]
    else:
        raise RuntimeError('Fusion: unrecognized communication type')

    to_node = sche[min([sche.index(to_node) for to_node in to_nodes])]

    #with fx_module.graph.inserting_after(node_list[-1]):
    with fx_module.graph.inserting_before(node_list[0]):
        new_from_node = fx_module.graph.call_function(comm_couple, args=tuple(from_nodes))
        new_from_node.meta = create_meta_from_node(new_from_node)
        new_from_node.ed_info = EDInfo()
        new_from_node.ed_info.node_type = EDNodeType.COMPUTATION

    with fx_module.graph.inserting_after(new_from_node):
        new_comm_node = fx_module.graph.call_function(eval(comm_op),
                                                      args=tuple([new_from_node] + comm_args))
        new_comm_node.meta = create_meta_from_node(new_comm_node)

    with fx_module.graph.inserting_before(to_node):
        new_to_node = fx_module.graph.call_function(comm_decouple,
                                                    args=tuple([new_comm_node] + decouple_args))
                                                    #args=(new_comm_node, tuple(retrive_points),
                                                    #      tuple(retrive_shapes)))
        #new_to_node.meta = create_meta_from_node(new_to_node)
        new_to_node.ed_info = EDInfo()
        new_to_node.ed_info.node_type = EDNodeType.COMPUTATION

    new_comm_node.ed_info = EDInfo()
    new_comm_node.ed_info.node_type = EDNodeType.COMMUNICATION
    new_comm_node.ed_info.comm_meta = {
        'to_node': new_to_node,
        'comm_vol': total_size,
        'comm_shape': torch.Size([total_size])
    }

    for idx, (comm_node, to_node) in enumerate(zip(node_list, to_nodes)):
        with fx_module.graph.inserting_before(to_node):
            retrive_node = fx_module.graph.call_function(operator.getitem, args=(new_to_node, idx))
        to_node.replace_input_with(comm_node.ed_info.comm_meta['end_node'], retrive_node)
        #retrive_node.meta = create_meta_from_node(retrive_node)
        retrive_node.ed_info = EDInfo()
        retrive_node.ed_info.node_type = EDNodeType.COMPUTATION

    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()

    return True


def is_to_group(n):
    n_op_name = _get_qualified_name(n.target)
    if n_op_name in group_ops:
        return True
    return False


def groupable(n1, n2):
    n1_op_name = _get_qualified_name(n1.target)
    n2_op_name = _get_qualified_name(n2.target)
    return n1_op_name == n2_op_name and n1.args[1:] == n2.args[1:]


def comm_group_(fx_module):
    if torch.distributed.get_rank() == 0:
        logger.info("Perform communication fusion...")
    sche = [node for node in fx_module.graph.nodes]
    nodes_to_be_group = []
    lbound = rbound = -1
    retrive_node = None

    # assuming one input for each light node
    light_node_types = ['__arg__', '_operator.getitem', 'torch.ops.aten.view.default', 'torch.ops.aten.t.default']
    def if_heavy_computation_of_comm_node_late_than_lbound(sche, node, lbound):
        comp_node = node.all_input_nodes[0]
        node_name = '__arg__' if comp_node.name.__contains__('arg') else _get_qualified_name(comp_node.target)
        light_node_list = []
        while node_name in light_node_types:
            if lbound > sche.index(comp_node):
                break
            light_node_list.append(comp_node)
            comp_node = comp_node.all_input_nodes[0]
            node_name = '__arg__' if comp_node.name.__contains__('arg') else _get_qualified_name(comp_node.target)
            #node_name = _get_qualified_name(comp_node.target)

        comp_idx = sche.index(comp_node)
        if lbound < comp_idx:
            return True
        
        if len(light_node_list) > 0:
            
            light_node_list.reverse()

            for light_node in light_node_list:
                sche.remove(light_node)

            for idx, light_node in enumerate(light_node_list):
                sche.insert(comp_idx + 1 + idx, light_node)
        
        return False

    fused_comm = 0
    output_comm = 0
    idx = 0
    while idx < len(sche):
        node = sche[idx]
        if node.ed_info.is_communication():
            if is_to_group(node):
                if len(nodes_to_be_group) == 0:
                    nodes_to_be_group.append(node)
                    lbound = idx
                    rbound = sche.index(node.ed_info.comm_meta['to_node'])
                else:
                    if groupable(node, nodes_to_be_group[0]):
                        if idx > rbound or if_heavy_computation_of_comm_node_late_than_lbound(sche, node, lbound):
                            _link_nodes(fx_module, sche)
                            if comm_nodes_group(fx_module, nodes_to_be_group):
                                fused_comm += len(nodes_to_be_group)
                                output_comm += 1
                                sche = [node for node in fx_module.graph.nodes]
                            nodes_to_be_group = []
                            idx = sche.index(retrive_node) if retrive_node is not None else sche.index(node)
                            retrive_node = None
                            continue
                        else:
                            nodes_to_be_group.append(node)
                            rbound = min(rbound, sche.index(node.ed_info.comm_meta['to_node']))
                    elif retrive_node is None:
                        retrive_node = node

        if idx + 1 == len(sche) and len(nodes_to_be_group) > 1:
            _link_nodes(fx_module, sche)
            if comm_nodes_group(fx_module, nodes_to_be_group):
                fused_comm += len(nodes_to_be_group)
                output_comm += 1
                sche = [node for node in fx_module.graph.nodes]
            nodes_to_be_group = []
            if retrive_node is not None:
                idx = sche.index(retrive_node) 
                retrive_node = None
                continue
        idx += 1
    if torch.distributed.get_rank() == 0:
        logger.info(f"Fuse {fused_comm} comms into {output_comm} comms.")
    return fx_module


def comm_group(fx_module, cap_limit, rg_limit):
    '''
    This function performs grouping on a fx graph

    Scan reversely searching for small comms and grouped current selected
    nodes when either dependencies or capacity limit is to be violated.

    Args:
    fx_module: fx graph to be optimized
    cap_limit:
    rg_limit: search range

    Returns:
    A grouped fx_module
    '''
    if torch.distributed.get_rank() == 0:
        logger.info("Perform communication fusion...")

    sche = [node for node in fx_module.graph.nodes]
    #idx = len(sche) - 1
    idx = 0
    num_nodes = len(sche)
    cur_cap = 0
    cur_range = 0
    cur_comm_list = []
    comm_list_dep = []
    retrive_node = None
    #while idx >= 0:
    fused_comm = 0
    output_comm = 0
    while idx < len(sche) or retrive_node is not None:
        cur_range += 1

        if idx == len(sche) and retrive_node is not None:
            idx = sche.index(retrive_node)
            retrive_node = None
        #if (not sche[idx].ed_info.is_communication() and sche[idx] in comm_list_dep) \
        if sche[idx] in comm_list_dep \
            or cur_range > rg_limit \
            or cur_cap > cap_limit:

            #cur_comm_list.reverse()
            if comm_nodes_group(fx_module, cur_comm_list):
                fused_comm += len(cur_comm_list)
                output_comm += 1
            sche = [node for node in fx_module.graph.nodes]

            cur_cap = 0
            cur_range = 0
            cur_comm_list = []
            comm_list_dep = []
            if retrive_node:
                idx = sche.index(retrive_node)
                retrive_node = None
                continue

        if not sche[idx].ed_info.is_communication():
            idx += 1
            continue

        node = sche[idx]
        comm_vol = node.ed_info.comm_meta['comm_vol']

        if comm_vol < cap_limit and is_to_group(node):
            if len(cur_comm_list) == 0 or \
                groupable(node, cur_comm_list[0]):
                cur_cap += comm_vol
                del sche[idx]
                cur_comm_list.append(node)
                comm_list_dep.append(node.ed_info.comm_meta['to_node'])
            elif retrive_node is None:
                retrive_node = node

        idx += 1
    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()
    if torch.distributed.get_rank() == 0:
        logger.info(f"Fuse {fused_comm} comms into {output_comm} comms.")
    return fx_module


def comm_optimize(fx_module: torch.fx.GraphModule,
                  sche_method,
                  grouping=False,
                  mem_restrain=False):
    '''
    This function performs multiple communciation optimizations on graph level

    Args:
    fx_module: fx graph to be optimized
    grouping: whether or not grouping is to be performed
    mem_restrain: whether or not mem_restrain is added to rcpsp

    Returns:
    A transformed fx_module with communication optimizations applied
    '''
    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()

    if mdconfig.log_level <= logging.DEBUG:
        fx_module.print_readable()

    #exit(1)
    # collect necessary communication node info, save at comm_meta in node.ed_info
    for node in fx_module.graph.nodes:
        if node.name.__contains__('fused_adam'):
            node.ed_info.node_type = EDNodeType.COMPUTATION
        if node.ed_info.is_communication():
            assert len(node.all_input_nodes) == 1
            from_node = node.all_input_nodes[0]
            comm_shape = from_node.meta['val'].shape
            # TODO support mixed precision
            node.ed_info.comm_meta = {
                'comm_vol': reduce(lambda x, y: x * y, comm_shape, 1),  #unit: dtype
                'comm_shape': comm_shape
            }
        else:
            for pre in node.all_input_nodes:
                if hasattr(pre, 'ed_info') and\
                    pre.ed_info.is_communication():
                    pre.ed_info.comm_meta['end_node'] = node
                elif len(pre.all_input_nodes) > 0:
                    for prepre in pre.all_input_nodes:
                        if hasattr(prepre, 'ed_info') and\
                            prepre.ed_info.is_communication():
                            prepre.ed_info.comm_meta['to_node'] = node


    # if torch.distributed.get_rank() == 0:
    #     for node in fx_module.graph.nodes:
    #         if node.ed_info.is_communication():
    #             print(node.ed_info.comm_meta['to_node'])
    #     print('000')
    #     fx_module.print_readable()
    #     print('000')
    #     for n in fx_module.graph.nodes:
    #         print(n.ed_info)

    # comm_map: node just computed -> commnications followed
    comm_map = {}
    if sche_method == 'eager':
        for node in fx_module.graph.nodes:
            if node.ed_info.is_communication():
                if comm_map.get(from_node) is None:
                    comm_map[from_node] = []
                comm_map[from_node].append(node)
    elif sche_method == 'rcpsp':
        sche = rcpsp_schedule(fx_module, mem_restrain)

        # reserve the placeholder order
        sche = [node for node in sche if node.op != "placeholder"]
        sche = [node for node in fx_module.graph.nodes if node.op == "placeholder"] + sche

        _link_nodes(fx_module, sche)

        '''
        if torch.distributed.get_rank() == 0:
            print('origin')
            fx_module.print_readable()
            for n in fx_module.graph.nodes:
                print('!')
                print(n.name)
                print(n.ed_info)
        '''

        grouping = True
        if grouping:
            #fx_module = comm_group(fx_module, 3 * 1024 * 1024 * 1024, 10000)
            fx_module = comm_group_(fx_module)

        '''
        if torch.distributed.get_rank() == 0:
            print('transformed')
            fx_module.print_readable()
        '''

        sche = [node for node in fx_module.graph.nodes]
        for idx, node in enumerate(sche):
            if not node.ed_info.is_communication() and \
                idx + 1 < len(sche) and \
                sche[idx + 1].ed_info.is_communication():
                comm_map[node] = []
                for follower in sche[idx + 1:]:
                    if follower.ed_info.is_communication():
                        comm_map[node].append(follower)
                    else:
                        break
                assert (len(comm_map[node]) > 0)

    # if torch.distributed.get_rank() == 0:
    #     fx_module.print_readable()
    #     print('000')
    #    for n in fx_module.graph.nodes:
    #        print(n.ed_info)


    def grouped_comm(input_tensors: list, comm_func: list, comm_args: list):
        res = []
        for input_tensor, comm_func, args in zip(input_tensors, comm_func, comm_args):
            res.append(eval(comm_func)(input_tensor, *args))
        return res

    # add after nodes followed by comms a grouped comm node
    for node in comm_map:
        if len(comm_map[node]) <= 1:
            continue

        input_nodes = [n.all_input_nodes[0] for n in comm_map[node]]
        comm_funcs = [_get_qualified_name(n.target) for n in comm_map[node]]
        comm_args = [n.args[1:] for n in comm_map[node]]

        # add grouped comm node
        with fx_module.graph.inserting_after(node):
            new_comm_node = fx_module.graph.call_function(grouped_comm,
                                                          args=(input_nodes, comm_funcs,
                                                                comm_args))

        # add retrive node
        for idx, comm_node in enumerate(comm_map[node]):
            with fx_module.graph.inserting_after(new_comm_node):
                idx_node = fx_module.graph.call_function(operator.getitem,
                                                         args=(new_comm_node, idx))
            comm_node.ed_info.comm_meta['to_node'].replace_input_with(comm_node, idx_node)

    # at this point all old comm operators should be eliminated
    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()

    if torch.distributed.get_rank() == 0:
        logger.info("Communication Optimization: Done!")
    return fx_module


def _link_nodes(fx_module, node_list):
    '''
    Change the topological order of fx_module according to node_list
    '''

    fx_module.graph._root._next = node_list[0]
    node_list[0]._prev = fx_module.graph._root
    for idx, node in enumerate(node_list[:-1]):
        node._next = node_list[idx + 1]
        node_list[idx + 1]._prev = node
    node_list[-1]._next = fx_module.graph._root
    fx_module.graph._root._prev = node_list[-1]
    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()
