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

from typing import List, Set

import torch
import torch.fx as fx


def get_partition(fx_module: fx.GraphModule) -> List[Set[str]]:
    partitions = []
    cur_par = set()
    for node in fx_module.graph.nodes:
        cur_par.add(node.name)
        if node.op == 'call_function' and node.target in [
            torch.ops.easydist.fw_bw_split.default,
            torch.ops.easydist.step_split.default
        ]:
            partitions.append(cur_par)
            cur_par = set()
    partitions.append(cur_par)
    return partitions
