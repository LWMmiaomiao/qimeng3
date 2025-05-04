import networkx as nx
import copy


class Netlist:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.PI_num = 0
        self.PO_num = 0
        self.Latch_num = 0
        self.LI_id = {}
        self.LO_id = {}
        self.PI_id = {}
        self.PO_id = {}
        self.mux_list = []
        self.topo_sort = []
        self.reverse_topo_sort = []
        self.n_stages = 1
        self.ctrl_LI_io_id_table = []
        self.stage_MI_list = []
        self.stage_MO_list = []
        self.stage_PI_list = []
        self.stage_PO_list = []
        self.stage_LI_list = []
        self.stage_LO_list = []
        self.stage_stall_signal = []

    def reset(self):
        for node in self.topo_sort:
            self.graph.nodes[node]["value"] = -1

    def add_node(self, node_type, node_id):
        self.graph.add_node(node_id, type=node_type, stage=-1, value=-1)

    def set_io_id(self, node_id, io_id):
        node_type = self.graph.nodes[node_id]["type"]
        if node_type == "LI":
            self.LI_id[io_id] = node_id
        elif node_type == "LO":
            self.LO_id[io_id] = node_id
        elif node_type == "PI":
            self.PI_id[io_id] = node_id
        elif node_type == "PO":
            self.PO_id[io_id] = node_id

    # 增加了weight信息，为了测试增益函数
    def add_edge(self, src_id, dst_id, edge_type, weight=1):
        self.graph.add_edge(src_id, dst_id, type=edge_type, weight=weight)

    def get_predecessors(self, node_id):
        return list(self.graph.predecessors(node_id))

    def get_successors(self, node_id):
        return list(self.graph.successors(node_id))

    def nodes(self):
        return self.graph.nodes()

    def get_top_levels(self):
        # 计算拓扑级别
        top_levels = {}
        for node in nx.topological_sort(self.graph):
            top_levels[node] = max(
                (top_levels.get(pred, -1) + 1 for pred in self.get_predecessors(node)),
                default=0,
            )
        return top_levels

    def get_topological_sort(self):

        # 如果拓扑排序为空，或者你有一个机制标记图结构被修改过，则重新计算
        if not self.topo_sort:
            # 使用 NetworkX 的拓扑排序功能来获取排序
            self.topo_sort = list(nx.topological_sort(self.graph))
        return self.topo_sort

    def get_io_id_from_LI(self, node_id):
        for key, value in self.LI_id.items():
            if value == node_id:
                return key

    def get_LI_from_io_id(self, io_id):
        return self.LI_id[io_id]

    def get_io_id_from_LO(self, node_id):
        for key, value in self.LO_id.items():
            if value == node_id:
                return key

    def get_LO_from_io_id(self, io_id):
        return self.LO_id[io_id]

    def get_io_id_from_PI(self, node_id):
        for key, value in self.PI_id.items():
            if value == node_id:
                return key

    def get_PI_from_io_id(self, io_id):
        return self.PI_id[io_id]

    def get_io_id_from_PO(self, node_id):
        for key, value in self.PO_id.items():
            if value == node_id:
                return key

    def get_PO_from_io_id(self, io_id):
        return self.PO_id[io_id]

    def set_stage(self, node_id, stage):
        if (
            self.graph.nodes[node_id]["type"] == "AND"
            or self.graph.nodes[node_id]["type"] == "PO"
            or self.graph.nodes[node_id]["type"] == "LO"
        ):
            self.graph.nodes[node_id]["stage"] = stage
        else:
            raise ValueError("Cannot assign stage to nodes other than AND gate.")

    def get_stage(self, node_id):
        node_type = self.graph.nodes[node_id]["type"]
        if node_type == "AND" or node_type == "PO" or node_type == "LO":
            return self.graph.nodes[node_id]["stage"]
        elif node_type == "PI" or node_type == "LI":
            min_stage = min(
                [self.graph.nodes[n]["stage"] for n in self.graph.successors(node_id)]
            )
            return min_stage
        else:
            raise ValueError("Cannot get stage of CONST0 node.")

    def get_type(self, node_id):
        return self.graph.nodes[node_id]["type"]

    def detect_mux(self):
        for node in self.graph.nodes:
            if self.graph.nodes[node]["type"] == "AND":
                # Check if one input edge is DIRECT and the other is NOT
                pred_edge_types = [
                    self.graph.edges[pred, node]["type"]
                    for pred in self.graph.predecessors(node)
                ]
                succ_edge_types = [
                    self.graph.edges[node, succ]["type"]
                    for succ in self.graph.successors(node)
                ]
                # if all(_ == 'NOT' for _ in pred_edge_types) and all(_ == 'NOT' for _ in succ_edge_types):
                if all(_ == "NOT" for _ in pred_edge_types):
                    # Get the two input nodes
                    input_node_0 = list(self.graph.predecessors(node))[0]
                    input_node_1 = list(self.graph.predecessors(node))[1]
                    if (self.graph.nodes[input_node_0]["type"] == "AND") and (
                        self.graph.nodes[input_node_1]["type"] == "AND"
                    ):
                        if (
                            len(
                                set(self.graph.predecessors(input_node_0))
                                | set(self.graph.predecessors(input_node_1))
                            )
                            == 3
                        ):
                            common_pred = list(
                                set(self.graph.predecessors(input_node_0))
                                & set(self.graph.predecessors(input_node_1))
                            )[0]
                            if (
                                self.graph.edges[common_pred, input_node_0]["type"]
                                == "DIRECT"
                                and self.graph.edges[common_pred, input_node_1]["type"]
                                == "NOT"
                            ):
                                mux_dict = {
                                    "M": node,
                                    "C": common_pred,
                                    "X": input_node_0,
                                    "Y": input_node_1,
                                }
                                self.mux_list.append(mux_dict)
                            elif (
                                self.graph.edges[common_pred, input_node_0]["type"]
                                == "NOT"
                                and self.graph.edges[common_pred, input_node_1]["type"]
                                == "DIRECT"
                            ):
                                mux_dict = {
                                    "M": node,
                                    "C": common_pred,
                                    "X": input_node_1,
                                    "Y": input_node_0,
                                }
                                self.mux_list.append(mux_dict)

    def topo_and_reverse_topo_sort(self):
        self.topo_sort = list(nx.topological_sort(self.graph))
        self.reverse_topo_sort = list(reversed(self.topo_sort))

    def determine_ctrl_io_id(self):
        self.ctrl_LI_io_id_table = []
        for i in range(self.n_stages):
            io_id_list = []
            if i == 0:
                for io_id, LO_node_id in self.LO_id.items():
                    if self.get_stage(LO_node_id) == 0:
                        io_id_list.append(io_id)
                self.ctrl_LI_io_id_table.append(io_id_list)
            else:
                for io_id, LI_node_id in self.LI_id.items():
                    LO_node_id = self.get_LO_from_io_id(io_id)
                    max_LI_stage = max(
                        [
                            self.get_stage(succ)
                            for succ in self.graph.successors(LI_node_id)
                        ]
                    )
                    if max_LI_stage <= i:
                        io_id_list.append(io_id)
                    elif self.get_stage(LO_node_id) <= i:
                        io_id_list.append(io_id)
                self.ctrl_LI_io_id_table.append(io_id_list)

    def execute(self, PI_signal: list, LI_signal: list):
        self.reset()
        for node in self.topo_sort:
            node_type = self.graph.nodes[node]["type"]
            if node_type == "PI":
                PI_io_id = self.get_io_id_from_PI(node)
                self.graph.nodes[node]["value"] = PI_signal[PI_io_id]
            elif node_type == "LI":
                LI_io_id = self.get_io_id_from_LI(node)
                self.graph.nodes[node]["value"] = LI_signal[LI_io_id]
            elif node_type == "CONST0":
                self.graph.nodes[node]["value"] = 0
            elif node_type == "AND":
                preds = list(self.graph.predecessors(node))
                input_node = []
                input_value_list = []
                for p in preds:
                    input_edge_type = self.graph.edges[p, node]["type"]
                    if self.graph.nodes[p]["value"] == -1:
                        input_value_list.append(-1)
                    else:
                        p_value = self.graph.nodes[p]["value"]
                        input_value_list.append(
                            p_value if input_edge_type == "DIRECT" else int(not p_value)
                        )
                match input_value_list:
                    case [x, y] if 0 in [x, y]:
                        self.graph.nodes[node]["value"] = 0
                    case [1, 1]:
                        self.graph.nodes[node]["value"] = 1
                    case _:
                        self.graph.nodes[node]["value"] = -1
            elif node_type == "PO" or node_type == "LO":
                input_node = list(self.graph.predecessors(node))[0]
                input_edge_type = self.graph.edges[input_node, node]["type"]
                input_value = self.graph.nodes[input_node]["value"]
                if self.graph.nodes[input_node]["value"] == -1:
                    self.graph.nodes[node]["value"] = -1
                else:
                    self.graph.nodes[node]["value"] = (
                        input_value
                        if input_edge_type == "DIRECT"
                        else int(not input_value)
                    )
        PO_signal = [-1 for i in range(self.PO_num)]
        for k, v in self.PO_id.items():
            PO_signal[k] = self.graph.nodes[v]["value"]
        LO_signal = [-1 for i in range(self.Latch_num)]
        for k, v in self.LO_id.items():
            LO_signal[k] = self.graph.nodes[v]["value"]
        return PO_signal, LO_signal

    def pipeline_stage_execute(self, stage_id: int, MI: dict, PI: dict, LI: dict):
        MO = {key: -1 for key in self.stage_MO_list[stage_id]}
        PO = {key: -1 for key in self.stage_PO_list[stage_id]}
        LO = {key: -1 for key in self.stage_LO_list[stage_id]}
        self.reset()
        if self.stage_stall_signal[stage_id] == 0:
            for node in self.topo_sort:
                node_type = self.graph.nodes[node]["type"]
                if node_type == "PI":
                    PI_io_id = self.get_io_id_from_PI(node)
                    if PI_io_id in PI.keys():
                        self.graph.nodes[node]["value"] = PI[PI_io_id]
                elif node_type == "LI":
                    LI_io_id = self.get_io_id_from_LI(node)
                    if LI_io_id in LI.keys():
                        self.graph.nodes[node]["value"] = LI[LI_io_id]
                elif node_type == "CONST0":
                    self.graph.nodes[node]["value"] = 0
                elif node_type == "AND" and self.graph.nodes[node]["stage"] == stage_id:
                    preds = list(self.graph.predecessors(node))
                    input_node = []
                    input_value_list = []
                    for p in preds:
                        input_edge_type = self.graph.edges[p, node]["type"]
                        p_value = (
                            self.graph.nodes[p]["value"]
                            if self.graph.nodes[p]["stage"] in [stage_id, -1]
                            else MI[p]
                        )
                        assert p_value != -1
                        input_value_list.append(
                            p_value if input_edge_type == "DIRECT" else int(not p_value)
                        )
                    self.graph.nodes[node]["value"] = 1 if all(input_value_list) else 0
                elif (node_type == "PO" or node_type == "LO") and self.graph.nodes[
                    node
                ]["stage"] == stage_id:
                    input_node = list(self.graph.predecessors(node))[0]
                    input_value = (
                        self.graph.nodes[input_node]["value"]
                        if (
                            self.graph.nodes[input_node]["stage"] == stage_id
                            or self.graph.nodes[input_node]["stage"] == -1
                        )
                        else MI[input_node]
                    )
                    is_input_edge_direct = (
                        self.graph.edges[input_node, node]["type"] == "DIRECT"
                    )
                    assert input_value != -1
                    self.graph.nodes[node]["value"] = (
                        input_value if is_input_edge_direct else int(not input_value)
                    )
            for k in MO.keys():
                if k in MI.keys():
                    MO[k] = MI[k]
                else:
                    MO[k] = self.graph.nodes[k]["value"]
                assert MO[k] != -1
            pass
            for k in PO.keys():
                node = self.get_PO_from_io_id(k)
                PO[k] = self.graph.nodes[node]["value"]
            for k in LO.keys():
                node = self.get_LO_from_io_id(k)
                LO[k] = self.graph.nodes[node]["value"]
        return MO, PO, LO

    def calculate_stage_IO_info(self):
        self.cal_stage_MI_MO_list()
        self.stage_LI_list = []
        self.stage_LO_list = []
        self.stage_PI_list = []
        self.stage_PO_list = []
        for i in range(self.n_stages):
            self.stage_LI_list.append(self.cal_stage_LI_list(i))
            self.stage_LO_list.append(self.cal_stage_LO_list(i))
            self.stage_PI_list.append(self.cal_stage_PI_list(i))
            self.stage_PO_list.append(self.cal_stage_PO_list(i))

    def cal_stage_MI_MO_list(self):
        self.stage_MI_list = [[] for _ in range(self.n_stages)]

        self.stage_MO_list = [[] for _ in range(self.n_stages)]

        for node in self.topo_sort:
            if self.graph.nodes[node]["type"] == "AND":
                node_stage = self.graph.nodes[node]["stage"]
                max_succ_stage = max(
                    [
                        self.graph.nodes[succ]["stage"]
                        for succ in self.graph.successors(node)
                    ]
                )
                if max_succ_stage > node_stage:
                    for s in range(node_stage + 1, max_succ_stage + 1):
                        if s >= len(self.stage_MI_list):
                            self.stage_MI_list.extend(
                                [[] for _ in range(s - len(self.stage_MI_list) + 1)]
                            )
                        self.stage_MI_list[s].append(node)
        for i in range(0, self.n_stages - 1):
            self.stage_MO_list[i] = self.stage_MI_list[i + 1]

    def cal_stage_PI_list(self, stage_id: int):
        PI_id_list = []
        for node in self.topo_sort:
            if self.graph.nodes[node]["type"] == "PI":
                for succ in self.graph.successors(node):
                    if self.graph.nodes[succ]["stage"] == stage_id:
                        PI_id_list.append(self.get_io_id_from_PI(node))
                        break
        return PI_id_list

    def cal_stage_PO_list(self, stage_id: int):
        PO_id_list = []
        for node in self.topo_sort:
            if (
                self.graph.nodes[node]["type"] == "PO"
                and self.graph.nodes[node]["stage"] == stage_id
            ):
                PO_id_list.append(self.get_io_id_from_PO(node))
        return PO_id_list

    def cal_stage_LI_list(self, stage_id: int):
        LI_id_list = []
        for node in self.topo_sort:
            if self.graph.nodes[node]["type"] == "LI":
                for succ in self.graph.successors(node):
                    if self.graph.nodes[succ]["stage"] == stage_id:
                        LI_id_list.append(self.get_io_id_from_LI(node))
                        break
        LI_id_list.sort()
        return LI_id_list

    def cal_stage_LO_list(self, stage_id: int):
        LO_id_list = []
        for node in self.topo_sort:
            if (
                self.graph.nodes[node]["type"] == "LO"
                and self.graph.nodes[node]["stage"] == stage_id
            ):
                LO_id_list.append(self.get_io_id_from_LO(node))
        LO_id_list.sort()
        return LO_id_list

    def get_topological_sort(self):
        if not self.topo_sort:
            self.topo_sort = list(nx.topological_sort(self.graph))
        return self.topo_sort

    def is_(self):
        return nx.is_directed_acyclic_graph(self.graph)

    def __deepcopy__(self, memo):
        # 创建新的 Netlist 实例
        new_netlist = type(self)()

        # 使用 copy.deepcopy 递归地拷贝每一个属性
        new_netlist.graph = copy.deepcopy(self.graph, memo)
        new_netlist.PI_num = self.PI_num
        new_netlist.PO_num = self.PO_num
        new_netlist.Latch_num = self.Latch_num
        new_netlist.LI_id = copy.deepcopy(self.LI_id, memo)
        new_netlist.LO_id = copy.deepcopy(self.LO_id, memo)
        new_netlist.PI_id = copy.deepcopy(self.PI_id, memo)
        new_netlist.PO_id = copy.deepcopy(self.PO_id, memo)
        new_netlist.mux_list = copy.deepcopy(self.mux_list, memo)
        new_netlist.topo_sort = copy.deepcopy(self.topo_sort, memo)
        new_netlist.reverse_topo_sort = copy.deepcopy(self.reverse_topo_sort, memo)
        new_netlist.n_stages = self.n_stages
        new_netlist.ctrl_LI_io_id_table = copy.deepcopy(self.ctrl_LI_io_id_table, memo)
        new_netlist.stage_MI_list = copy.deepcopy(self.stage_MI_list, memo)
        new_netlist.stage_MO_list = copy.deepcopy(self.stage_MO_list, memo)
        new_netlist.stage_PI_list = copy.deepcopy(self.stage_PI_list, memo)
        new_netlist.stage_PO_list = copy.deepcopy(self.stage_PO_list, memo)
        new_netlist.stage_LI_list = copy.deepcopy(self.stage_LI_list, memo)
        new_netlist.stage_LO_list = copy.deepcopy(self.stage_LO_list, memo)
        new_netlist.stage_stall_signal = copy.deepcopy(self.stage_stall_signal, memo)

        return new_netlist
