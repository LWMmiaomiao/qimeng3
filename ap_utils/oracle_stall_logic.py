import ap_utils.netlist as nl


def execute_once(netlist: nl.Netlist, PI_dict: dict, LI_dict: dict):
    for value in PI_dict.values():
        if value != 0 and value != 1:
            raise ValueError("The value of PI dictionary is not 0 or 1.")
    for value in LI_dict.values():
        # print("LI_dict values:", LI_dict.values())
        if value != 0 and value != 1:
            raise ValueError("The value of LI dictionary is not 0 or 1.")
    for node in netlist.topo_sort:
        node_type = netlist.graph.nodes[node]["type"]
        if node_type == "PI":
            PI_io_id = netlist.get_io_id_from_PI(node)
            if PI_io_id in PI_dict.keys():
                netlist.graph.nodes[node]["value"] = PI_dict[PI_io_id]
            else:
                netlist.graph.nodes[node]["value"] = -1
        elif node_type == "LI":
            LI_io_id = netlist.get_io_id_from_LI(node)
            if LI_io_id in LI_dict.keys():
                netlist.graph.nodes[node]["value"] = LI_dict[LI_io_id]
            else:
                netlist.graph.nodes[node]["value"] = -1
        elif node_type == "CONST0":
            netlist.graph.nodes[node]["value"] = 0
        elif node_type == "AND":
            preds = list(netlist.graph.predecessors(node))
            input_value_list = []
            for p in preds:
                input_edge_type = netlist.graph.edges[p, node]["type"]
                if netlist.graph.nodes[p]["value"] == -1:
                    input_value_list.append(-1)
                else:
                    p_value = netlist.graph.nodes[p]["value"]
                    input_value_list.append(
                        p_value if input_edge_type == "DIRECT" else int(not p_value)
                    )
            match input_value_list:
                case [x, y] if 0 in [x, y]:
                    netlist.graph.nodes[node]["value"] = 0
                case [1, 1]:
                    netlist.graph.nodes[node]["value"] = 1
                case _:
                    netlist.graph.nodes[node]["value"] = -1
        elif node_type == "PO" or node_type == "LO":
            input_node = list(netlist.graph.predecessors(node))[0]
            input_edge_type = netlist.graph.edges[input_node, node]["type"]
            input_value = netlist.graph.nodes[input_node]["value"]
            if netlist.graph.nodes[input_node]["value"] == -1:
                netlist.graph.nodes[node]["value"] = -1
            else:
                netlist.graph.nodes[node]["value"] = (
                    input_value if input_edge_type == "DIRECT" else int(not input_value)
                )


def path_pruning(netlist: nl.Netlist, PI_dict: dict, LI_dict: dict):
    removed_node = []
    removed_edge = {node: set() for node in netlist.graph.nodes}
    for value in PI_dict.values():
        if value != 0 and value != 1:
            raise ValueError("The value of PI dictionary is not 0 or 1.")
    for value in LI_dict.values():
        # print("LI_dict values:", LI_dict.values())
        if value != 0 and value != 1:

            raise ValueError("The value of LI dictionary is not 0 or 1.")
    for node in netlist.topo_sort:
        node_type = netlist.graph.nodes[node]["type"]
        if node_type == "PI":
            PI_io_id = netlist.get_io_id_from_PI(node)
            if PI_io_id in PI_dict.keys():
                netlist.graph.nodes[node]["value"] = PI_dict[PI_io_id]
            else:
                netlist.graph.nodes[node]["value"] = -1
        elif node_type == "LI":
            LI_io_id = netlist.get_io_id_from_LI(node)
            if LI_io_id in LI_dict.keys():
                netlist.graph.nodes[node]["value"] = LI_dict[LI_io_id]
            else:
                netlist.graph.nodes[node]["value"] = -1
        elif node_type == "CONST0":
            netlist.graph.nodes[node]["value"] = 0
        elif node_type == "AND":
            preds = list(netlist.graph.predecessors(node))
            input_value_list = []
            for p in preds:
                input_edge_type = netlist.graph.edges[p, node]["type"]
                if netlist.graph.nodes[p]["value"] == -1:
                    input_value_list.append(-1)
                else:
                    p_value = netlist.graph.nodes[p]["value"]
                    input_value_list.append(
                        p_value if input_edge_type == "DIRECT" else int(not p_value)
                    )
            match input_value_list:
                case [x, y] if 0 in [x, y]:
                    netlist.graph.nodes[node]["value"] = 0
                    match input_value_list:
                        case [0, -1] | [0, 1]:
                            # case [0, _]:
                            removed_edge[preds[1]].add(node)
                        case [-1, 0] | [1, 0]:
                            # case [_, 0]:
                            removed_edge[preds[0]].add(node)
                        case [0, 0]:
                            removed_edge[preds[0]].add(node)
                            removed_edge[preds[1]].add(node)
                case [1, 1]:
                    netlist.graph.nodes[node]["value"] = 1
                case _:
                    netlist.graph.nodes[node]["value"] = -1
        elif node_type == "PO" or node_type == "LO":
            input_node = list(netlist.graph.predecessors(node))[0]
            input_edge_type = netlist.graph.edges[input_node, node]["type"]
            input_value = netlist.graph.nodes[input_node]["value"]
            if netlist.graph.nodes[input_node]["value"] == -1:
                netlist.graph.nodes[node]["value"] = -1
            else:
                netlist.graph.nodes[node]["value"] = (
                    input_value if input_edge_type == "DIRECT" else int(not input_value)
                )
    for node in netlist.reverse_topo_sort:
        if len(removed_edge[node]) == netlist.graph.out_degree(
            node
        ) and netlist.graph.nodes[node]["type"] not in [
            "PO",
            "LO",
        ]:
            removed_node.append(node)
            for p in netlist.graph.predecessors(node):
                removed_edge[p].add(node)
    return removed_node


def LI_stages_of_stage_0(netlist: nl.Netlist, PI_dict: dict, LI_dict: dict):
    removed_node = path_pruning(netlist, PI_dict, LI_dict)
    LI_stages = [0 for i in range(len(netlist.LI_id))]
    for io_id, node_id in netlist.LI_id.items():
        if node_id not in removed_node:
            # 新修复的一个bug：如果LI节点后续节点的值已知或者被删掉，其实说明LI的输入流水级不用考虑这个后续
            min_stage = 65535
            for succ in netlist.graph.successors(node_id):
                if (
                    netlist.graph.nodes[succ]["value"] == -1
                    and succ not in removed_node
                    and netlist.get_stage(succ) < min_stage
                ):
                    min_stage = netlist.get_stage(succ)
            LI_stages[io_id] = min_stage
            # LI_stages[io_id] = netlist.get_stage(node_id)
        else:
            LI_stages[io_id] = 65535
    return LI_stages


def is_straight_through(
    netlist: nl.Netlist,
    LI_node_id: int,
    LO_node_id: int,
    stall_signals: list,
    stage_id: int,
    lo_stages: list,
):
    graph = netlist.graph
    is_straight_through = False

    io_id = netlist.get_io_id_from_LI(LI_node_id)
    forward_stage = 0
    for s in range(1, netlist.n_stages):
        if (
            io_id not in netlist.ctrl_LI_io_id_table[s - 1]
            and io_id in netlist.ctrl_LI_io_id_table[s]
        ):
            forward_stage = s
            break
    has_previous_new_value = False
    for i in range(0, netlist.get_stage(LO_node_id) - forward_stage):
        s = stage_id + i
        if s < netlist.n_stages - 1 and stall_signals[s] == 0 and lo_stages[s] != -1:
            has_previous_new_value = True
            break
    if not has_previous_new_value:
        if graph.nodes[LI_node_id]["value"] != -1:
            is_straight_through = (
                graph.nodes[LI_node_id]["value"] == graph.nodes[LO_node_id]["value"]
            )
        else:
            for i in range(2):
                changed_node = {}
                changed_node[LI_node_id] = graph.nodes[LI_node_id]["value"]
                graph.nodes[LI_node_id]["value"] = i
                LO_value = graph.nodes[LO_node_id]["value"]
                for node in netlist.topo_sort:
                    if graph.nodes[node]["value"] == -1:
                        preds = list(graph.predecessors(node))
                        if any(p in changed_node.keys() for p in preds):
                            node_type = graph.nodes[node]["type"]
                            if node_type == "AND":
                                input_value = []
                                for p in preds:
                                    if graph.nodes[p]["value"] == -1:
                                        input_value.append(-1)
                                    else:
                                        input_edge_type = graph.edges[p, node]["type"]
                                        p_value = graph.nodes[p]["value"]
                                        input_value.append(
                                            p_value
                                            if input_edge_type == "DIRECT"
                                            else int(not p_value)
                                        )
                                match input_value:
                                    case [x, y] if 0 in [x, y]:
                                        changed_node[node] = graph.nodes[node]["value"]
                                        graph.nodes[node]["value"] = 0
                                    case [1, 1]:
                                        changed_node[node] = graph.nodes[node]["value"]
                                        graph.nodes[node]["value"] = 1
                            elif node == LO_node_id:
                                input_node = preds[0]
                                input_edge_type = graph.edges[input_node, node]["type"]
                                input_value = graph.nodes[input_node]["value"]
                                assert input_value != -1
                                LO_value = (
                                    input_value
                                    if input_edge_type == "DIRECT"
                                    else int(not input_value)
                                )
                                break
                for node, original_value in changed_node.items():
                    graph.nodes[node]["value"] = original_value
                if LO_value != i:
                    break
            else:
                is_straight_through = True
    return is_straight_through


def LO_stages_of_stage_i(
    netlist: nl.Netlist,
    PI_dict: dict,
    LI_dict: dict,
    stall_signals: list,
    stage_id: int,
    in_pipeline_LO_stages_list: list,
):
    execute_once(netlist, PI_dict, LI_dict)
    LO_stages = [0 for i in range(len(netlist.LI_id))]
    for io_id, LO_node_id in netlist.LO_id.items():
        in_pipeline_lo_stages = [lo_s[io_id] for lo_s in in_pipeline_LO_stages_list]
        LI_node_id = netlist.get_LI_from_io_id(io_id)
        is_straight = is_straight_through(
            netlist,
            LI_node_id,
            LO_node_id,
            stall_signals,
            stage_id,
            in_pipeline_lo_stages,
        )
        if is_straight:
            LO_stages[io_id] = -1
        else:
            LO_stages[io_id] = netlist.get_stage(LO_node_id)
    return LO_stages


def oracle_stall_logic(input: list, netlist: nl.Netlist):
    PI_num = len(netlist.PI_id)
    LI_num = [len(netlist.ctrl_LI_io_id_table[s]) for s in range(netlist.n_stages)]
    stall_signals = input[0 : netlist.n_stages - 1]
    PI_signals = [
        input[
            netlist.n_stages - 1 + s * PI_num : netlist.n_stages - 1 + (s + 1) * PI_num
        ]
        for s in range(netlist.n_stages)
    ]
    LI_signals = [
        input[
            netlist.n_stages * (1 + PI_num)
            - 1
            + sum(LI_num[0:s]) : netlist.n_stages * (1 + PI_num)
            - 1
            + sum(LI_num[0 : s + 1])
        ]
        for s in range(netlist.n_stages)
    ]
    PI_dict_list = [
        {i: v for i, v in enumerate(PI_signals[s])} for s in range(netlist.n_stages)
    ]
    LI_dict_list = [
        {netlist.ctrl_LI_io_id_table[s][i]: v for i, v in enumerate(LI_signals[s])}
        for s in range(netlist.n_stages)
    ]
    LI_stages = LI_stages_of_stage_0(netlist, PI_dict_list[0], LI_dict_list[0])
    LO_stages = [
        [65536 for _ in range(netlist.Latch_num)] for _ in range(netlist.n_stages - 1)
    ]
    for s in range(netlist.n_stages - 1, 0, -1):
        LO_stages[s - 1] = (
            LO_stages_of_stage_i(
                netlist, PI_dict_list[s], LI_dict_list[s], stall_signals, s, LO_stages
            )
            if stall_signals[s - 1] == 0
            else [65536 for _ in range(netlist.Latch_num)]
        )
    is_hazard_list = [1 for s in range(netlist.n_stages - 1)]
    for i in range(netlist.n_stages - 1):
        if stall_signals[i] == 1:
            is_hazard_list[i] = 0
        else:
            hazard = False
            for io_id in range(len(netlist.LI_id)):
                if LI_stages[io_id] < LO_stages[i][io_id] - i:
                    hazard = True
                    break
            is_hazard_list[i] = 1 if hazard else 0
    if all(h == 0 for h in is_hazard_list):
        return 0
    else:
        return 1
