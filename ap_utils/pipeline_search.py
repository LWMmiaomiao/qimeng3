import ap_utils.netlist as nl
import networkx as nx


def stage_assignment_align_with_old(netlist: nl.Netlist, depth: int):
    for node in netlist.get_topological_sort():
        netlist.graph.nodes[node]["stage"] = -1
    highest_level_old = {}
    levels = netlist.get_top_levels()
    max_level = max(levels.values())
    stage_delay = int(1.05 * max_level / depth)
    for io_id in range(netlist.PO_num):
        po_node = netlist.get_PO_from_io_id(io_id)
        stage = min(int(levels[po_node] / max_level * depth), depth - 1)
        netlist.set_stage(po_node, stage)
        highest_level_old[po_node] = (stage + 1) * stage_delay - 1
    for io_id in range(netlist.Latch_num):
        lo_node = netlist.get_LO_from_io_id(io_id)
        stage = min(int(levels[lo_node] / max_level * depth), depth - 1)
        netlist.set_stage(lo_node, stage)
        highest_level_old[lo_node] = (stage + 1) * stage_delay - 1
    if not netlist.reverse_topo_sort:
        netlist.reverse_topo_sort = list(nx.topological_sort(netlist.graph))
    for node in netlist.reverse_topo_sort:
        if netlist.get_type(node) in ["PI", "LI", "CONST0"]:
            continue
        if node not in highest_level_old:
            lowest_succ = min(
                [highest_level_old[succ] for succ in netlist.graph.successors(node)]
            )
            highest_level_old[node] = max(lowest_succ - 1, 1)
    for node in netlist.get_topological_sort():
        if netlist.get_type(node) == "AND":
            stage = min(highest_level_old[node] // stage_delay, depth - 1)
            netlist.set_stage(node, stage)
    netlist.n_stages = depth
    netlist.determine_ctrl_io_id()
