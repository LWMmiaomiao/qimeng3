import json
import pickle
import random
import ap_utils.netlist as nl


def netlist_verilog(
    netlist: nl.Netlist,
    clock_name: str,
    rst_name: str,
    rstn_name: str,
    top_name: str = "top",
):
    verilog = []
    assert (
        rst_name == None and rstn_name != None or rst_name != None and rstn_name == None
    )
    is_rst = rst_name != None
    # module头
    verilog.append(f"module {top_name} (\n")
    verilog.append(f"  input {clock_name},\n")
    if is_rst:
        verilog.append(f"  input {rst_name},\n")
    else:
        verilog.append(f"  input {rstn_name},\n")
    for i in range(netlist.PI_num):
        verilog.append(f"  input n{netlist.PI_id[i]},\n")
    for i in range(netlist.PO_num):
        verilog.append(f"  output n{netlist.PO_id[i]},\n")
    verilog.append("  output ready,\n")
    verilog.append("  output done\n);\n")
    for i in range(netlist.n_stages):
        verilog.append(f"  wire [{len(netlist.stage_LI_list[i])-1}:0] this_li_{i};\n")
        verilog.append(f"  wire [{len(netlist.stage_LO_list[i])-1}:0] this_lo_{i};\n")
        verilog.append(f"  wire [{len(netlist.stage_MI_list[i])-1}:0] this_mi_{i};\n")
        verilog.append(f"  wire [{len(netlist.stage_MO_list[i])-1}:0] this_mo_{i};\n")
        verilog.append(f"  wire [{len(netlist.stage_PI_list[i])-1}:0] this_pi_{i};\n")
        verilog.append(f"  wire [{len(netlist.stage_PO_list[i])-1}:0] this_po_{i};\n")

    for i in range(netlist.n_stages - 1):
        verilog.append(f"    reg [{len(netlist.stage_MO_list[i])-1}:0] m{i};\n")
    for i in range(netlist.n_stages):
        verilog.appedn(f"  reg [{netlist.Latch_num-1}:0] li{i};\n")
        verilog.append(f"  reg [{netlist.PI_num-1}:0] pi{i};\n")
        verilog.append(f"  reg [{netlist.PO_num-1}:0] po{i};\n")
    verilog.append(f"  reg [{netlist.n_stages-1}:0] stall;\n")
    for i in range(netlist.n_stages):
        verilog.append(f"  assign this_li_{i} = li{i};\n")
        verilog.append(f"  wire [{len(netlist.stage_LO_list[i])-1}:0] this_lo_{i};\n")
        verilog.append(f"  wire [{len(netlist.stage_MI_list[i])-1}:0] this_mi_{i};\n")
        verilog.append(f"  wire [{len(netlist.stage_MO_list[i])-1}:0] this_mo_{i};\n")
        verilog.append(f"  wire [{len(netlist.stage_PI_list[i])-1}:0] this_pi_{i};\n")
        verilog.append(f"  wire [{len(netlist.stage_PO_list[i])-1}:0] this_po_{i};\n")
    pass


def stage_verilog(netlist: nl.Netlist, stage_id: int):
    verilog = []
    verilog.append(f"module stage_{stage_id} (\n")
    io_list = []
    if netlist.stage_LI_list[stage_id]:
        io_list.append(f"  input [{len(netlist.stage_LI_list[stage_id])-1}:0] li")
    if netlist.stage_MI_list[stage_id]:
        io_list.append(f"  input [{len(netlist.stage_MI_list[stage_id])-1}:0] mi")
    if netlist.stage_PI_list[stage_id]:
        io_list.append(f"  input [{len(netlist.stage_PI_list[stage_id])-1}:0] pi")
    if netlist.stage_LO_list[stage_id]:
        io_list.append(f"  output [{len(netlist.stage_LO_list[stage_id])-1}:0] lo")
    if netlist.stage_MO_list[stage_id]:
        io_list.append(f"  output [{len(netlist.stage_MO_list[stage_id])-1}:0] mo")
    if netlist.stage_PO_list[stage_id]:
        io_list.append(f"  output [{len(netlist.stage_PO_list[stage_id])-1}:0] po")
    verilog.append(",\n".join(io_list))
    verilog.append(f"\n  );\n")
    for node in netlist.topo_sort:
        node_type = netlist.graph.nodes[node]["type"]
        node_stage = netlist.graph.nodes[node]["stage"]
        if node_type == "AND" and node_stage == stage_id:
            preds = list(netlist.graph.predecessors(node))
            input_node = []
            for p in preds:
                input_edge_type = netlist.graph.edges[p, node]["type"]
                p_type = netlist.graph.nodes[p]["type"]
                p_stage = netlist.graph.nodes[p]["stage"]
                if p_type == "LI":
                    li_io_id = netlist.get_io_id_from_LI(p)
                    p_name = f"li[{netlist.stage_LI_list[stage_id].index(li_io_id)}]"
                if p_type == "PI":
                    pi_io_id = netlist.get_io_id_from_PI(p)
                    p_name = f"pi[{netlist.stage_PI_list[stage_id].index(pi_io_id)}]"
                if p_type == "AND":
                    if p_stage == stage_id:
                        p_name = f"node_{p}"
                    elif p in netlist.stage_MI_list[stage_id]:
                        p_name = f"mi[{netlist.stage_MI_list[stage_id].index(p)}]"
                    else:
                        raise ValueError("Invalid node")
                if p_type == "CONST0":
                    p_name = "1'b0"
                input_node.append(
                    p_name if input_edge_type == "DIRECT" else f"~{p_name}"
                )
            verilog.append(f"  wire node_{node} = {' & '.join(input_node)};\n")
    for lo_io_id in netlist.stage_LO_list[stage_id]:
        lo_node_id = netlist.get_LO_from_io_id(lo_io_id)
        lo_pred = list(netlist.graph.predecessors(lo_node_id))[0]
        lo_pred_type = netlist.graph.nodes[lo_pred]["type"]
        lo_pred_stage = netlist.graph.nodes[lo_pred]["stage"]
        lo_pred_edge_type = netlist.graph.edges[lo_pred, lo_node_id]["type"]

        if lo_pred_type == "AND" and lo_pred_stage == stage_id:
            p_name = f"node_{lo_pred}"
        elif lo_pred_type == "LI":
            li_io_id = netlist.get_io_id_from_LI(lo_pred)
            p_name = f"li[{netlist.stage_LI_list[stage_id].index(li_io_id)}]"
        elif lo_pred_type == "PI":
            pi_io_id = netlist.get_io_id_from_PI(lo_pred)
            p_name = f"pi[{netlist.stage_PI_list[stage_id].index(pi_io_id)}]"
        elif lo_pred_type == "CONST0":
            p_name = "1'b0"
        elif lo_pred in netlist.stage_MI_list[stage_id]:
            p_name = f"mi[{netlist.stage_MI_list[stage_id].index(lo_pred)}]"
        else:
            raise ValueError("Invalid node")
        input_node = p_name if lo_pred_edge_type == "DIRECT" else f"~{p_name}"
        verilog.append(
            f"  assign lo[{netlist.stage_LO_list[stage_id].index(lo_io_id)}] = {input_node};\n"
        )
    for po_io_id in netlist.stage_PO_list[stage_id]:
        po_node_id = netlist.get_PO_from_io_id(po_io_id)
        po_pred = list(netlist.graph.predecessors(po_node_id))[0]
        po_pred_type = netlist.graph.nodes[po_pred]["type"]
        po_pred_stage = netlist.graph.nodes[po_pred]["stage"]
        po_pred_edge_type = netlist.graph.edges[po_pred, po_node_id]["type"]
        if po_pred_type == "AND" and po_pred_stage == stage_id:
            p_name = f"node_{po_pred}"
        elif po_pred_type == "LI":
            li_io_id = netlist.get_io_id_from_LI(po_pred)
            p_name = f"li[{netlist.stage_LI_list[stage_id].index(li_io_id)}]"
        elif po_pred_type == "PI":
            pi_io_id = netlist.get_io_id_from_PI(po_pred)
            p_name = f"pi[{netlist.stage_PI_list[stage_id].index(pi_io_id)}]"
        elif po_pred_type == "CONST0":
            p_name = "1'b0"
        elif po_pred in netlist.stage_MI_list[stage_id]:
            p_name = f"mi[{netlist.stage_MI_list[stage_id].index(po_pred)}]"
        else:
            raise ValueError("Invalid node")
        input_node = p_name if po_pred_edge_type == "DIRECT" else f"~{p_name}"
        verilog.append(
            f"  assign po[{netlist.stage_PO_list[stage_id].index(po_io_id)}] = {input_node};\n"
        )
    for mo in netlist.stage_MO_list[stage_id]:
        if mo in netlist.stage_MI_list[stage_id]:
            verilog.append(
                f"  assign mo[{netlist.stage_MO_list[stage_id].index(mo)}] = mi[{netlist.stage_MI_list[stage_id].index(mo)}];\n"
            )
        elif (
            netlist.graph.nodes[mo]["stage"] == stage_id
            and netlist.graph.nodes[mo]["type"] == "AND"
        ):
            verilog.append(
                f"  assign mo[{netlist.stage_MO_list[stage_id].index(mo)}] = node_{mo};\n"
            )
    verilog.append("endmodule\n")
    return verilog


def diff_controller_verilog(diff_controller_json: dict, n_stages: int):
    ctrl_input_len = diff_controller_json["input_width_"]
    node_num = diff_controller_json["node_num_"]
    output_node_id = diff_controller_json["output_id_list_"][0]
    node_list = diff_controller_json["node_vector_"]
    verilog = []
    verilog.append(f"module controller (\n")
    verilog.append(f"  input [{ctrl_input_len-1}:0] ctrl_in,\n")
    verilog.append(f"  output ctrl_out\n);\n")
    verilog.append(f"  wire basic_ctrl_out = ~(&ctrl_in[{n_stages-2}:0]);\n")
    for n in range(node_num):
        verilog.append(f"  wire node_{n};\n")
    for n in range(node_num):
        node_dict = node_list[n]
        verilog.append(
            f"  assign node_{node_dict['node_id_']} = ctrl_in[{node_dict['input_bits_id_']}] ? "
        )
        match [node_dict["is_right_leaf_"], node_dict["is_right_neg_"]]:
            case [0, 0]:
                verilog.append(f"node_{node_dict['right_id_']} : ")
            case [0, 1]:
                verilog.append(f"~node_{node_dict['right_id_']} : ")
            case [1, 0]:
                verilog.append(f"1'b{int(node_dict['right_leaf_value_'])} : ")
            case [1, 1]:
                verilog.append(f"1'b{int(not node_dict['right_leaf_value_'])} : ")
        match [node_dict["is_left_leaf_"], node_dict["is_left_neg_"]]:
            case [0, 0]:
                verilog.append(f"node_{node_dict['left_id_']};\n")
            case [0, 1]:
                verilog.append(f"~node_{node_dict['left_id_']};\n")
            case [1, 0]:
                verilog.append(f"1'b{int(node_dict['left_leaf_value_'])};\n")
            case [1, 1]:
                verilog.append(f"1'b{int(not node_dict['left_leaf_value_'])};\n")
    verilog.append(f"  assign ctrl_out = node_{output_node_id} & basic_ctrl_out;\n")
    verilog.append(f"endmodule\n")
    return verilog


def controller_verilog(controller_json: dict):
    ctrl_input_len = controller_json["input_width_"]
    node_num = controller_json["node_num_"]
    output_node_id = controller_json["output_id_list_"][0]
    node_list = controller_json["node_vector_"]
    verilog = []
    verilog.append(f"module controller (\n")
    verilog.append(f"  input [{ctrl_input_len-1}:0] ctrl_in,\n")
    verilog.append(f"  output ctrl_out\n);\n")
    for n in range(node_num):
        verilog.append(f"  wire node_{n};\n")
    for n in range(node_num):
        node_dict = node_list[n]
        verilog.append(
            f"  assign node_{node_dict['node_id_']} = ctrl_in[{node_dict['input_bits_id_']}] ? "
        )
        match [node_dict["is_right_leaf_"], node_dict["is_right_neg_"]]:
            case [0, 0]:
                verilog.append(f"node_{node_dict['right_id_']} : ")
            case [0, 1]:
                verilog.append(f"~node_{node_dict['right_id_']} : ")
            case [1, 0]:
                verilog.append(f"1'b{int(node_dict['right_leaf_value_'])} : ")
            case [1, 1]:
                verilog.append(f"1'b{int(not node_dict['right_leaf_value_'])} : ")
        match [node_dict["is_left_leaf_"], node_dict["is_left_neg_"]]:
            case [0, 0]:
                verilog.append(f"node_{node_dict['left_id_']};\n")
            case [0, 1]:
                verilog.append(f"~node_{node_dict['left_id_']};\n")
            case [1, 0]:
                verilog.append(f"1'b{int(node_dict['left_leaf_value_'])};\n")
            case [1, 1]:
                verilog.append(f"1'b{int(not node_dict['left_leaf_value_'])};\n")
    verilog.append(f"  assign ctrl_out = node_{output_node_id};\n")
    verilog.append(f"endmodule\n")
    return verilog


def naive_controller_verilog(netlist: nl.Netlist):
    verilog = []
    ctrl_input_len = (
        netlist.n_stages
        - 1
        + netlist.n_stages * netlist.PI_num
        + sum([len(netlist.ctrl_LI_io_id_table[i]) for i in range(netlist.n_stages)])
    )
    verilog.append(f"module controller (\n")
    verilog.append(f"  input [{ctrl_input_len - 1}:0] ctrl_in,\n")
    verilog.append(f"  output ctrl_out\n);\n")
    verilog.append(f"  assign ctrl_out = ~(&ctrl_in[1:0]);\n")
    verilog.append(f"endmodule\n")
    return verilog


def autopipeline_top_verilog(netlist: nl.Netlist):
    verilog = []
    verilog.append(f"module ap_top (\n")
    verilog.append(f"  input clock,\n")
    verilog.append(f"  input reset,\n")
    if netlist.PI_num:
        verilog.append(f"  input [{netlist.PI_num-1}:0] pi,\n")
    if netlist.PO_num:
        verilog.append(f"  output [{netlist.PO_num-1}:0] po,\n")
    verilog.append("  output ready,\n")
    verilog.append("  output done\n);\n")
    # wire声明
    for stage_id in range(netlist.n_stages):
        if netlist.stage_LI_list[stage_id]:
            verilog.append(
                f"  wire [{len(netlist.stage_LI_list[stage_id])-1}:0] s{stage_id}_li_wire;\n"
            )
    for stage_id in range(netlist.n_stages):
        if netlist.stage_MI_list[stage_id]:
            verilog.append(
                f"  wire [{len(netlist.stage_MI_list[stage_id])-1}:0] s{stage_id}_mi_wire;\n"
            )
    for stage_id in range(netlist.n_stages):
        if netlist.stage_PI_list[stage_id]:
            verilog.append(
                f"  wire [{len(netlist.stage_PI_list[stage_id])-1}:0] s{stage_id}_pi_wire;\n"
            )
    for stage_id in range(netlist.n_stages):
        if netlist.stage_LO_list[stage_id]:
            verilog.append(
                f"  wire [{len(netlist.stage_LO_list[stage_id])-1}:0] s{stage_id}_lo_wire;\n"
            )
    for stage_id in range(netlist.n_stages):
        if netlist.stage_MO_list[stage_id]:
            verilog.append(
                f"  wire [{len(netlist.stage_MO_list[stage_id])-1}:0] s{stage_id}_mo_wire;\n"
            )
    for stage_id in range(netlist.n_stages):
        if netlist.stage_PO_list[stage_id]:
            verilog.append(
                f"  wire [{len(netlist.stage_PO_list[stage_id])-1}:0] s{stage_id}_po_wire;\n"
            )
    ctrl_input_len = (
        netlist.n_stages
        - 1
        + netlist.n_stages * netlist.PI_num
        + sum([len(netlist.ctrl_LI_io_id_table[i]) for i in range(netlist.n_stages)])
    )
    verilog.append(f"  wire [{ctrl_input_len-1}:0] ctrl_in_wire;\n")
    verilog.append(f"  wire ctrl_out_wire;\n")
    for stage_id in range(netlist.n_stages):
        verilog.append(f"  wire [{netlist.Latch_num-1}:0] li_reg_{stage_id}_wire_in;\n")
    for stage_id in range(netlist.n_stages):
        verilog.append(f"  wire [{netlist.PO_num-1}:0] po_reg_{stage_id}_wire_in;\n")
    # reg声明
    for stage_id in range(netlist.n_stages):
        verilog.append(f"  reg [{netlist.Latch_num-1}:0] li_reg_{stage_id};\n")
    if netlist.PI_num:
        for stage_id in range(1, netlist.n_stages):
            verilog.append(f"  reg [{netlist.PI_num-1}:0] pi_reg_{stage_id};\n")
    for stage_id in range(netlist.n_stages - 1):
        verilog.append(f"  reg [{netlist.PO_num-1}:0] po_reg_{stage_id};\n")
    for stage_id in range(netlist.n_stages - 1):
        if netlist.stage_MO_list[stage_id]:
            verilog.append(
                f"  reg [{len(netlist.stage_MO_list[stage_id])-1}:0] m_reg_{stage_id}_{stage_id+1};\n"
            )
    for stage_id in range(1, netlist.n_stages):
        verilog.append(f"  reg stall_reg_{stage_id};\n")
    # module例化
    for stage_id in range(netlist.n_stages):
        verilog.append(f"  stage_{stage_id} s{stage_id} (\n")
        io_list = []
        if netlist.stage_LI_list[stage_id]:
            io_list.append(f"    .li(s{stage_id}_li_wire)")
        if netlist.stage_MI_list[stage_id]:
            io_list.append(f"    .mi(s{stage_id}_mi_wire)")
        if netlist.stage_PI_list[stage_id]:
            io_list.append(f"    .pi(s{stage_id}_pi_wire)")
        if netlist.stage_LO_list[stage_id]:
            io_list.append(f"    .lo(s{stage_id}_lo_wire)")
        if netlist.stage_MO_list[stage_id]:
            io_list.append(f"    .mo(s{stage_id}_mo_wire)")
        if netlist.stage_PO_list[stage_id]:
            io_list.append(f"    .po(s{stage_id}_po_wire)")
        verilog.append(",\n".join(io_list))
        verilog.append(f"\n  );\n")
    verilog.append(f"  controller ctrl (\n")
    verilog.append(f"    .ctrl_in(ctrl_in_wire),\n")
    verilog.append(f"    .ctrl_out(ctrl_out_wire)\n")
    verilog.append(f"  );\n")
    # module input wire assign
    for stage_id in range(netlist.n_stages):
        if netlist.stage_LI_list[stage_id]:
            li_reg_signal_names = [
                f"li_reg_{stage_id}[{i}]" for i in netlist.stage_LI_list[stage_id]
            ]
            concat = ", ".join(li_reg_signal_names[::-1])
            verilog.append(f"  assign s{stage_id}_li_wire = {{{concat}}};\n")
    if netlist.stage_PI_list[0]:
        pi_signal_names = [f"pi[{i}]" for i in netlist.stage_PI_list[0]]
        concat = ", ".join(pi_signal_names[::-1])
        verilog.append(f"  assign s0_pi_wire = {{{concat}}};\n")
    for stage_id in range(1, netlist.n_stages):
        if netlist.stage_PI_list[stage_id]:
            pi_reg_signal_names = [
                f"pi_reg_{stage_id}[{i}]" for i in netlist.stage_PI_list[stage_id]
            ]
            concat = ", ".join(pi_reg_signal_names[::-1])
            verilog.append(f"  assign s{stage_id}_pi_wire = {{{concat}}};\n")
    for stage_id in range(1, netlist.n_stages):
        if netlist.stage_MO_list[stage_id - 1]:
            verilog.append(
                f"  assign s{stage_id}_mi_wire = m_reg_{stage_id-1}_{stage_id};\n"
            )
    ctrl_input_name_list = []
    for i in range(1, netlist.n_stages):
        ctrl_input_name_list.append(f"stall_reg_{i}")
    if netlist.PI_num:
        ctrl_input_name_list.append("pi")
        for i in range(1, netlist.n_stages):
            ctrl_input_name_list.append(f"pi_reg_{i}")
    for i in range(netlist.n_stages):
        for li_io_id in netlist.ctrl_LI_io_id_table[i]:
            ctrl_input_name_list.append(f"li_reg_{i}[{li_io_id}]")
    concat = ", ".join(ctrl_input_name_list[::-1])
    verilog.append(f"  assign ctrl_in_wire = {{{concat}}};\n")
    # reg input wire assign
    for i in range(netlist.n_stages):
        po_reg_wire_in_signal_name = []
        for j in range(netlist.PO_num):
            if j in netlist.stage_PO_list[i]:
                po_reg_wire_in_signal_name.append(
                    f"s{i}_po_wire[{netlist.stage_PO_list[i].index(j)}]"
                )
            elif i == 0:
                po_reg_wire_in_signal_name.append("1'b0")
            elif i > 0:
                po_reg_wire_in_signal_name.append(f"po_reg_{i-1}[{j}]")
        concat = ", ".join(po_reg_wire_in_signal_name[::-1])
        verilog.append(f"  assign po_reg_{i}_wire_in = {{{concat}}};\n")
    # LI逻辑
    for i in range(netlist.n_stages):
        lo_reg_wire_in_signal_name = []
        for j in range(netlist.Latch_num):
            # TODO 现在又改回去了，对应evaluation里面改了回去
            # # TODO 重点检查这里对不对，这对应了evaluation中的更改
            # if i > 0 and j in netlist.ctrl_LI_io_id_table[i - 1]:
            #     lo_reg_wire_in_signal_name.append(f"li_reg_{i-1}[{j}]")
            #     continue
            for k in range(0, i):
                if j in netlist.stage_LO_list[k]:
                    lo_reg_wire_in_signal_name.append(f"li_reg_{i-1}[{j}]")
            for k in range(i, netlist.n_stages):
                if j in netlist.stage_LO_list[k]:
                    if k == 0:
                        lo_reg_wire_in_signal_name.append(
                            f"(ctrl_out_wire ? li_reg_{i}[{j}] : s{k}_lo_wire[{netlist.stage_LO_list[k].index(j)}])"
                        )
                    else:
                        lo_reg_wire_in_signal_name.append(
                            f"(stall_reg_{k} ? li_reg_{i}[{j}] : s{k}_lo_wire[{netlist.stage_LO_list[k].index(j)}])"
                        )
        concat = ", ".join(lo_reg_wire_in_signal_name[::-1])
        verilog.append(f"  assign li_reg_{i}_wire_in = {{{concat}}};\n")

    # reg input
    verilog.append(f"  always @(posedge clock) begin\n")
    verilog.append(f"    if (reset) begin\n")
    for i in range(netlist.n_stages):
        verilog.append(f"      li_reg_{i} <= 0;\n")
    if netlist.PI_num:
        for i in range(1, netlist.n_stages):
            verilog.append(f"      pi_reg_{i} <= 0;\n")
    for i in range(netlist.n_stages - 1):
        verilog.append(f"      po_reg_{i} <= 0;\n")
    for i in range(netlist.n_stages - 1):
        if netlist.stage_MO_list[i]:
            verilog.append(f"      m_reg_{i}_{i+1} <= 0;\n")
    for i in range(1, netlist.n_stages):
        verilog.append(f"      stall_reg_{i} <= 1;\n")
    verilog.append(f"    end else begin\n")
    for i in range(netlist.n_stages):
        # LI逻辑
        verilog.append(f"      li_reg_{i} <= li_reg_{i}_wire_in;\n")
    if netlist.PI_num:
        if netlist.n_stages > 1:
            verilog.append(f"      pi_reg_1 <= pi;\n")
        for i in range(2, netlist.n_stages):
            verilog.append(f"      pi_reg_{i} <= pi_reg_{i-1};\n")
    for i in range(netlist.n_stages - 1):

        verilog.append(f"      po_reg_{i} <= po_reg_{i}_wire_in;\n")
    for i in range(netlist.n_stages - 1):
        if netlist.stage_MO_list[i]:
            verilog.append(f"      m_reg_{i}_{i+1} <= s{i}_mo_wire;\n")
    if netlist.n_stages > 1:
        verilog.append(f"      stall_reg_1 <= ctrl_out_wire;\n")
    for i in range(2, netlist.n_stages):
        verilog.append(f"      stall_reg_{i} <= stall_reg_{i-1};\n")
    verilog.append(f"    end\n")
    verilog.append(f"  end\n")
    # OUTPUT
    verilog.append(f"  assign ready = ~ctrl_out_wire;\n")
    verilog.append(f"  assign done = ~stall_reg_{netlist.n_stages-1};\n")
    verilog.append(f"  assign po = po_reg_{netlist.n_stages-1}_wire_in;\n")
    verilog.append("endmodule\n")
    return verilog


def baseline_pipeline_top_verilog(netlist: nl.Netlist):
    verilog = []
    verilog.append(f"module baseline_top (\n")
    verilog.append(f"  input clock,\n")
    verilog.append(f"  input reset,\n")
    if netlist.PI_num:
        verilog.append(f"  input [{netlist.PI_num-1}:0] pi,\n")
    if netlist.PO_num:
        verilog.append(f"  output [{netlist.PO_num-1}:0] po,\n")
    verilog.append("  output ready,\n")
    verilog.append("  output done\n);\n")
    # wire声明
    for stage_id in range(netlist.n_stages):
        if netlist.stage_LI_list[stage_id]:
            verilog.append(
                f"  wire [{len(netlist.stage_LI_list[stage_id])-1}:0] s{stage_id}_li_wire;\n"
            )
    for stage_id in range(netlist.n_stages):
        if netlist.stage_MI_list[stage_id]:
            verilog.append(
                f"  wire [{len(netlist.stage_MI_list[stage_id])-1}:0] s{stage_id}_mi_wire;\n"
            )
    for stage_id in range(netlist.n_stages):
        if netlist.stage_PI_list[stage_id]:
            verilog.append(
                f"  wire [{len(netlist.stage_PI_list[stage_id])-1}:0] s{stage_id}_pi_wire;\n"
            )
    for stage_id in range(netlist.n_stages):
        if netlist.stage_LO_list[stage_id]:
            verilog.append(
                f"  wire [{len(netlist.stage_LO_list[stage_id])-1}:0] s{stage_id}_lo_wire;\n"
            )
    for stage_id in range(netlist.n_stages):
        if netlist.stage_MO_list[stage_id]:
            verilog.append(
                f"  wire [{len(netlist.stage_MO_list[stage_id])-1}:0] s{stage_id}_mo_wire;\n"
            )
    for stage_id in range(netlist.n_stages):
        if netlist.stage_PO_list[stage_id]:
            verilog.append(
                f"  wire [{len(netlist.stage_PO_list[stage_id])-1}:0] s{stage_id}_po_wire;\n"
            )
    verilog.append(f"  wire ctrl_out_wire;\n")

    for stage_id in range(netlist.n_stages):
        verilog.append(f"  wire [{netlist.Latch_num-1}:0] lo_reg_{stage_id}_wire_in;\n")
    for stage_id in range(netlist.n_stages):
        verilog.append(f"  wire [{netlist.PO_num-1}:0] po_reg_{stage_id}_wire_in;\n")
    # reg声明
    for stage_id in range(netlist.n_stages):
        verilog.append(f"  reg [{netlist.Latch_num-1}:0] li_reg_{stage_id};\n")
    for stage_id in range(netlist.n_stages - 1):
        verilog.append(f"  reg [{netlist.Latch_num-1}:0] lo_reg_{stage_id};\n")
    if netlist.PI_num:
        for stage_id in range(1, netlist.n_stages):
            verilog.append(f"  reg [{netlist.PI_num-1}:0] pi_reg_{stage_id};\n")
    for stage_id in range(netlist.n_stages - 1):
        verilog.append(f"  reg [{netlist.PO_num-1}:0] po_reg_{stage_id};\n")
    for stage_id in range(netlist.n_stages - 1):
        if netlist.stage_MO_list[stage_id]:
            verilog.append(
                f"  reg [{len(netlist.stage_MO_list[stage_id])-1}:0] m_reg_{stage_id}_{stage_id+1};\n"
            )
    for stage_id in range(1, netlist.n_stages):
        verilog.append(f"  reg stall_reg_{stage_id};\n")

    # module例化
    for stage_id in range(netlist.n_stages):
        verilog.append(f"  stage_{stage_id} s{stage_id} (\n")
        io_list = []
        if netlist.stage_LI_list[stage_id]:
            io_list.append(f"    .li(s{stage_id}_li_wire)")
        if netlist.stage_MI_list[stage_id]:
            io_list.append(f"    .mi(s{stage_id}_mi_wire)")
        if netlist.stage_PI_list[stage_id]:
            io_list.append(f"    .pi(s{stage_id}_pi_wire)")
        if netlist.stage_LO_list[stage_id]:
            io_list.append(f"    .lo(s{stage_id}_lo_wire)")
        if netlist.stage_MO_list[stage_id]:
            io_list.append(f"    .mo(s{stage_id}_mo_wire)")
        if netlist.stage_PO_list[stage_id]:
            io_list.append(f"    .po(s{stage_id}_po_wire)")
        verilog.append(",\n".join(io_list))
        verilog.append(f"\n  );\n")
    verilog.append(
        f"  assign ctrl_out_wire = ~({' & '.join([f'stall_reg_{s}' for s in range(1, netlist.n_stages)])});\n"
    )
    # module input wire assign
    for stage_id in range(netlist.n_stages):
        if netlist.stage_LI_list[stage_id]:
            li_reg_signal_names = [
                f"li_reg_{stage_id}[{i}]" for i in netlist.stage_LI_list[stage_id]
            ]
            concat = ", ".join(li_reg_signal_names[::-1])
            verilog.append(f"  assign s{stage_id}_li_wire = {{{concat}}};\n")
    if netlist.stage_PI_list[0]:
        pi_signal_names = [f"pi[{i}]" for i in netlist.stage_PI_list[0]]
        concat = ", ".join(pi_signal_names[::-1])
        verilog.append(f"  assign s0_pi_wire = {{{concat}}};\n")
    for stage_id in range(1, netlist.n_stages):
        if netlist.stage_PI_list[stage_id]:
            pi_reg_signal_names = [
                f"pi_reg_{stage_id}[{i}]" for i in netlist.stage_PI_list[stage_id]
            ]
            concat = ", ".join(pi_reg_signal_names[::-1])
            verilog.append(f"  assign s{stage_id}_pi_wire = {{{concat}}};\n")
    for stage_id in range(1, netlist.n_stages):
        if netlist.stage_MO_list[stage_id - 1]:
            verilog.append(
                f"  assign s{stage_id}_mi_wire = m_reg_{stage_id-1}_{stage_id};\n"
            )
    # reg input wire assign
    for i in range(netlist.n_stages):
        po_reg_wire_in_signal_name = []
        for j in range(netlist.PO_num):
            if j in netlist.stage_PO_list[i]:
                po_reg_wire_in_signal_name.append(
                    f"s{i}_po_wire[{netlist.stage_PO_list[i].index(j)}]"
                )
            elif i == 0:
                po_reg_wire_in_signal_name.append("1'b0")
            elif i > 0:
                po_reg_wire_in_signal_name.append(f"po_reg_{i-1}[{j}]")
        concat = ", ".join(po_reg_wire_in_signal_name[::-1])
        verilog.append(f"  assign po_reg_{i}_wire_in = {{{concat}}};\n")
    # lo reg wire in
    for i in range(netlist.n_stages):
        lo_reg_wire_in_signal_name = []
        for j in range(netlist.Latch_num):
            if j in netlist.stage_LO_list[i]:
                lo_reg_wire_in_signal_name.append(
                    f"s{i}_lo_wire[{netlist.stage_LO_list[i].index(j)}]"
                )
            elif i == 0:
                lo_reg_wire_in_signal_name.append("1'b0")
            elif i > 0:
                lo_reg_wire_in_signal_name.append(f"lo_reg_{i-1}[{j}]")
        concat = ", ".join(lo_reg_wire_in_signal_name[::-1])
        verilog.append(f"  assign lo_reg_{i}_wire_in = {{{concat}}};\n")

    # reg input
    verilog.append(f"  always @(posedge clock) begin\n")
    verilog.append(f"    if (reset) begin\n")
    for i in range(netlist.n_stages):
        verilog.append(f"      li_reg_{i} <= 0;\n")
    if netlist.PI_num:
        for i in range(1, netlist.n_stages):
            verilog.append(f"      pi_reg_{i} <= 0;\n")
    for i in range(netlist.n_stages - 1):
        verilog.append(f"      po_reg_{i} <= 0;\n")
    for i in range(netlist.n_stages - 1):
        if netlist.stage_MO_list[i]:
            verilog.append(f"      m_reg_{i}_{i+1} <= 0;\n")
    for i in range(1, netlist.n_stages):
        verilog.append(f"      stall_reg_{i} <= 1;\n")
    verilog.append(f"    end else begin\n")
    if netlist.n_stages > 1:
        verilog.append(f"      li_reg_0 <= lo_reg_{netlist.n_stages-1}_wire_in;\n")
    for i in range(1, netlist.n_stages):
        # LI逻辑
        verilog.append(f"      li_reg_{i} <= li_reg_{i-1};\n")
    if netlist.PI_num:
        if netlist.n_stages > 1:
            verilog.append(f"      pi_reg_1 <= pi;\n")
        for i in range(2, netlist.n_stages):
            verilog.append(f"      pi_reg_{i} <= pi_reg_{i-1};\n")
    for i in range(netlist.n_stages - 1):
        verilog.append(f"      po_reg_{i} <= po_reg_{i}_wire_in;\n")
    for i in range(netlist.n_stages - 1):
        verilog.append(f"      lo_reg_{i} <= lo_reg_{i}_wire_in;\n")
    for i in range(netlist.n_stages - 1):
        if netlist.stage_MO_list[i]:
            verilog.append(f"      m_reg_{i}_{i+1} <= s{i}_mo_wire;\n")
    if netlist.n_stages > 1:
        verilog.append(f"      stall_reg_1 <= ctrl_out_wire;\n")
    for i in range(2, netlist.n_stages):
        verilog.append(f"      stall_reg_{i} <= stall_reg_{i-1};\n")
    verilog.append(f"    end\n")
    verilog.append(f"  end\n")
    # OUTPUT
    verilog.append(f"  assign ready = ~ctrl_out_wire;\n")
    verilog.append(f"  assign done = ~stall_reg_{netlist.n_stages-1};\n")
    verilog.append(f"  assign po = po_reg_{netlist.n_stages-1}_wire_in;\n")
    verilog.append("endmodule\n")
    return verilog


def ref_top_verilog(netlist: nl.Netlist):
    verilog = []
    verilog.append(f"module ref_top (\n")
    verilog.append(f"  input clock,\n")
    verilog.append(f"  input reset,\n")
    if netlist.PI_num:
        verilog.append(f"  input [{netlist.PI_num-1}:0] pi,\n")
    if netlist.PO_num:
        verilog.append(f"  output [{netlist.PO_num-1}:0] po,\n")
    verilog.append("  output ready,\n")
    verilog.append("  output done\n);\n")
    for stage_id in range(netlist.n_stages):
        if netlist.stage_MO_list[stage_id]:
            verilog.append(
                f"  wire [{len(netlist.stage_MO_list[stage_id])-1}:0] m_{stage_id}_{stage_id+1};\n"
            )
    verilog.append(f"  reg [{netlist.Latch_num-1}:0] latch;\n")
    for stage_id in range(netlist.n_stages):
        if netlist.stage_LI_list[stage_id]:
            li_signal_name = []
            for li_io_id in netlist.stage_LI_list[stage_id]:
                li_signal_name.append(f"latch[{li_io_id}]")
            concat = ", ".join(li_signal_name[::-1])
            verilog.append(
                f"  wire [{len(netlist.stage_LI_list[stage_id])-1}:0] li_{stage_id} = {{{concat}}};\n"
            )
    for stage_id in range(netlist.n_stages):
        if netlist.stage_LO_list[stage_id]:
            verilog.append(
                f"  wire [{len(netlist.stage_LO_list[stage_id])-1}:0] lo_{stage_id};\n"
            )
    for stage_id in range(netlist.n_stages):
        if netlist.stage_PI_list[stage_id]:
            pi_signal_name = []
            for pi_io_id in netlist.stage_PI_list[stage_id]:
                pi_signal_name.append(f"pi[{pi_io_id}]")
            concat = ", ".join(pi_signal_name[::-1])
            verilog.append(
                f"  wire [{len(netlist.stage_PI_list[stage_id])-1}:0] pi_{stage_id} = {{{concat}}};\n"
            )
    for stage_id in range(netlist.n_stages):
        if netlist.stage_PO_list[stage_id]:
            verilog.append(
                f"  wire [{len(netlist.stage_PO_list[stage_id])-1}:0] po_{stage_id};\n"
            )
    lo_signal_name = []
    for lo_id in range(netlist.Latch_num):
        for stage_id in range(netlist.n_stages):
            if lo_id in netlist.stage_LO_list[stage_id]:
                lo_signal_name.append(
                    f"lo_{stage_id}[{netlist.stage_LO_list[stage_id].index(lo_id)}]"
                )
    concat = ", ".join(lo_signal_name[::-1])
    verilog.append(f"  wire [{netlist.Latch_num-1}:0] lo = {{{concat}}};\n")
    verilog.append(f"  always @(posedge clock) begin\n")
    verilog.append(f"    if (reset) begin\n")
    verilog.append(f"      latch <= 0;\n")
    verilog.append(f"    end else begin\n")
    verilog.append(f"      latch <= lo;\n")
    verilog.append(f"    end\n")
    verilog.append(f"  end\n")
    # module例化
    for stage_id in range(netlist.n_stages):
        verilog.append(f"  stage_{stage_id} s{stage_id} (\n")
        io_list = []
        if netlist.stage_LI_list[stage_id]:
            io_list.append(f"    .li(li_{stage_id})")
        if netlist.stage_MI_list[stage_id]:
            io_list.append(f"    .mi(m_{stage_id-1}_{stage_id})")
        if netlist.stage_PI_list[stage_id]:
            io_list.append(f"    .pi(pi_{stage_id})")
        if netlist.stage_LO_list[stage_id]:
            io_list.append(f"    .lo(lo_{stage_id})")
        if netlist.stage_MO_list[stage_id]:
            io_list.append(f"    .mo(m_{stage_id}_{stage_id+1})")
        if netlist.stage_PO_list[stage_id]:
            io_list.append(f"    .po(po_{stage_id})")
        verilog.append(",\n".join(io_list))
        verilog.append(f"\n  );\n")
    verilog.append(f"  assign done = 1;")
    verilog.append(f"  assign ready = 1;\n")
    po_signal_name = []
    for po_io_id in range(netlist.PO_num):
        for stage_id in range(netlist.n_stages):
            if po_io_id in netlist.stage_PO_list[stage_id]:
                po_signal_name.append(
                    f"po_{stage_id}[{netlist.stage_PO_list[stage_id].index(po_io_id)}]"
                )
    concat = ", ".join(po_signal_name[::-1])
    verilog.append(f"  assign po = {{{concat}}};\n")
    verilog.append("endmodule\n")
    return verilog


def ref_top_top_verilog(netlist: nl.Netlist):
    verilog = []
    verilog.append(f"module ref_top_top (\n")
    verilog.append(f"  input clock,\n")
    verilog.append(f"  input reset,\n")
    if netlist.PI_num:
        verilog.append(f"  input [{netlist.PI_num-1}:0] pi,\n")
    if netlist.PO_num:
        verilog.append(f"  output reg [{netlist.PO_num-1}:0] po,\n")
    verilog.append("  output reg ready,\n")
    verilog.append("  output reg done\n);\n")

    if netlist.PI_num:
        verilog.append(f"  reg [{netlist.PI_num-1}:0] pi_r;\n")
    if netlist.PO_num:
        verilog.append(f"  wire [{netlist.PO_num-1}:0] po_w;\n")
    verilog.append("  wire ready_w;\n")
    verilog.append("  wire done_w;\n")
    verilog.append("  always @(posedge clock) begin\n")
    verilog.append("    if (reset) begin\n")
    if netlist.PI_num:
        verilog.append(f"      pi_r <= 0;\n")
    if netlist.PO_num:
        verilog.append(f"      po <= 0;\n")
    verilog.append("      ready <= 0;\n")
    verilog.append("      done <= 0;\n")
    verilog.append("    end else begin\n")
    if netlist.PI_num:
        verilog.append(f"      pi_r <= pi;\n")
    if netlist.PO_num:
        verilog.append(f"      po <= po_w;\n")
    verilog.append("      ready <= ready_w;\n")
    verilog.append("      done <= done_w;\n")
    verilog.append("    end\n")
    verilog.append("  end\n")
    verilog.append("  ref_top instance_ref_top (\n")
    verilog.append("    .clock(clock),\n")
    verilog.append("    .reset(reset),\n")
    if netlist.PI_num:
        verilog.append("    .pi(pi_r),\n")
    if netlist.PO_num:
        verilog.append("    .po(po_w),\n")
    verilog.append("    .ready(ready_w),\n")
    verilog.append("    .done(done_w)\n")
    verilog.append("  );\n")
    verilog.append("endmodule\n")
    return verilog


def ap_top_top_verilog(netlist: nl.Netlist):
    verilog = []
    verilog.append(f"module ap_top_top (\n")
    verilog.append(f"  input clock,\n")
    verilog.append(f"  input reset,\n")
    if netlist.PI_num:
        verilog.append(f"  input [{netlist.PI_num-1}:0] pi,\n")
    if netlist.PO_num:
        verilog.append(f"  output reg [{netlist.PO_num-1}:0] po,\n")
    verilog.append("  output reg ready,\n")
    verilog.append("  output reg done\n);\n")

    if netlist.PI_num:
        verilog.append(f"  reg [{netlist.PI_num-1}:0] pi_r;\n")
    if netlist.PO_num:
        verilog.append(f"  wire [{netlist.PO_num-1}:0] po_w;\n")
    verilog.append("  wire ready_w;\n")
    verilog.append("  wire done_w;\n")
    verilog.append("  always @(posedge clock) begin\n")
    verilog.append("    if (reset) begin\n")
    if netlist.PI_num:
        verilog.append(f"      pi_r <= 0;\n")
    if netlist.PO_num:
        verilog.append(f"      po <= 0;\n")
    verilog.append("      ready <= 0;\n")
    verilog.append("      done <= 0;\n")
    verilog.append("    end else begin\n")
    if netlist.PI_num:
        verilog.append(f"      pi_r <= pi;\n")
    if netlist.PO_num:
        verilog.append(f"      po <= po_w;\n")
    verilog.append("      ready <= ready_w;\n")
    verilog.append("      done <= done_w;\n")
    verilog.append("    end\n")
    verilog.append("  end\n")
    verilog.append("  ap_top instance_ap_top (\n")
    verilog.append("    .clock(clock),\n")
    verilog.append("    .reset(reset),\n")
    if netlist.PI_num:
        verilog.append("    .pi(pi_r),\n")
    if netlist.PO_num:
        verilog.append("    .po(po_w),\n")
    verilog.append("    .ready(ready_w),\n")
    verilog.append("    .done(done_w)\n")
    verilog.append("  );\n")
    verilog.append("endmodule\n")
    return verilog


def baseline_top_top_verilog(netlist: nl.Netlist):
    verilog = []
    verilog.append(f"module baseline_top_top (\n")
    verilog.append(f"  input clock,\n")
    verilog.append(f"  input reset,\n")
    if netlist.PI_num:
        verilog.append(f"  input [{netlist.PI_num-1}:0] pi,\n")
    if netlist.PO_num:
        verilog.append(f"  output reg [{netlist.PO_num-1}:0] po,\n")
    verilog.append("  output reg ready,\n")
    verilog.append("  output reg done\n);\n")

    if netlist.PI_num:
        verilog.append(f"  reg [{netlist.PI_num-1}:0] pi_r;\n")
    if netlist.PO_num:
        verilog.append(f"  wire [{netlist.PO_num-1}:0] po_w;\n")
    verilog.append("  wire ready_w;\n")
    verilog.append("  wire done_w;\n")
    verilog.append("  always @(posedge clock) begin\n")
    verilog.append("    if (reset) begin\n")
    if netlist.PI_num:
        verilog.append(f"      pi_r <= 0;\n")
    if netlist.PO_num:
        verilog.append(f"      po <= 0;\n")
    verilog.append("      ready <= 0;\n")
    verilog.append("      done <= 0;\n")
    verilog.append("    end else begin\n")
    if netlist.PI_num:
        verilog.append(f"      pi_r <= pi;\n")
    if netlist.PO_num:
        verilog.append(f"      po <= po_w;\n")
    verilog.append("      ready <= ready_w;\n")
    verilog.append("      done <= done_w;\n")
    verilog.append("    end\n")
    verilog.append("  end\n")
    verilog.append("  baseline_top instance_baseline_top (\n")
    verilog.append("    .clock(clock),\n")
    verilog.append("    .reset(reset),\n")
    if netlist.PI_num:
        verilog.append("    .pi(pi_r),\n")
    if netlist.PO_num:
        verilog.append("    .po(po_w),\n")
    verilog.append("    .ready(ready_w),\n")
    verilog.append("    .done(done_w)\n")
    verilog.append("  );\n")
    verilog.append("endmodule\n")
    return verilog


def testbench_verilog(
    netlist: nl.Netlist,
    n_test=1024,
    input_path="input.txt",
    output_path="output.txt",
    waveform_path=None,
):
    verilog = []
    verilog.append(f"module testbench;\n")
    verilog.append(f"  reg clock;\n")
    verilog.append(f"  reg reset;\n")
    if netlist.PI_num:
        verilog.append(f"  wire [{netlist.PI_num-1}:0] pi;\n")
    if netlist.PO_num:
        verilog.append(f"  wire [{netlist.PO_num-1}:0] po;\n")
    verilog.append(f"  wire ready;\n")
    verilog.append(f"  wire done;\n")
    if netlist.PI_num:
        verilog.append(f"  reg [{netlist.PI_num-1}:0] fifo [0:{n_test-1}];\n")
    verilog.append(f"  integer fifo_head = 0;\n")
    verilog.append(f"  integer fifo_tail = 0;\n")
    verilog.append(f"  integer data_count = 0;\n")
    verilog.append(f"  integer clock_count = 0;\n")
    verilog.append(f"  integer n_cycles = 0;\n")
    verilog.append(f"  integer input_file;\n")
    verilog.append(f"  integer output_file;\n")
    verilog.append(f"  assign pi = fifo[fifo_tail];\n")
    verilog.append(f"  top uut (\n")
    verilog.append(f"    .clock(clock),\n")
    verilog.append(f"    .reset(reset),\n")
    if netlist.PI_num:
        verilog.append(f"    .pi(pi),\n")
    if netlist.PO_num:
        verilog.append(f"    .po(po),\n")
    verilog.append(f"    .ready(ready),\n")
    verilog.append(f"    .done(done)\n")
    verilog.append(f"  );\n")
    verilog.append(f"  initial begin\n")
    verilog.append(f'    input_file = $fopen("{input_path}", "r");\n')
    verilog.append(f"    if (input_file == 0) begin\n")
    verilog.append(f'      $display("Error: cannot open input file");\n')
    verilog.append(f"      $finish;\n")
    verilog.append(f"    end\n")
    verilog.append(f"    while (!$feof(input_file)) begin\n")
    verilog.append(f'      $fscanf(input_file, "%b\\n", fifo[fifo_head]);\n')
    verilog.append(f"      fifo_head = fifo_head + 1;\n")
    verilog.append(f"      data_count = data_count + 1;\n")
    verilog.append(f"    end\n")
    verilog.append(f"    $fclose(input_file);\n")
    verilog.append(f"  end\n")
    verilog.append(f"  initial begin\n")
    verilog.append(f'    output_file = $fopen("{output_path}", "w");\n')
    verilog.append(f"    if (output_file == 0) begin\n")
    verilog.append(f'      $display("Error: cannot open output file");\n')
    verilog.append(f"      $finish;\n")
    verilog.append(f"    end\n")
    verilog.append(f"  end\n")
    verilog.append(f"  initial begin\n")
    verilog.append(f"    clock = 0;\n")
    verilog.append(f"    forever #5 clock = ~clock;\n")
    verilog.append(f"  end\n")
    verilog.append(f"  initial begin\n")
    verilog.append(f"    reset = 1;\n")
    verilog.append(f"    #10 reset = 0;\n")
    verilog.append(f"  end\n")
    verilog.append(f"  always @(posedge clock) begin\n")
    verilog.append(f"    if (reset) begin\n")
    verilog.append(f"      clock_count = 0;\n")
    verilog.append(f"      fifo_tail = 0;\n")
    verilog.append(f"    end else if (ready && (fifo_tail < fifo_head)) begin\n")
    verilog.append(f"      fifo_tail <= fifo_tail + 1;\n")
    verilog.append(f"    end\n")
    verilog.append(f"    if (~reset) begin\n")
    verilog.append(f"      clock_count = clock_count + 1;\n")
    verilog.append(f"    end\n")
    verilog.append(f"    if (done && data_count > 0) begin\n")
    verilog.append(f"      data_count = data_count - 1;\n")
    verilog.append(f'      $fwrite(output_file, "%b\\n", po);\n')
    verilog.append(f"      n_cycles = clock_count;\n")
    verilog.append(f"    end\n")
    verilog.append(f"  end\n")
    verilog.append(f"  initial begin\n")
    verilog.append(f"    #{10 * n_test * netlist.n_stages + 1000};\n")
    verilog.append(f'    $display("cycle num: %d", n_cycles);\n')
    verilog.append(f"    $fclose(output_file);\n")
    verilog.append(f"    $finish;\n")
    verilog.append(f"  end\n")
    if waveform_path:
        verilog.append(f"  initial begin\n")
        verilog.append(f'    $dumpfile("{waveform_path}");\n')
        verilog.append(f"    $dumpvars(0, uut);\n")
        verilog.append(f"  end\n")
    verilog.append(f"endmodule\n")
    return verilog


def dut_testbench_verilog(
    netlist: nl.Netlist,
    n_test=1024,
    waveform_path=None,
):
    verilog = []
    verilog.append(f"module testbench;\n")
    verilog.append(f"  reg clock;\n")
    verilog.append(f"  reg reset;\n")
    if netlist.PI_num:
        verilog.append(f"  wire [{netlist.PI_num-1}:0] ref_pi;\n")
        verilog.append(f"  wire [{netlist.PI_num-1}:0] dut_pi;\n")
    if netlist.PO_num:
        verilog.append(f"  wire [{netlist.PO_num-1}:0] ref_po;\n")
        verilog.append(f"  wire [{netlist.PO_num-1}:0] dut_po;\n")
    verilog.append(f"  wire ref_ready;\n")
    verilog.append(f"  wire ref_done;\n")
    verilog.append(f"  wire dut_ready;\n")
    verilog.append(f"  wire dut_done;\n")

    verilog.append(f"  reg [{netlist.PI_num-1}:0] fifo_in [0:{n_test-1}];\n")
    if netlist.PO_num:
        verilog.append(f"  reg [{netlist.PO_num-1}:0] ref_fifo_out [0:{n_test-1}];\n")
        verilog.append(f"  reg [{netlist.PO_num-1}:0] dut_fifo_out [0:{n_test-1}];\n")
    verilog.append(f"  integer ref_fifo_in_tail = 0;\n")
    verilog.append(f"  integer ref_fifo_out_head = 0;\n")
    verilog.append(f"  integer ref_n_cycles = 0;\n")
    verilog.append(f"  integer dut_fifo_in_tail = 0;\n")
    verilog.append(f"  integer dut_fifo_out_head = 0;\n")
    verilog.append(f"  integer dut_n_cycles = 0;\n")
    verilog.append(f"  integer clock_count = 0;\n")
    verilog.append(f"  integer i;\n")
    verilog.append(f"  integer error = 0;\n")

    verilog.append(f"  assign ref_pi = fifo_in[ref_fifo_in_tail];\n")
    verilog.append(f"  assign dut_pi = fifo_in[dut_fifo_in_tail];\n")
    verilog.append(f"  ref_top ref (\n")
    verilog.append(f"    .clock(clock),\n")
    verilog.append(f"    .reset(reset),\n")
    if netlist.PI_num:
        verilog.append(f"    .pi(ref_pi),\n")
    if netlist.PO_num:
        verilog.append(f"    .po(ref_po),\n")
    verilog.append(f"    .ready(ref_ready),\n")
    verilog.append(f"    .done(ref_done)\n")
    verilog.append(f"  );\n")
    verilog.append(f"  ap_top dut (\n")
    verilog.append(f"    .clock(clock),\n")
    verilog.append(f"    .reset(reset),\n")
    if netlist.PI_num:
        verilog.append(f"    .pi(dut_pi),\n")
    if netlist.PO_num:
        verilog.append(f"    .po(dut_po),\n")
    verilog.append(f"    .ready(dut_ready),\n")
    verilog.append(f"    .done(dut_done)\n")
    verilog.append(f"  );\n")
    verilog.append(f"  initial begin\n")
    verilog.append(f"    for (i = 0; i < {n_test}; i = i + 1) begin\n")
    verilog.append(f"      fifo_in[i] = $random;\n")
    verilog.append(f"    end\n")
    verilog.append(f"  end\n")
    verilog.append(f"  initial begin\n")
    verilog.append(f"    clock = 0;\n")
    verilog.append(f"    forever #5 clock = ~clock;\n")
    verilog.append(f"  end\n")
    verilog.append(f"  initial begin\n")
    verilog.append(f"    reset = 1;\n")
    verilog.append(f"    #10 reset = 0;\n")
    verilog.append(f"  end\n")
    verilog.append(f"  always @(posedge clock) begin\n")
    verilog.append(f"    if (reset) begin\n")
    verilog.append(f"      clock_count = 0;\n")
    verilog.append(f"      ref_fifo_in_tail = 0;\n")
    verilog.append(f"      dut_fifo_in_tail = 0;\n")
    verilog.append(f"      ref_fifo_out_head = 0;\n")
    verilog.append(f"      dut_fifo_out_head = 0;\n")
    verilog.append(f"      for (i = 0; i < {n_test}; i = i + 1) begin\n")
    verilog.append(f"        ref_fifo_out[i] = 0;\n")
    verilog.append(f"        dut_fifo_out[i] = 0;\n")
    verilog.append(f"      end\n")
    verilog.append(f"    end else begin\n")
    verilog.append(f"      if (ref_ready && (ref_fifo_in_tail < {n_test})) begin\n")
    verilog.append(f"        ref_fifo_in_tail <= ref_fifo_in_tail + 1;\n")
    verilog.append(f"      end\n")
    verilog.append(f"      if (dut_ready && (dut_fifo_in_tail < {n_test})) begin\n")
    verilog.append(f"        dut_fifo_in_tail <= dut_fifo_in_tail + 1;\n")
    verilog.append(f"      end\n")
    verilog.append(f"      clock_count = clock_count + 1;\n")
    verilog.append(f"      if (ref_done && ref_fifo_out_head < {n_test}) begin\n")
    verilog.append(f"        ref_fifo_out[ref_fifo_out_head] = ref_po;\n")
    verilog.append(f"        ref_fifo_out_head = ref_fifo_out_head + 1;\n")
    verilog.append(f"        ref_n_cycles = clock_count;\n")
    verilog.append(f"      end\n")
    verilog.append(f"      if (dut_done && dut_fifo_out_head < {n_test}) begin\n")
    verilog.append(f"        dut_fifo_out[dut_fifo_out_head] = dut_po;\n")
    verilog.append(f"        dut_fifo_out_head = dut_fifo_out_head + 1;\n")
    verilog.append(f"        dut_n_cycles = clock_count;\n")
    verilog.append(f"      end\n")
    verilog.append(f"    end\n")
    verilog.append(f"  end\n")
    verilog.append(f"  initial begin\n")
    verilog.append(f"    #{10 * n_test * netlist.n_stages + 1000};\n")
    verilog.append(f"    for (i = 0; i < {n_test}; i = i + 1) begin\n")
    verilog.append(f"       if (ref_fifo_out[i] !== dut_fifo_out[i]) begin\n")
    verilog.append(f"         error = error + 1;\n")
    verilog.append(f'         $display("error at %d", i);\n')
    verilog.append(f'         $display("ref: %b", ref_fifo_out[i]);\n')
    verilog.append(f'         $display("dut: %b", dut_fifo_out[i]);\n')
    verilog.append(f"       end\n")
    verilog.append(f"    end\n")
    verilog.append(
        f'    $display("============== SIMULATION RESULTS ==============");\n'
    )
    verilog.append(f'    $display("errors:     %-d", error);\n')
    verilog.append(f'    $display("dut cycles: %-d", dut_n_cycles);\n')
    verilog.append(
        f'    $display("dut CPI:    %-.4f", dut_n_cycles / {float(n_test)});\n'
    )
    verilog.append(
        f'    $display("================================================");\n'
    )
    verilog.append(f"    $finish;\n")
    verilog.append(f"  end\n")
    if waveform_path:
        verilog.append(f"  initial begin\n")
        verilog.append(f'    $dumpfile("{waveform_path}");\n')
        verilog.append(f"    $dumpvars(0, uut);\n")
        verilog.append(f"  end\n")
    verilog.append(f"endmodule\n")
    return verilog


def stimuli_file(netlist: nl.Netlist, n_test=1024, stimuli_path="input.txt"):
    with open(stimuli_path, "w") as file:
        for _ in range(n_test):
            binary_number = "".join(random.choice("01") for _ in range(netlist.PI_num))
            file.write(binary_number + "\n")


def dump_verilog(
    netlist_dir: str,
    ctrl_dir: str,
    design_dir: str,
    testbench_dir: str = None,
    use_diff_ctrl: bool = False,
    n_test: int = 1000000,
    waverform_dir: str = None,
):
    """
    netlist_dir为netlist的路径\n
    ctrl_dir为controller json文件的路径\n
    design_dir是生成的可综合的设计文件的路径\n
    testbench_dir是生成的testbench文件的路径，如果为None则不生成testbench文件，默认为None\n
    n_test是testbench中生成随机测试io的数量，默认100万\n
    waveform_dir是生成的波形文件的路径，如果为None则不生成波形文件，默认为None\n
    """
    netlist = pickle.load(open(netlist_dir, "rb"))
    if ctrl_dir is not None:
        with open(ctrl_dir, "r") as c_file:
            ctrl_json = json.load(c_file)

    with open(design_dir, "w") as d_file:
        for s in range(netlist.n_stages):
            d_file.writelines(stage_verilog(netlist, s))
        if ctrl_dir is not None:
            if use_diff_ctrl:
                d_file.writelines(diff_controller_verilog(ctrl_json, netlist.n_stages))
            else:
                d_file.writelines(controller_verilog(ctrl_json))
        else:
            d_file.writelines(naive_controller_verilog(netlist))
        d_file.writelines(autopipeline_top_verilog(netlist))
        d_file.writelines(ref_top_verilog(netlist))
        d_file.writelines(baseline_pipeline_top_verilog(netlist))
        d_file.writelines(ap_top_top_verilog(netlist))
        d_file.writelines(ref_top_top_verilog(netlist))
        d_file.writelines(baseline_top_top_verilog(netlist))

    if testbench_dir is not None:
        with open(testbench_dir, "w") as tb_file:
            tb_file.writelines(
                dut_testbench_verilog(
                    netlist, n_test=n_test, waveform_path=waverform_dir
                )
            )


def dump_json(netlist: nl.Netlist, output_path: str):
    n_pi = netlist.PI_num
    n_po = netlist.PO_num
    n_latch = netlist.Latch_num
    n_and = 0
    for n in netlist.graph.nodes():
        if netlist.graph.nodes[n]["type"] == "AND":
            n_and += 1

    n_node = n_and + n_pi + 2 * n_latch + n_po + 1
    n_stages = netlist.n_stages
    node_info_list = []
    if not netlist.graph.has_node(0):
        node_info_list.append({"id": 0, "type": "ZERO", "port_id": -1, "stage": -1})
    for node in netlist.graph.nodes():
        id = int(node)
        type = netlist.graph.nodes[node]["type"]
        if type == "CONST0":
            type = "ZERO"
        stage = netlist.graph.nodes[node]["stage"]
        if type == "LI":
            port_id = netlist.get_io_id_from_LI(node)
        elif type == "PI":
            port_id = netlist.get_io_id_from_PI(node)
        elif type == "LO":
            port_id = netlist.get_io_id_from_LO(node)
        elif type == "PO":
            port_id = netlist.get_io_id_from_PO(node)
        else:
            port_id = -1
        info = {"id": id, "type": type, "port_id": port_id, "stage": stage}
        node_info_list.append(info)
    edge_info_list = []
    for edge in netlist.graph.edges():
        src = int(edge[0])
        dest = int(edge[1])
        type = netlist.graph.edges[edge]["type"]
        if type == "DIRECT":
            type = "WIRE"
        info = {"src": src, "dest": dest, "type": type}
        edge_info_list.append(info)

    netlist_json = {
        "n_pi": n_pi,
        "n_po": n_po,
        "n_latch": n_latch,
        "n_and": n_and,
        "n_node": n_node,
        "n_stages": n_stages,
        "nodes": node_info_list,
        "edges": edge_info_list,
    }
    with open(output_path, "w") as f:
        json.dump(netlist_json, f)
