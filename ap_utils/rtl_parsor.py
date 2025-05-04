import os
import argparse
import networkx as nx
import ap_utils.netlist as nl


def rtl2aag(
    rtl_path: str, clock_name: str, rst_name: str, rstn_name: str, top_name: str
):
    if rst_name is None and rstn_name is None:
        raise ValueError("At least one of rst and rstn must be used")
    if rst_name and rstn_name:
        raise ValueError("rst and rstn cannot be used together")
    rtl_design_name, extension = os.path.splitext(rtl_path)
    if rstn_name:
        yosys_command = (
            "read_verilog "
            + rtl_path
            + "; "
            + (f"hierarchy -top {top_name}; " if top_name else "hierarchy -auto-top; ")
            + "flatten; proc; zinit -all; "
            + "async2sync; dffunmap; techmap -map +/dff2ff.v; "
            + "delete */"
            + clock_name
            + "; opt_clean; "
            + "delete */"
            + rstn_name
            + "; setundef -one -undriven; "
            + "synth; aigmap; "
            + "write_aiger -ascii "
            + rtl_design_name
            + ".aag;"
        )
    elif rst_name:
        yosys_command = (
            "read_verilog "
            + rtl_path
            + "; "
            + (f"hierarchy -top {top_name}; " if top_name else "hierarchy -auto-top; ")
            + "flatten; proc; zinit -all; "
            + "async2sync; dffunmap; techmap -map +/dff2ff.v; "
            + "delete */"
            + clock_name
            + "; opt_clean; "
            + "delete */"
            + rst_name
            + "; setundef -zero -undriven; "
            + "synth; aigmap; "
            + "write_aiger -ascii "
            + rtl_design_name
            + ".aag;"
        )
    os.system('yosys -q -p "' + yosys_command + '"')


def aag2netlist(aag_path: str, draw_netlist: bool = False):
    netlist = nl.Netlist()
    with open(aag_path, "r") as f:
        lines = f.readlines()
        max_num, pi_num, latch_num, po_num, and_num = map(int, lines[0].split()[1:]) #提取最大节点数、主输入数、锁存器数、主输出数、逻辑门数
        netlist.PI_num = pi_num
        netlist.PO_num = po_num
        netlist.Latch_num = latch_num
        for i in range(pi_num):
            n0 = int(lines[i + 1])
            pi_id = n0 // 2
            netlist.add_node("PI", pi_id)
            netlist.set_io_id(pi_id, i)
        for i in range(latch_num):
            n0, n1 = map(int, lines[pi_num + i + 1].split()[:2])
            li_id = n0 // 2
            node_id = n1 // 2
            netlist.add_node("LI", li_id)
            if node_id not in netlist.graph.nodes():
                netlist.add_node("AND" if node_id != 0 else "CONST0", node_id)
            lo_id = max_num + i + 1
            netlist.add_node("LO", lo_id)
            netlist.add_edge(node_id, lo_id, "DIRECT" if n1 % 2 == 0 else "NOT")
            netlist.set_io_id(li_id, i)
            netlist.set_io_id(lo_id, i)
        for i in range(po_num):
            n0 = int(lines[pi_num + latch_num + i + 1])
            node_id = n0 // 2
            if node_id not in netlist.graph.nodes():
                netlist.add_node("AND" if node_id != 0 else "CONST0", node_id)
            po_id = max_num + latch_num + i + 1
            netlist.add_node("PO", po_id)
            netlist.add_edge(node_id, po_id, "DIRECT" if n0 % 2 == 0 else "NOT")
            netlist.set_io_id(po_id, i)
        for i in range(and_num):
            n0, n1, n2 = map(int, lines[pi_num + latch_num + po_num + i + 1].split())
            if n0 // 2 not in netlist.graph.nodes():
                netlist.add_node("AND" if n0 // 2 != 0 else "CONST0", n0 // 2)
            if n1 // 2 not in netlist.graph.nodes():
                netlist.add_node("AND" if n1 // 2 != 0 else "CONST0", n1 // 2)
            if n2 // 2 not in netlist.graph.nodes():
                netlist.add_node("AND" if n2 // 2 != 0 else "CONST0", n2 // 2)
            if n1 % 2 == 0:
                netlist.add_edge(n1 // 2, n0 // 2, "DIRECT")
            else:
                netlist.add_edge(n1 // 2, n0 // 2, "NOT")
            if n2 % 2 == 0:
                netlist.add_edge(n2 // 2, n0 // 2, "DIRECT")
            else:
                netlist.add_edge(n2 // 2, n0 // 2, "NOT")
        if (
            len(lines[pi_num + latch_num + po_num + and_num + 1].split()) != 1
            or lines[pi_num + latch_num + po_num + and_num + 1].strip() != "c"
        ):
            raise ValueError(
                "Invalid input. The last line does not contain a single 'c'."
            )
    if draw_netlist:
        design_name, extension = os.path.splitext(aag_path)
        graph_draw = netlist.graph.copy()
        for n in graph_draw.nodes():
            if (
                graph_draw.nodes[n]["type"] == "PI"
                or graph_draw.nodes[n]["type"] == "PO"
            ):
                graph_draw.nodes[n]["shape"] = "invtriangle"
            elif (
                graph_draw.nodes[n]["type"] == "LI"
                or graph_draw.nodes[n]["type"] == "LO"
            ):
                graph_draw.nodes[n]["shape"] = "invhouse"
            elif graph_draw.nodes[n]["type"] == "CONST0":
                graph_draw.nodes[n]["shape"] = "square"
            else:
                graph_draw.nodes[n]["shape"] = "oval"
        for e in graph_draw.edges():
            graph_draw.edges[e]["style"] = (
                "dashed" if graph_draw.edges[e]["type"] == "NOT" else "solid"
            )
        nx.drawing.nx_pydot.write_dot(graph_draw, design_name + ".dot")
    return netlist


def rtl2netlist(
    rtl_path: str,
    clock_name: str,
    rst_name: str,
    rstn_name: str,
    top_name: str,
    draw_netlist: bool = False,
):
    rtl_design_name, extension = os.path.splitext(rtl_path) # 文件名称/扩展名
    rtl2aag(
        rtl_path=rtl_path,
        clock_name=clock_name,
        rst_name=rst_name,
        rstn_name=rstn_name,
        top_name=top_name,
    )
    netlist = aag2netlist(aag_path=rtl_design_name + ".aag", draw_netlist=draw_netlist)
    netlist.detect_mux()
    netlist.topo_and_reverse_topo_sort()
    return netlist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtl", required=True, help="path to rtl")
    parser.add_argument("--clk", required=True, help="name of clock")
    parser.add_argument("--rst", help="name of reset")
    parser.add_argument("--rstn", help="name of resetn")
    parser.add_argument("--draw", action="store_true", help="draw the netlist")
    args = parser.parse_args()
    if args.rst is None and args.rstn is None:
        raise ValueError("At least one of -rst and -rstn must be used")
    if args.rst and args.rstn:
        raise ValueError("-rst and -rstn cannot be used together")

    netlist = rtl2netlist(
        rtl_path=args.rtl,
        clock_name=args.clk,
        rst_name=args.rst,
        rstn_name=args.rstn,
        draw_netlist=args.draw,
    )


if __name__ == "__main__":
    main()
