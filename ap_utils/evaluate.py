import random
import ap_utils.netlist as nl
import tqdm
import ap_utils.oracle_stall_logic as osl
import copy


def forward(source: dict, dest: dict):
    for k, v in source.items():
        if k in dest.keys():
            dest[k] = v


def connect_cotroller_input(
    netlist: nl.Netlist, global_li_pipeline_regs: list, global_pi_pipeline_regs: list
):
    ctrl_stall_input = copy.deepcopy(netlist.stage_stall_signal[1::])
    ctrl_pi_input = [
        sublist[k] for sublist in global_pi_pipeline_regs for k in range(netlist.PI_num)
    ]
    ctrl_li_input = [
        global_li_pipeline_regs[s][k]
        for s in range(netlist.n_stages)
        for k in netlist.ctrl_LI_io_id_table[s]
    ]
    controller_input = []
    controller_input.extend(ctrl_stall_input)
    controller_input.extend(ctrl_pi_input)
    controller_input.extend(ctrl_li_input)
    return controller_input


def execute_original_netlist(original_netlist: nl.Netlist, PI_list: list):
    lo = [0 for i in range(len(original_netlist.LO_id))]
    PO_list = []
    for pi in PI_list:
        li = lo
        po, lo = original_netlist.execute(pi, li)
        PO_list.append(po)
    return PO_list, lo


def execute_pipelined_netlist(netlist: nl.Netlist, PI_fifo: list):
    this_LI_list = [
        {key: 0 for key in netlist.stage_LI_list[i]} for i in range(netlist.n_stages)
    ]
    this_LO_list = [
        {key: 0 for key in netlist.stage_LO_list[i]} for i in range(netlist.n_stages)
    ]
    this_MI_list = [
        {key: 0 for key in netlist.stage_MI_list[i]} for i in range(netlist.n_stages)
    ]
    this_MO_list = [
        {key: 0 for key in netlist.stage_MO_list[i]} for i in range(netlist.n_stages)
    ]
    this_PI_list = [
        {key: 0 for key in netlist.stage_PI_list[i]} for i in range(netlist.n_stages)
    ]
    this_PO_list = [
        {key: 0 for key in netlist.stage_PO_list[i]} for i in range(netlist.n_stages)
    ]
    global_latch = {key: 0 for key in range(netlist.Latch_num)}
    global_li_pipeline_regs = [
        {key: 0 for key in range(netlist.Latch_num)} for i in range(netlist.n_stages)
    ]
    global_po_pipeline_regs = [
        {key: 0 for key in range(netlist.PO_num)} for i in range(netlist.n_stages)
    ]
    global_pi_pipeline_regs = [
        {key: 0 for key in range(netlist.PI_num)} for i in range(netlist.n_stages)
    ]
    netlist.stage_stall_signal = [1 for i in range(netlist.n_stages)]
    pipelined_PO_signal_list = []
    num_inputs = len(PI_fifo)
    with tqdm.tqdm(total=num_inputs, desc="Processing", ncols=100) as pbar:
        completed = 0
        cycle = 0
        while completed < num_inputs and cycle < netlist.n_stages * num_inputs:
            # 准备输入
            pi_pop = PI_fifo[0] if PI_fifo != [] else [0 for i in range(netlist.PI_num)]
            global_pi_pipeline_regs[0] = {i: pi_pop[i] for i in range(len(pi_pop))}
            for i in range(netlist.n_stages):
                forward(global_li_pipeline_regs[i], this_LI_list[i])
                forward(global_pi_pipeline_regs[i], this_PI_list[i])

            controller_input = connect_cotroller_input(
                netlist, global_li_pipeline_regs, global_pi_pipeline_regs
            )
            controller_output = osl.oracle_stall_logic(controller_input, netlist)

            netlist.stage_stall_signal[0] = controller_output if PI_fifo != [] else 1

            # 如果不stall，即发射，则从fifo中丢掉第一个输入
            if netlist.stage_stall_signal[0] == 0:
                PI_fifo.pop(0)

            # 执行每个流水级
            for i in range(netlist.n_stages):
                this_MO_list[i], this_PO_list[i], this_LO_list[i] = (
                    netlist.pipeline_stage_execute(
                        i, this_MI_list[i], this_PI_list[i], this_LI_list[i]
                    )
                )

            for i in range(netlist.n_stages - 1, -1, -1):
                if netlist.stage_stall_signal[i] != 1:
                    forward(this_PO_list[i], global_po_pipeline_regs[i])
                    forward(this_LO_list[i], global_latch)
            for i in range(0, netlist.n_stages - 1):
                forward(this_MO_list[i], this_MI_list[i + 1])
            if netlist.stage_stall_signal[netlist.n_stages - 1] == 0:
                completed_po_signal = [
                    global_po_pipeline_regs[netlist.n_stages - 1][k]
                    for k in range(netlist.PO_num)
                ]
                pipelined_PO_signal_list.append(completed_po_signal)

            # 流水寄存器移位
            for i in range(netlist.n_stages - 1, 0, -1):
                forward(global_pi_pipeline_regs[i - 1], global_pi_pipeline_regs[i])
                forward(global_po_pipeline_regs[i - 1], global_po_pipeline_regs[i])
                forward(global_li_pipeline_regs[i - 1], global_li_pipeline_regs[i])
            # latch前递
            for i in range(netlist.n_stages):
                if netlist.stage_stall_signal[i] == 0:
                    for j in range(i, -1, -1):
                        # TODO: 这里依然让他前递，反正不会被用到，所以不会导致计算错误。因为现在判断LO_stage不需要prune了，所以用更新的值是不会有坏处的。同时因为给到的都是最新的值，所以可以用来判断straight_through。这里还要check！！！是又改回去了
                        forward(this_LO_list[i], global_li_pipeline_regs[j])
            # cycle，issue计数
            cycle += 1
            if netlist.stage_stall_signal[netlist.n_stages - 1] == 0:
                completed += 1
                pbar.update(1)

            # stall信号移位
            for i in range(netlist.n_stages - 1, 0, -1):
                netlist.stage_stall_signal[i] = netlist.stage_stall_signal[i - 1]
    pipeline_LO_signal = [global_latch[i] for i in range(netlist.Latch_num)]
    return pipelined_PO_signal_list, pipeline_LO_signal, cycle


def estimate_cpi_with_oracle(netlist: nl.Netlist, n_test: int = 100):
    PI_signal_list = [
        [random.choice([1, 0]) for i in range(netlist.PI_num)] for n in range(n_test)
    ]
    _, _, cycle = execute_pipelined_netlist(netlist, PI_signal_list)
    return cycle / n_test
