import random
import re
import ap_utils.rtl_parsor as r_parsor
import ap_utils.evaluate as eval
import ap_utils.pipeline_search as psearch
import ap_utils.oracle_stall_logic as osl
import ap_utils.verilog_dump as vdump
import pickle
import os
import secrets
import ap_utils.netlist as nl
from joblib import Parallel, delayed

seed = secrets.randbits(32)
seed = 429462822
random.seed(seed)

opencores_pattern = r"ip-cores-.+"
rtllm_pattern = r"verified_.+"
itc_pattern = r"b\d+"


def netlist_file_dir(design_name):
    if re.match(opencores_pattern, design_name):
        dir = f"./data/opencores/{design_name}.pipeline.netlist"
    elif re.match(rtllm_pattern, design_name):
        dir = f"./data/rtllm/{design_name}.pipeline.netlist"
    elif re.match(itc_pattern, design_name):
        dir = f"./data/itc/{design_name}.pipeline.netlist"
    else:
        raise Exception(f"design name {design_name} not recognized")
    return dir


def ctrl_json_dir(design_name):
    if re.match(opencores_pattern, design_name):
        dir = f"./data/opencores/ctrl_{design_name}.json"
    elif re.match(rtllm_pattern, design_name):
        dir = f"./data/rtllm/ctrl_{design_name}.json"
    elif re.match(itc_pattern, design_name):
        dir = f"./data/itc/ctrl_{design_name}.json"
    else:
        raise Exception(f"design name {design_name} not recognized")
    return dir


def diff_ctrl_json_dir(design_name):
    if re.match(opencores_pattern, design_name):
        dir = f"./data/opencores/ctrl_diff_{design_name}.json"
    elif re.match(rtllm_pattern, design_name):
        dir = f"./data/rtllm/ctrl_diff_{design_name}.json"
    elif re.match(itc_pattern, design_name):
        dir = f"./data/itc/ctrl_diff_{design_name}.json"
    else:
        raise Exception(f"design name {design_name} not recognized")
    return dir


def rtl_file_dir(design_name):
    if re.match(opencores_pattern, design_name):
        dir = f"./rtl/opencores/{design_name}.pipeline.v"
    elif re.match(rtllm_pattern, design_name):
        dir = f"./rtl/rtllm/{design_name}.pipeline.v"
    elif re.match(itc_pattern, design_name):
        dir = f"./rtl/itc/{design_name}.pipeline.v"
    else:
        raise Exception(f"design name {design_name} not recognized")
    return dir


def tb_file_dir(design_name):
    if re.match(opencores_pattern, design_name):
        dir = f"./data/opencores/{design_name}.tb.v"
    elif re.match(rtllm_pattern, design_name):
        dir = f"./data/rtllm/{design_name}.tb.v"
    elif re.match(itc_pattern, design_name):
        dir = f"./data/itc/{design_name}.tb.v"
    else:
        raise Exception(f"design name {design_name} not recognized")
    return dir


def cal_oracle_cpi(design, n_test_i):
    netlist_dir = netlist_file_dir(design)

    netlist = pickle.load(open(netlist_dir, "rb"))
    pi_fifo = [
        [random.choice([1, 0]) for i in range(netlist.PI_num)] for n in range(n_test_i)
    ]
    netlist.calculate_stage_IO_info()
    ppl_po, ppl_lo, cycles = eval.execute_pipelined_netlist(netlist, pi_fifo)
    print(cycles / n_test_i)


def save_rtl_and_tb(design, diff=False):
    netlist_dir = netlist_file_dir(design)
    ctrl_dir = ctrl_json_dir(design) if not diff else diff_ctrl_json_dir(design)
    rtl_dir = rtl_file_dir(design)
    tb_dir = tb_file_dir(design)
    if os.path.exists(netlist_dir) and os.path.exists(ctrl_dir):
        vdump.dump_verilog(netlist_dir, ctrl_dir, rtl_dir, tb_dir, use_diff_ctrl=diff)
        dc_env_dir = "./dc_env"
        os.system(f"cp {rtl_dir} {os.path.join(dc_env_dir,'rtl/')}")
    elif os.path.exists(netlist_dir) and ctrl_dir == None:
        vdump.dump_verilog(netlist_dir, ctrl_dir, rtl_dir, tb_dir, use_diff_ctrl=False)
        dc_env_dir = "./dc_env"
        os.system(f"cp {rtl_dir} {os.path.join(dc_env_dir,'rtl/')}")
    else:
        print(f"netlist do not exists. {design}")


def synth_with_dc_once(design, my_top, clk_period):
    rtl_dir = rtl_file_dir(design)
    if os.path.exists(rtl_dir):
        slack = None
        met = None
        rtl_file_name = os.path.splitext(os.path.basename(rtl_dir))[0]
        original_dir = os.getcwd()
        dc_env_dir = "./dc_env"
        os.chdir(dc_env_dir)
        os.system(
            f"make dc_all FILE_NAME={rtl_file_name} MY_TOP={my_top} CLK_PERIOD={clk_period:.2f} >> ../log/synth/{design}.log"
        )
        timing_report = f"./dc_env/report/{rtl_file_name}-{my_top}-dc_timing.rpt"
        with open(timing_report, "r") as f:
            report = f.read()
            slack_matches = re.findall(r"slack\s+(\(.+\))\s*(-?\d+\.\d+)", report)
            assert slack_matches != []
            met = slack_matches[0][0] == "(MET)"
            slack = float(slack_matches[0][1])
        os.chdir(original_dir)
        return slack, met
    else:
        return None, None


def synth_with_dc_find_min_clk(design, my_top):
    rtl_dir = rtl_file_dir(design)
    if os.path.exists(rtl_dir):
        clk_period = 0.01
        for i in range(20):
            slack, met = synth_with_dc_once(design, my_top, clk_period)
            if met is False:
                clk_period = clk_period + max(-slack, 0.01)
            elif met is True:
                break
            else:
                raise Exception("Error when synthesis")
        else:
            print(f"Failed to find min clk for {design} in 20 trials")
            return None
        return clk_period
    return None


def synth_with_dc(design):
    rtl_dir = rtl_file_dir(design)
    if os.path.exists(rtl_dir):
        rtl_file_name = os.path.splitext(os.path.basename(rtl_dir))[0]
        original_dir = os.getcwd()
        dc_env_dir = "./dc_env"
        os.chdir(dc_env_dir)
        os.system(
            f"make dc_all FILE_NAME={rtl_file_name} MY_TOP=ref_top CLK_PERIOD=0.01 >> ../log/synth/{design}.log"
        )
        os.system(
            f"make dc_all FILE_NAME={rtl_file_name} MY_TOP=baseline_top CLK_PERIOD=0.01 >> ../log/synth/{design}.log"
        )
        os.system(
            f"make dc_all FILE_NAME={rtl_file_name} MY_TOP=ap_top CLK_PERIOD=0.01 >> ../log/synth/{design}.log"
        )
        os.chdir(original_dir)


def verify_with_vcs(design):
    rtl_dir = rtl_file_dir(design)
    tb_dir = tb_file_dir(design)
    if os.path.exists(rtl_dir) and os.path.exists(tb_dir):
        verified = None
        cpi = None
        original_dir = os.getcwd()
        vcs_dir = "./vcs"
        os.chdir(vcs_dir)
        os.system(f"vcs -full64 {tb_dir} {rtl_dir} > /dev/null")
        os.system(f"./simv >> ../log/verify/{design}.log")
        with open(f"../log/verify/{design}.log") as f:
            text = f.read()
            err_matches = re.findall(r"errors:\s+(\d+)", text)
            assert err_matches is not []
            error_num = int(err_matches[0])
            if error_num > 0:
                verified = False
            else:
                verified = True
            cpi_matches = re.findall(r"dut CPI:\s+(\d+\.\d+)", text)
            assert cpi_matches is not []
            cpi = float(cpi_matches[0])
        os.chdir(original_dir)
        return verified, cpi
    return None, None


def main():
    design_list = [
        # "verified_calendar",
        # "verified_counter_12",
        # "verified_freq_div",
        # "verified_freq_divbyeven",
        # "verified_freq_divbyfrac",
        # "verified_freq_divbyodd",
        # "verified_pulse_detect",
        # "verified_sequence_detector",
        # "verified_serial2parallel",
        # "verified_width_8to16",
        # "b01",
        # "b02",
        "b08",
    ]
    Parallel(n_jobs=16)(
        delayed(save_rtl_and_tb)(design, diff=True) for design in design_list
    )

    # for design in design_list:
    #     ref_clk, baseline_clk, ap_clk = Parallel(n_jobs=3)(
    #         delayed(synth_with_dc_find_min_clk)(design, my_top)
    #         for my_top in ["ref_top_top", "baseline_top_top", "ap_top_top"]
    #     )
    #     print(
    #         f"{design}: ref_clk={ref_clk}, baseline_clk={baseline_clk}, ap_clk={ap_clk}"
    #     )

    # clk = synth_with_dc_find_min_clk("verified_multi_16bit", "baseline_top_top")
    # print(clk)
    # design_verified = Parallel(n_jobs=1)(
    #     delayed(verify_with_vcs)(design) for design in design_list
    # )
    # print(design_verified)


if __name__ == "__main__":
    # main()
    cal_oracle_cpi(
        "b08",
        1000,
    )
