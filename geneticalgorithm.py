import numpy as np
import copy
import random
import ap_utils.rtl_parsor as r_parsor
import ap_utils.evaluate as eval
import ap_utils.pipeline_search as psearch
import ap_utils.geneticalgorithm as ga
import ap_utils.oracle_stall_logic as osl
import numpy as np
import ap_utils.verilog_dump as vdump
import tqdm
import pickle
import networkx as nx
import os
from multiprocessing import Pool
import time
import multiprocessing
import pandas as pd
from collections import Counter
import math
import ap_utils.netlist as nl
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse
from xgboost import XGBRegressor
#import xgboost as xgb

def process_main(netlist):
    args = parse_args()
    num_stages = args.num_stages
    population_size = args.population_size
    num_generations = args.num_generations
    crossover_rate = args.crossover_rate
    mutation_rate = args.mutation_rate
    tournament_size = args.tournament_size
    li_nodes, _ , real_and_nodes_sorted, lo_nodes, po_nodes, real_longest_paths,real_li_pi_mapping,real_lo_mapping,real_po_mapping,real_li_lo_mapping = ga.extract_graph_info(netlist)
    new_netlist = ga.insert_virtual_nodes_into_netlist(netlist)
    li_nodes, _ , virtual_and_nodes_sorted, lo_nodes, po_nodes, virtual_longest_paths,virtual_li_pi_mapping,virtual_lo_mapping,virtual_po_mapping,virtual_li_lo_mapping = ga.extract_graph_info(new_netlist,value='False')
    psearch.stage_assignment_align_with_old(netlist,3)
    origin_stage_matrix = ga.initialize_partition(netlist, real_and_nodes_sorted, lo_nodes, po_nodes)
    _,max_scpl = ga.fitness_function(origin_stage_matrix, real_longest_paths, real_and_nodes_sorted, lo_nodes, po_nodes, li_nodes,real_lo_mapping,real_po_mapping,real_li_lo_mapping)
    stage_matrix = ga.initial_virtual_stage_matrix(netlist,new_netlist,virtual_and_nodes_sorted,lo_nodes,po_nodes)
    best_solution, best_solution_fitness = ga.genetic_algorithm(new_netlist,virtual_and_nodes_sorted, lo_nodes, po_nodes, li_nodes, virtual_longest_paths,virtual_lo_mapping,
                                                             virtual_po_mapping,stage_matrix,virtual_li_lo_mapping,
                                                             num_stages, population_size, num_generations, crossover_rate, mutation_rate, tournament_size)
    print("best_solution:", best_solution)
    print("best_solution_fitness:", best_solution_fitness)
    
    for node in new_netlist.graph.nodes():
        new_netlist.graph.nodes[node]["stage"] = -1
        
    best_solution = ga.process_lo_po(best_solution,virtual_and_nodes_sorted,lo_nodes,po_nodes,virtual_lo_mapping,virtual_po_mapping)
    
    
    
    node_partition_mapping = ga.assign_prenodes(best_solution, virtual_and_nodes_sorted, lo_nodes, po_nodes)
    for node_id, stage in node_partition_mapping.items():
        try:
            new_netlist.set_stage(node_id, stage)
        except ValueError as e:
            print(f"Skipping node {node_id}: {e}")
    
    real_stage_matrix = ga.real_successors(new_netlist,real_and_nodes_sorted,lo_nodes,po_nodes)
    _,max_scpl= ga.fitness_function(real_stage_matrix, real_longest_paths, real_and_nodes_sorted, lo_nodes, po_nodes, li_nodes,real_lo_mapping,real_po_mapping,real_li_lo_mapping)
    stage_dict = ga.assign_nodes(netlist, real_stage_matrix, real_and_nodes_sorted, lo_nodes, po_nodes, max_scpl)
    return stage_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline Optimization Configuration")
    # parser.add_argument("--module_name", type=str, required=True, help="Module name")
    parser.add_argument("--num_stages", type=int, default=3, help="Number of pipeline stages (0, 1, 2)")
    parser.add_argument("--population_size", type=int, default=30, help="Population size")
    parser.add_argument("--num_generations", type=int, default=50, help="Number of generations")
    parser.add_argument("--crossover_rate", type=float, default=0.5, help="Crossover probability")
    parser.add_argument("--mutation_rate", type=float, default=0.05, help="Mutation probability")
    parser.add_argument("--tournament_size", type=int, default=15, help="Tournament selection size")
    return parser.parse_args()

def random_stage_assignment(netlist, n_stages=3):
    netlist.n_stages = n_stages    
    # 重置所有节点阶段
    for node in netlist.graph.nodes:
        if netlist.get_type(node) in ["AND", "PO", "LO"]:
            netlist.graph.nodes[node]["stage"] = -1
    # 随机分配阶段
    for node in netlist.graph.nodes:
        node_type = netlist.get_type(node)
        if node_type in ["AND", "PO", "LO"]:
            stage = random.randint(0, n_stages-1)
            netlist.set_stage(node, stage)
    
    # 更新依赖信息
    netlist.calculate_stage_IO_info()
    netlist.determine_ctrl_io_id()
    netlist.topo_and_reverse_topo_sort()
    return netlist

def extract_features(netlist): #目前：提取netlist中LI和LO流水级分配情况
    features = {}
    
    # 提取LI信息
    for io_id, li_node in netlist.LI_id.items():
        stage = netlist.get_stage(li_node)
        features[f"LI_{io_id}_stage"] = stage
        
    # 提取LO信息
    for io_id, lo_node in netlist.LO_id.items():
        stage = netlist.get_stage(lo_node)
        features[f"LO_{io_id}_stage"] = stage

    return features

def train_xgboost_model(df):
    """训练XGBoost回归模型"""
    # 分离特征和标签
    X = df.drop("CPI", axis=1)
    y = df["CPI"]
    
    # 训练模型
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X, y)
    return model

def main():

    start_time = time.time()  
    args = parse_args()

    #model = ga.xgboost_train()

    # module_name = args.module_name
    module_name = "c17"

    # TODO：改成自己的路径
    #rtl_path = f"./data/rtl_input_4reg/alu_mod_{module_name}.v"
    rtl_path = f"/nfs_global/I/qimeng3/guoziying/qmlib_autopipe/python/data/rtl_input_4reg/alu_mod_{module_name}.v"
    netlist = r_parsor.rtl2netlist(
                rtl_path=rtl_path,
                clock_name="clk",
                rst_name="reset",
                rstn_name=None,
                top_name="top_module",
                draw_netlist=True,
            )
    original_netlist = copy.deepcopy(netlist)

    # 对原始网表的每个节点，设置阶段为-1，并设置网表阶段数为1
    for node in original_netlist.graph.nodes():
        original_netlist.graph.nodes[node]["stage"] = -1
    original_netlist.n_stages = 1

    netlist.reset()
    # 生成随机输入序列
    pi_fifo = [
            [random.choice([1, 0]) for i in range(netlist.PI_num)] for n in range(1000)
        ]
    # 保存PO和LO输出作为金标准
    golde_po, golden_lo = eval.execute_original_netlist(original_netlist, pi_fifo)
    netlist.reset()

    #[TODO]对原始网表进行多次随机阶段分配并计算CPI
    training_data = []
    num_samples = 100  # 样本数量，根据需要调整
    for _ in range(num_samples):
        netlist_copy = copy.deepcopy(original_netlist)
        # 随机分配阶段
        randomized_netlist = random_stage_assignment(netlist_copy, args.num_stages)

        # 提取LI和LO特征
        features = extract_features(randomized_netlist)

        # 执行网表计算CPI
        ppl_po, ppl_lo, cycles = eval.execute_pipelined_netlist(randomized_netlist, pi_fifo)
        cpi = cycles / 1000
        # 添加到训练数据集
        training_data.append({
            "features": features,
            "cpi": cpi
        })

    #[TODO]训练xgboost
    X = []
    y = []
    for sample in training_data:
        X.append(sample["features"])
        y.append(sample["cpi"])
    df = pd.DataFrame(X) # 将特征转换为DataFrame（一行）
    df["CPI"] = y
    model = train_xgboost_model(df)
    
    # 保存模型
    model.save_model("cpi_predictor.model")
    
    # 打印特征重要性
    importance = pd.DataFrame({
        "feature": model.feature_names_in_,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("==========================================================")
    print("特征重要性:")
    print(importance.head(10))
    print("==========================================================")
    #end TODO

    
    netlist.reset()
    
    stage_dict = process_main(netlist) #process_main
    for node_id , stage in stage_dict.items():
        netlist.set_stage(node_id,stage)
        
    netlist.determine_ctrl_io_id()
    netlist.calculate_stage_IO_info()
    print("==========================================================")
    for stage_id in range(netlist.n_stages):
        print(f"Stage {stage_id} LI: {netlist.stage_LI_list[stage_id]}")
        print(f"Stage {stage_id} LO: {netlist.stage_LO_list[stage_id]}")
    print("==========================================================")

    partition_test = {}
    for node in netlist.graph.nodes:
        partition_test[node] = netlist.graph.nodes[node]['stage']
        
    ppl_po, ppl_lo, cycles = eval.execute_pipelined_netlist(netlist, pi_fifo)
    for i in range(len(ppl_po)):
        if ppl_po[i] != golde_po[i]:
            print(f"PO {i} not equal, golden: {golde_po[i]}, pipeline: {ppl_po[i]}")
    for i in range(len(ppl_lo)):
        if ppl_lo[i] != golden_lo[i]:
            print(f"LO {i} not equal, golden: {golden_lo[i]}, pipeline: {ppl_lo[i]}")
    cpi = cycles / 1000
    print("CPI:",cpi)

    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f"Total execution time: {elapsed_time:.2f} seconds")  
       
if __name__ == "__main__":
    main()