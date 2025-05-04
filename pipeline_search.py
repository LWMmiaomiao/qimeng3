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


def main():
    start_time = time.time()  
    args = parse_args()

    model = ga.xgboost_train()

    # module_name = args.module_name
    module_name = "c17"
    num_stages = args.num_stages #流水线的阶段数？
    population_size = args.population_size #[算法]种群个体数量
    num_generations = args.num_generations #[算法]进化迭代次数
    crossover_rate = args.crossover_rate #[算法]交叉概率
    mutation_rate = args.mutation_rate #[算法]变异概率
    tournament_size = args.tournament_size #[算法]选择机制中每轮参赛个体数，影响选择压力

    # TODO：改成自己的路径
    rtl_path = f"./data/rtl_input_4reg/alu_mod_{module_name}.v"
    # rtl_path = f"/workspace/S/wangqicheng1/gate_level_auto_pipeline/benchmark/ITC_99/{module_name}.v"
    netlist = r_parsor.rtl2netlist(
                rtl_path=rtl_path,
                clock_name="clk",
                rst_name="reset",
                rstn_name=None,
                top_name="top_module",
                draw_netlist=True,
            )
    
    # netlist_dir = f"/workspace/S/wangqicheng1/qmlib_autopipe/data/itc/{module_name}.pipeline.netlist"
    # netlist = pickle.load(open(netlist_dir, "rb"))
    # netlist_path = f"/workspace/S/wangqicheng1/gate_level_auto_pipeline/benchmark/ITC_99/{module_name}.pipeline.netlist"
    # netlist = pickle.load(open(netlist_path, "rb"))
    # netlist_dir = f"/workspace/S/wangqicheng1/qmlib_autopipe/data/rtllm/verified_{module_name}.pipeline.netlist"
    # netlist = pickle.load(open(netlist_dir, "rb"))

    print("finish read netlist")
    li_nodes, pi_nodes, real_and_nodes_sorted, lo_nodes, po_nodes, real_longest_paths,real_li_pi_mapping,real_lo_mapping,real_po_mapping,real_li_lo_mapping = ga.extract_graph_info(netlist)
    print("li_nodes:", li_nodes)
    print("pi_nodes:", pi_nodes)
    print("real_and_nodes_sorted:", real_and_nodes_sorted)
    print("lo_nodes:", lo_nodes)
    print("po_nodes:", po_nodes)
    print("real_longest_paths:", real_longest_paths)
    print("real_li_pi_mapping:", real_li_pi_mapping)
    print("real_lo_mapping:", real_lo_mapping)
    print("real_po_mapping:", real_po_mapping)
    print("real_li_lo_mapping:",real_li_lo_mapping)

    new_netlist = ga.insert_virtual_nodes_into_netlist(netlist) #遍历所有输入节点（LI/PI）的后继节点，若后继是 AND 节点，则中间插入 AND 虚拟节点 
    li_nodes, pi_nodes, virtual_and_nodes_sorted, lo_nodes, po_nodes, virtual_longest_paths,virtual_li_pi_mapping,virtual_lo_mapping,virtual_po_mapping,virtual_li_lo_mapping = ga.extract_graph_info(new_netlist,value='False')
    print("li_nodes:", li_nodes)
    print("pi_nodes:", pi_nodes)
    print("virtual_and_nodes_sorted:", virtual_and_nodes_sorted)
    print("lo_nodes:", lo_nodes)
    print("po_nodes:", po_nodes)
    print("virtual_longest_paths:", virtual_longest_paths)
    print("virtual_li_pi_mapping:", virtual_li_pi_mapping)
    print("virtual_lo_mapping:", virtual_lo_mapping)
    print("virtual_po_mapping:", virtual_po_mapping)
    print("virtual_li_lo_mapping:",virtual_li_lo_mapping)
    
    psearch.stage_assignment_align_with_old(netlist,3)
    origin_stage_matrix = ga.initialize_partition(netlist, real_and_nodes_sorted, lo_nodes, po_nodes)
    origin_fit = ga.fitness_function(origin_stage_matrix, real_longest_paths, real_and_nodes_sorted, lo_nodes, po_nodes, li_nodes,real_lo_mapping,real_po_mapping,real_li_lo_mapping,model)
    print("origin_stage_matrix:",origin_stage_matrix)
    print("origin_fitness:",origin_fit)
    
    stage_matrix = ga.initial_virtual_stage_matrix(netlist,new_netlist,virtual_and_nodes_sorted,lo_nodes,po_nodes)
    print("stage_matrix:",stage_matrix)

    best_solution, best_solution_fitness = ga.genetic_algorithm(new_netlist,virtual_and_nodes_sorted, lo_nodes, po_nodes, li_nodes, virtual_longest_paths,virtual_lo_mapping,
                                                             virtual_po_mapping,stage_matrix,virtual_li_lo_mapping,
                                                             num_stages, population_size, num_generations, crossover_rate, mutation_rate, tournament_size, model)
    print("best_solution:", best_solution)
    print("best_solution_fitness:", best_solution_fitness)
    
    for node in new_netlist.graph.nodes():
        new_netlist.graph.nodes[node]["stage"] = -1
    
    # assign LO\PO nodes 处理锁存器输出（LO）和主输出（PO）节点的阶段分配
    best_solution = ga.process_lo_po(best_solution,virtual_and_nodes_sorted,lo_nodes,po_nodes,virtual_lo_mapping,virtual_po_mapping)
    
    node_partition_mapping = ga.assign_prenodes(best_solution, virtual_and_nodes_sorted, lo_nodes, po_nodes) #根据最佳解生成节点到阶段的映射
    print("node_partition_mapping:",node_partition_mapping)
    for node_id, stage in node_partition_mapping.items():
        try:
            new_netlist.set_stage(node_id, stage)
        except ValueError as e:
            print(f"Skipping node {node_id}: {e}")
    
    real_stage_matrix = ga.real_successors(new_netlist,real_and_nodes_sorted,lo_nodes,po_nodes)
    print("real_stage_matrix:",real_stage_matrix)
    
    fit = ga.fitness_function(real_stage_matrix, real_longest_paths, real_and_nodes_sorted, lo_nodes, po_nodes, li_nodes,real_lo_mapping,real_po_mapping,real_li_lo_mapping,model)
    print("===========================================")
    print("origin_fitness:",origin_fit)
    print("fit:",fit)
    print(f"Improvement rate: {origin_fit/fit:.3f}")  
    print("===========================================")

    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f"Total execution time: {elapsed_time:.2f} seconds")  
       
if __name__ == "__main__":
    main()