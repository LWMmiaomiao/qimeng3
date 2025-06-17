import numpy as np
import copy
import random
import ap_utils.rtl_parsor as r_parsor
import ap_utils.evaluate as eval
import ap_utils.pipeline_search as psearch
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
import xgboost as xgb

def insert_virtual_nodes_into_netlist(netlist): #遍历所有输入节点（LI/PI）的后继节点，若后继是 AND 节点，则中间插入 AND 虚拟节点 
    #根据输入网表划分LI、PI输入节点
    new_netlist = copy.deepcopy(netlist)
    li_nodes = []
    pi_nodes = []
    for node_id in netlist.graph.nodes():
        node_type = netlist.graph.nodes[node_id]["type"]
        if node_type == "LI":
            li_nodes.append(node_id)
        elif node_type == "PI":
            pi_nodes.append(node_id)
    
    edges_to_replace = []
    #遍历所有输入节点（LI/PI）的后继节点，若后继是 AND 节点，则将该边标记为需要替换
    for node_id in (li_nodes + pi_nodes):
        successors = list(netlist.graph.successors(node_id)) # 获取当前节点的所有直接后继
        for succ_id in successors:
            if netlist.graph.nodes[succ_id]["type"] == "AND":
                edges_to_replace.append((node_id, succ_id))
                
    for (src, dst) in edges_to_replace:
        virtual_node_name = f"{src}-{dst}"
        
        if virtual_node_name in new_netlist.graph:
            continue
        
        new_netlist.add_node("AND", virtual_node_name)
        
        if new_netlist.graph.has_edge(src, dst):
            new_netlist.graph.remove_edge(src, dst)
        
        new_netlist.add_edge(src, virtual_node_name,'DIRECT')
        new_netlist.add_edge(virtual_node_name, dst,'DIRECT')
    new_netlist.topo_and_reverse_topo_sort()
    return new_netlist

def compute_dp_for_target(t, nodes, edges, top_order): # 动态规划计算从所有 AND 节点到该目标的最长路径长度
    reverse_g = nx.DiGraph()
    reverse_g.add_nodes_from(nodes)
    reverse_g.add_edges_from(edges)
    dp = {node: -float('inf') for node in nodes}
    dp[t] = 0
    for u in top_order:
        for v in reverse_g.predecessors(u):
            if dp[v] + 1 > dp[u]:
                dp[u] = dp[v] + 1
    return (t, dp)
    
def extract_graph_info(netlist,value='True'):
    li_nodes = set()
    pi_nodes = set()
    lo_nodes = set()
    po_nodes = set()
    for node_id in netlist.graph.nodes():
        node_type = netlist.graph.nodes[node_id]["type"]
        if node_type == "LI":
            li_nodes.add(node_id)
        elif node_type == "PI":
            pi_nodes.add(node_id)
        elif node_type == "LO":
            lo_nodes.add(node_id)
        elif node_type == "PO":
            po_nodes.add(node_id)
        else:
            pass

    li_pi_successors = set() # PI/LI 的后续节点 ADD
    li_pi_mapping = {} # 有后续节点的 PI/LI ——map—— 后续节点ADD
    for node_id in li_nodes.union(pi_nodes):
        successors = netlist.graph.successors(node_id)
        for succ_id in successors:
            if netlist.graph.nodes[succ_id]["type"] == "AND":
                li_pi_successors.add(succ_id)
                if node_id not in li_pi_mapping:
                    li_pi_mapping[node_id] = []
                li_pi_mapping[node_id].append(succ_id)

    and_nodes_sorted = sorted(li_pi_successors, key=lambda x: netlist.get_topological_sort().index(x))

    reverse_g = netlist.graph.reverse() # 反转边方向
    top_order = list(nx.topological_sort(reverse_g)) # 从输出节点（LO/PO）向输入方向遍历，计算最长路径
    nodes = list(reverse_g.nodes())
    edges = list(reverse_g.edges())

    lo_po_nodes = list(lo_nodes) + list(po_nodes)
    n = len(and_nodes_sorted)
    m = len(lo_po_nodes)
    longest_paths = np.zeros((n, m))

    longest_path_dict = {}
    with ProcessPoolExecutor() as executor:
        args = [(t, nodes, edges, top_order) for t in lo_po_nodes]
        futures = [executor.submit(compute_dp_for_target, *arg) for arg in args]
        for future in futures:
            t, dp = future.result()
            longest_path_dict[t] = dp

    for i, and_node in enumerate(and_nodes_sorted):
        for j, t in enumerate(lo_po_nodes):
            path_len = longest_path_dict[t].get(and_node, -float('inf'))
            if value == 'True':
                longest_paths[i, j] = max(path_len, 0) #直接使用原始路径长度
            else :
                longest_paths[i, j] = max(path_len-1,0) #插入虚拟节点后调整路径计数-1

    lo_mapping = {lo: [] for lo in lo_nodes}
    po_mapping = {po: [] for po in po_nodes}
    #记录每个 LO/PO 节点可通过哪些 AND 节点到达
    for i, and_node in enumerate(and_nodes_sorted):
        for j, t in enumerate(lo_po_nodes):
            if longest_paths[i, j] > 0:  
                if j < len(lo_nodes):
                    lo_mapping[t].append(and_node) 
                else:
                    po_mapping[t].append(and_node) 
    # MAP：LO——LI——ADD
    li_lo_mapping = {lo: [] for lo in lo_nodes}
    for node in lo_nodes:
        LI_id = netlist.get_io_id_from_LO(node)
        li_node = netlist.get_LI_from_io_id(LI_id)
        succ_node = li_pi_mapping[li_node]
        
        li_lo_mapping[node] = succ_node
                    
    
    
    return (li_nodes, pi_nodes, and_nodes_sorted, lo_nodes, po_nodes, 
            longest_paths, li_pi_mapping,lo_mapping,po_mapping,li_lo_mapping)

def dfs_longest_path(graph, node, target, visited, path_length):
    if node in visited:
        return 0

    visited.add(node)

    if node == target:
        return path_length

    max_path = 0
    for neighbor in graph.successors(node):
        max_path = max(max_path, dfs_longest_path(graph, neighbor, target, visited, path_length + 1))

    visited.remove(node)
    return max_path

def calculate_longest_path(graph, from_node, to_node):
    visited = set()
    return dfs_longest_path(graph, from_node, to_node, visited, 0)

def calculate_deta_s_and_scpl(stage_matrix, longest_paths, and_nodes, lo_nodes, po_nodes,lo_mapping,po_mapping):
    n = len(and_nodes)
    m = len(lo_nodes) + len(po_nodes)
    matrix = stage_matrix
    
    deta_s = np.zeros((n, m))
    scpl = np.zeros((n, m))
    and_index = {node: idx for idx, node in enumerate(and_nodes)} # AND节点索引：0, 1, ..., n-1
    lo_index = {node: idx + len(and_nodes) for idx, node in enumerate(lo_mapping)} # LO节点索引：n, n+1, ..., n+m_lo-1
    po_index = {node: idx + len(and_nodes) + len(lo_mapping) for idx, node in enumerate(po_mapping)} # PO节点索引：n+m_lo, ..., n+m_lo+m_po-1

    for lo,and_nodes_num in lo_mapping.items():
        if and_nodes_num:
            stages = [stage_matrix[and_index[node],0] for node in and_nodes_num if node in and_index]
            if stages:
                max_stage = max(stages)
                if lo in lo_index:
                    idx = lo_index[lo]
                    stage = min(stage_matrix[idx,0] + max_stage,2) # LO节点的阶段由其驱动AND节点的最大阶段决定，并限制最大阶段数为2
                    matrix[idx,0] = stage

    for po,and_nodes_num in po_mapping.items():
        if and_nodes_num:
            stages = [stage_matrix[and_index[node],0] for node in and_nodes_num if node in and_index]
            if stages:
                max_stage = max(stages)
                if po in po_index:
                    idx = po_index[po]
                    stage = min(stage_matrix[idx,0] + max_stage,2)
                    matrix[idx,0] = stage
    for i, and_node in enumerate(and_nodes):
        for j, lo_po_node in enumerate(list(lo_nodes) + list(po_nodes)):
            
            deta_s[i,j] = matrix[n+j] - matrix[i] + 1
            if longest_paths[i,j] == 0:
                scpl[i,j] = 0
            else:
                scpl[i, j] = math.ceil(longest_paths[i, j] / deta_s[i, j])

    max_scpl = np.max(scpl)
    return max_scpl, deta_s, scpl


def calculate_rms(deta_s, lo_nodes,li_lo_mapping,and_nodes_sorted):
    lo_list = sorted(list(lo_nodes))
    average_sum = 0
    
    for lo in lo_list:
        squared_sum = 0
        and_nodes = li_lo_mapping[lo]
        row_indices = [and_nodes_sorted.index(node) for node in and_nodes]
        col_idx = lo_list.index(lo)
        
        if row_indices:
            values = deta_s[row_indices,col_idx]
            squard_values = values**2
            squared_sum = np.sum(squard_values)
            average_sum += pow(squared_sum / len(and_nodes),2) 
    return np.sqrt(average_sum/len(lo_nodes))

def fitness_function(stage_matrix, longest_paths, and_nodes_sorted, lo_nodes, po_nodes, li_nodes,lo_mapping,po_mapping,li_lo_mapping):
    max_scpl, deta_s, scpl = calculate_deta_s_and_scpl(stage_matrix, longest_paths, and_nodes_sorted, lo_nodes, po_nodes,lo_mapping,po_mapping)
    rms = calculate_rms(deta_s,lo_nodes,li_lo_mapping,and_nodes_sorted)
    fitness =  max_scpl * rms
    
    return fitness,max_scpl

def fitness_xgboost(stage_matrix, longest_paths, and_nodes_sorted, lo_nodes, po_nodes, li_nodes,lo_mapping,po_mapping,li_lo_mapping,model):
    max_scpl, deta_s, scpl = calculate_deta_s_and_scpl(stage_matrix, longest_paths, and_nodes_sorted, lo_nodes, po_nodes,lo_mapping,po_mapping)
    rms = calculate_rms(deta_s,lo_nodes,li_lo_mapping,and_nodes_sorted)
    with open("output.txt", "a") as f:
        print("####", file=f)
        print("max_scpl:", max_scpl, file=f)
        print("rms:", rms, file=f)
    fitness =  max_scpl * rms #(model.predict(np.array([[max_scpl]]))[0]) * (model.predict(np.array([[rms]]))[0])
    with open("output.txt", "a") as f:
        print("fitness:", fitness, file=f)
    
    return fitness

def xgboost_train():
    # 生成1000个符合 y = x 的样本（添加轻微噪声防止过拟合警告）
    X = np.random.rand(3000, 1) * 10  # 生成0-100之间的随机数
    y = X.ravel() # y = x + 微小噪声

    # 创建模型（关键参数配置）
    model = xgb.XGBRegressor(
        max_depth=5,        # 限制树复杂度
        n_estimators=100,    # 增加树的数量
        learning_rate=1.0,  # 提高学习率
        objective='reg:squarederror'
    )

    # 训练模型
    model.fit(X, y)

    # 评估训练误差
    train_pred = model.predict(X)
    mse = np.mean((train_pred - y) ** 2)
    print(f"[XGBOOST]:Training MSE: {mse:.10f}")  # 预期输出接近0
    return model


def initialize_partition(netlist, and_nodes_sorted, lo_nodes, po_nodes):
    node_order = list(and_nodes_sorted) + list(lo_nodes) + list(po_nodes)
    stage_array = []
    for node in node_order:
        stage = netlist.graph.nodes[node]["stage"]
        stage_array.append(stage)
    
    stage_matrix = np.array(stage_array, dtype=int).reshape(-1, 1)

    return stage_matrix

def initial_virtual_stage_matrix(netlist,new_netlist,virtual_and_nodes_sorted,lo_nodes,po_nodes):
    for node in virtual_and_nodes_sorted:
        succ = new_netlist.get_successors(node)
        stage = netlist.graph.nodes[succ[0]]["stage"]
        new_netlist.set_stage(node, stage)
    stage_array = []
    
    and_node_order = list(virtual_and_nodes_sorted)
    
    for node in and_node_order:
        stage = new_netlist.graph.nodes[node]["stage"]
        stage_array.append(stage)
        
    lo_po_nodes = list(lo_nodes)+list(po_nodes)
    for node in lo_po_nodes:
        stage = netlist.graph.nodes[node]["stage"]
        stage_array.append(stage)
    stage_matrix = np.array(stage_array, dtype=int).reshape(-1, 1)
    return stage_matrix


def initialize_virtual_partition(netlist, and_nodes_sorted, lo_nodes, po_nodes):
    psearch.stage_assignment_align_with_old(netlist,3)

    node_order = list(and_nodes_sorted) + list(lo_nodes) + list(po_nodes)
    
    stage_array = []
    for node in node_order:
        stage = netlist.graph.nodes[node]["stage"]
        stage_array.append(stage)
    
    stage_matrix = np.array(stage_array, dtype=int).reshape(-1, 1)

    return stage_matrix

def generate_initial_population(pop_size, netlist,and_nodes_sorted, lo_nodes, po_nodes,stage_matrix):
    population = []
    population.append(stage_matrix)
    
    for _ in range(pop_size-1):
        partition_matrix = initialize_virtual_partition(
            netlist,and_nodes_sorted, lo_nodes, po_nodes
        )
        population.append(partition_matrix)
    return population

def tournament_selection(population, fitnesses, tournament_size):
    indices = np.random.choice(len(population), tournament_size, replace=False)
    best_index = indices[0]
    best_fitness = fitnesses[indices[0]]
    for idx in indices[1:]:
        if fitnesses[idx] < best_fitness:
            best_fitness = fitnesses[idx]
            best_index = idx
    return population[best_index].copy()

def one_point_crossover(parent1, parent2):
    flat1 = parent1.flatten()
    flat2 = parent2.flatten()   
    point = np.random.randint(1, len(flat1))
    child1_flat = np.concatenate((flat1[:point], flat2[point:]))
    child2_flat = np.concatenate((flat2[:point], flat1[point:]))
    child1 = child1_flat.reshape(parent1.shape)
    child2 = child2_flat.reshape(parent1.shape)
    return child1, child2


def mutation(individual, mutation_rate, num_stages):
    random_values = np.random.rand(individual.shape[0])
    mutation_mask = random_values < mutation_rate
    individual[mutation_mask, 0] = np.random.randint(0, num_stages, mutation_mask.sum())
    return individual

def genetic_algorithm(netlist,and_nodes_sorted, lo_nodes, po_nodes, li_nodes, longest_paths,lo_mapping,po_mapping,stage_matrix,li_lo_mapping,num_stages, population_size, num_generations, crossover_rate, mutation_rate, tournament_size):
    
    population = generate_initial_population(population_size,netlist, and_nodes_sorted, lo_nodes, po_nodes,stage_matrix)
    best_individual = None
    best_fitness = float('inf')
    
    for generation in tqdm(range(num_generations), desc="Genetic Algorithm Progress"):
        fitnesses = []
        for ind in population:
            f , _= fitness_function(ind, longest_paths, and_nodes_sorted, lo_nodes, po_nodes, li_nodes,lo_mapping,po_mapping,li_lo_mapping)
            fitnesses.append(f)
        fitnesses = np.array(fitnesses)
        
        gen_best_index = np.argmin(fitnesses)
        if fitnesses[gen_best_index] < best_fitness:
            best_fitness = fitnesses[gen_best_index]
            
            best_individual = population[gen_best_index].copy()
        
        print(f"Generation {generation}, Best Fitness: {best_fitness} ")        
        new_population = []
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses,tournament_size)
            parent2 = tournament_selection(population, fitnesses,tournament_size)
            if np.random.rand() < crossover_rate:
                child1, child2 = one_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
                
            child1 = mutation(child1,mutation_rate,num_stages)
            child2 = mutation(child2,mutation_rate,num_stages)
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)
        
        population = new_population

    return best_individual, best_fitness

def process_lo_po(best_solution,and_nodes_sorted,lo_nodes,po_nodes,lo_mapping,po_mapping):
    
    and_index = {node: idx for idx, node in enumerate(and_nodes_sorted)}
    lo_index = {node: idx + len(and_nodes_sorted) for idx, node in enumerate(lo_mapping)}
    po_index = {node: idx + len(and_nodes_sorted) + len(lo_mapping) for idx, node in enumerate(po_mapping)}
    for lo,and_nodes_num in lo_mapping.items():
        if and_nodes_num:
            stages = [best_solution[and_index[node],0] for node in and_nodes_num if node in and_index]
            if stages:
                max_stage = max(stages)
                if lo in lo_index:
                    idx = lo_index[lo]
                    stage = min(best_solution[idx,0] + max_stage,2)
                    best_solution[idx,0] = stage
    for po,and_nodes_num in po_mapping.items():
        if and_nodes_num:
            stages = [best_solution[and_index[node],0] for node in and_nodes_num if node in and_index]
            if stages:
                max_stage = max(stages)
                if po in po_index:
                    idx = po_index[po]
                    stage = min(best_solution[ idx,0] + max_stage,2)
                    best_solution[idx,0] = stage
    return best_solution
    

def assign_prenodes(best_solution, and_nodes_sorted, lo_nodes, po_nodes):
    node_partition_mapping = {}
    for idx, node in enumerate(and_nodes_sorted):
        node_partition_mapping[node] = best_solution[idx, 0]
    for idx, node in enumerate(lo_nodes):
        node_partition_mapping[node] = best_solution[len(and_nodes_sorted) + idx, 0]
    for idx, node in enumerate(po_nodes):
        node_partition_mapping[node] = best_solution[len(and_nodes_sorted) + len(lo_nodes) + idx, 0]
    for node, partition in node_partition_mapping.items():
        if isinstance(partition, np.ndarray):
            node_partition_mapping[node] = partition.item()  
    return node_partition_mapping

def real_successors(new_netlist,real_and_nodes_sorted,lo_nodes,po_nodes):
    for node in real_and_nodes_sorted:
        pred = new_netlist.get_predecessors(node)
        max_stage = max([new_netlist.graph.nodes[p]["stage"] for p in pred])
        new_netlist.graph.nodes[node]["stage"] = max_stage
        
    node_order = list(real_and_nodes_sorted) + list(lo_nodes) + list(po_nodes)
    stage_array = []
    for node in node_order:
        stage = new_netlist.graph.nodes[node]["stage"]
        stage_array.append(stage)
    
    real_stage_matrix = np.array(stage_array, dtype=int).reshape(-1, 1)
    return real_stage_matrix

def assign_nodes(netlist, real_stage_matrix, real_and_nodes_sorted, lo_nodes, po_nodes, max_scpl):
    topo = netlist.get_topological_sort()
    assigned_nodes = list(real_and_nodes_sorted)  
    assigned_nodes.extend(list(lo_nodes))
    assigned_nodes.extend(list(po_nodes))     
    #尚未分配阶段的节点（不在real_and_nodes_sorted、lo_nodes、po_nodes中的节点），根据它们的前驱节点的阶段来分配阶段
    remaining_nodes = [
        node for node in topo 
        if node not in assigned_nodes and netlist.graph.nodes[node]["type"] not in {"PI", "LI", "CONST0"}
    ]
    
    level_dict = {}
    for idx, node in enumerate(assigned_nodes):
        stage = real_stage_matrix[idx, 0]  
        level_dict[node] = max_scpl * stage  
    
    #获取其所有前驱节点，并找出前驱节点中最大的层级.该节点的层级为这个最大层级加1（表示增加一个逻辑门延迟）。
    for node in remaining_nodes:
        predecessors = netlist.get_predecessors(node)
        max_predecessor_level = max([level_dict[pred] for pred in predecessors if pred in level_dict], default=0)
        level_dict[node] = max_predecessor_level + 1    
        
    stage_dict = {}
    for node, level in level_dict.items():
        stage = int(level // max_scpl)  
        stage_dict[node] = stage        
    return stage_dict #节点到阶段的映射字典


def process_main(netlist):
    num_stages = 3
    population_size = 200
    num_generations = 50
    crossover_rate = 0.5
    mutation_rate = 0.05
    tournament_size = 150
    
    li_nodes, pi_nodes, real_and_nodes_sorted, lo_nodes, po_nodes, real_longest_paths,real_li_pi_mapping,real_lo_mapping,real_po_mapping,real_li_lo_mapping = extract_graph_info(netlist)
    new_netlist = insert_virtual_nodes_into_netlist(netlist)
    li_nodes, pi_nodes, virtual_and_nodes_sorted, lo_nodes, po_nodes, virtual_longest_paths,virtual_li_pi_mapping,virtual_lo_mapping,virtual_po_mapping,virtual_li_lo_mapping = extract_graph_info(new_netlist,value='False')
    psearch.stage_assignment_align_with_old(netlist,3)
    origin_stage_matrix = initialize_partition(netlist, real_and_nodes_sorted, lo_nodes, po_nodes)
    origin_fit,max_scpl = fitness_function(origin_stage_matrix, real_longest_paths, real_and_nodes_sorted, lo_nodes, po_nodes, li_nodes,real_lo_mapping,real_po_mapping,real_li_lo_mapping)
    stage_matrix = initial_virtual_stage_matrix(netlist,new_netlist,virtual_and_nodes_sorted,lo_nodes,po_nodes)
    
    

    best_solution, best_solution_fitness = genetic_algorithm(new_netlist,virtual_and_nodes_sorted, lo_nodes, po_nodes, li_nodes, virtual_longest_paths,virtual_lo_mapping,
                                                             virtual_po_mapping,stage_matrix,virtual_li_lo_mapping,
                                                             num_stages, population_size, num_generations, crossover_rate, mutation_rate, tournament_size)
    print("best_solution_fitness:", best_solution_fitness)
    for node in new_netlist.graph.nodes():
        new_netlist.graph.nodes[node]["stage"] = -1
    best_solution = process_lo_po(best_solution,virtual_and_nodes_sorted,lo_nodes,po_nodes,virtual_lo_mapping,virtual_po_mapping)
    node_partition_mapping = assign_prenodes(best_solution, virtual_and_nodes_sorted, lo_nodes, po_nodes)
    for node_id, stage in node_partition_mapping.items():
        try:
            new_netlist.set_stage(node_id, stage)
        except ValueError as e:
            print(f"Skipping node {node_id}: {e}")
    real_stage_matrix = real_successors(new_netlist,real_and_nodes_sorted,lo_nodes,po_nodes)
    fit,max_scpl= fitness_function(real_stage_matrix, real_longest_paths, real_and_nodes_sorted, lo_nodes, po_nodes, li_nodes,real_lo_mapping,real_po_mapping,real_li_lo_mapping)
    stage_dict = assign_nodes(netlist, real_stage_matrix, real_and_nodes_sorted, lo_nodes, po_nodes, max_scpl)
    
    return stage_dict
