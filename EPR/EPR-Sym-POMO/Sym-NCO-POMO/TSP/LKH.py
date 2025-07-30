import torch
import numpy as np
import lkh
import time
import os

# LKH3引入
def generate_tsp_string(coords, name="instance"):
    dimension = len(coords)
    header = f"""NAME : {name}
COMMENT : {dimension}-city problem
TYPE : TSP
DIMENSION : {dimension}
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
"""
    nodes_str = ""
    for idx, (x, y) in enumerate(coords, start=1):
        nodes_str += f"{idx} {x} {y}\n"
    nodes_str += "EOF\n"
    return header + nodes_str

def extract_sections(input_string):
    lines = input_string.strip().split('\n')
    coordinates = {}
    current_section = None
    for line in lines:
        line = line.strip()
        if line.startswith("NODE_COORD_SECTION"):
            current_section = "coordinates"
        elif line.startswith("EOF"):
            break
        elif current_section == "coordinates" and line:
            parts = line.split()
            if len(parts) == 3:
                node_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                coordinates[node_id] = (x, y)
    return coordinates

def euclidean_distance(p1, p2):
    return round(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

def calculate_total_cost(routes, coordinates):
    total_cost = 0
    depot = coordinates[1]
    for route in routes:
        current_node = depot
        route_cost = 0
        for node in route:
            next_node = coordinates[node]
            route_cost += euclidean_distance(current_node, next_node)
            current_node = next_node
        route_cost += euclidean_distance(current_node, depot)  # Return to depot
        total_cost += route_cost
    return total_cost

def LKH_solution(batch, device, max_trials=3000, runs=20):
    """
    批量求解 TSP 问题

    参数：
        batch: numpy array，shape 为 (batch_size, n_cities, 2)
        device: 字符串，仅做占位（与LKH无关）
        max_trials: LKH最大尝试次数
        runs: LKH运行次数

    返回：
        result_list: 一个列表，每个元素是一个字典，包含 'cost' 和 'routes'
    """
    solver_path = "/home/lining/CVRP/LKH-3.0.13/LKH"  # 请确保路径正确
    result_list = []
    costs = []
    start_time = time.time()
    for i, coords in enumerate(batch):
        tsp_str = generate_tsp_string(coords, name=f"instance_{i}")
        problem = lkh.LKHProblem.parse(tsp_str)
        routes = lkh.solve(solver_path, problem=problem, max_trials=max_trials, runs=runs)

        coordinates = extract_sections(tsp_str)
        total_cost = calculate_total_cost(routes, coordinates)

        result_list.append({
            "cost": -total_cost,
            "routes": routes
        })

    costs.append(-total_cost)
    costs = torch.tensor(costs, dtype=torch.float32).view(-1, 1).to(device)
    return costs

def save_lkh3_result(file_path, epoch, episode, lkh_result):
    """
    保存编号 i 的 LKH3 结果（包括 cost 和 routes）到文件。

    参数:
        file_path (str): 保存的文件路径 (.pt)
        i (int): 当前实例编号
        lkh_result (dict): 包含 "cost" 和 "routes" 的字典
    """
    try:
        existing_data = torch.load(file_path)
    except FileNotFoundError:
        existing_data = {}

    existing_data[(epoch, episode)] = lkh_result
    torch.save(existing_data, file_path)

def load_lkh3_result(file_path, epoch, episode):
    """
    加载编号 i 的 LKH3 结果（cost 和 routes）从文件中。

    参数:
        file_path (str): 保存的文件路径 (.pt)
        i (int): 实例编号

    返回:
        dict: 包含 "cost" 和 "routes" 的字典
    """
    data = torch.load(file_path)
    key = (epoch, episode)
    if key in data:
        return data[key]
    else:
        raise KeyError(f"Key {key} not found in the saved data.")

# 分类高质量和低质量解
def split_solutions_by_rewards(rewards, solutions, bl_val):
    higher = []
    lower = []

    for i in range(rewards.shape[0]):
        for j in range(rewards.shape[1]):
            if rewards[i, j] > bl_val[i, 0]:
                higher.append(solutions[i, j])
            elif rewards[i, j] < bl_val[i, 0]:
                lower.append(solutions[i, j])

    return higher, lower

# 截取子串及翻转
def generate_substrings_with_reverse(solution, K):
    substrings = []
    start = 0

    for i in range(len(solution)):
        if solution[i] == 0:
            if start < i:
                segment = solution[start:i]
                for j in range(len(segment) - K + 1):
                    subseq = segment[j:j+K]
                    substrings.append(subseq)
                    substrings.append(subseq[::-1])
            start = i + 1

    # 处理末尾不是 0 的情况
    if start < len(solution):
        segment = solution[start:]
        for j in range(len(segment) - K + 1):
            subseq = segment[j:j+K]
            substrings.append(subseq)
            substrings.append(subseq[::-1])

    return substrings

# 计算HGS指导幅度
def compute_cost_difference(hgs_costs, rewards, advantage, weight=1.0):
    cost_difference = hgs_costs - rewards
    cost_difference = weight / (cost_difference + 1e-6)
    mask1 = (advantage > 0) & (cost_difference > 0)
    return cost_difference, mask1