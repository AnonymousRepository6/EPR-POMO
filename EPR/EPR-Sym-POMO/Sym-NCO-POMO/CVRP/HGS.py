import torch
import numpy as np
import json
import random
import math
import hygese as hgs

# 读取 batch 并转换为求解格式
def prepare_vrp_data(depot, node_coords, node_demand, num_vehicles=100, vehicle_capacity=1.0):
    data = dict()
    coordinates = torch.cat((depot, node_coords), dim=0)
    demands = torch.cat((torch.tensor([0.0]), node_demand))

    # 计算距离矩阵
    num_nodes = len(coordinates)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                x1, y1 = coordinates[i][0], coordinates[i][1]
                x2, y2 = coordinates[j][0], coordinates[j][1]
                distance_matrix[i][j] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    data['coordinates'] = coordinates.tolist()
    data['demands'] = demands.tolist()
    data['distance_matrix'] = distance_matrix.tolist()
    data['depot'] = 0
    data['service_times'] = np.zeros(len(data['demands'])).tolist()
    data['num_vehicles'] = num_vehicles
    data['vehicle_capacity'] = vehicle_capacity

    return data

# HGS求解近似最优解
def hgs_solution(batch, device, num_vehicles=100, vehicle_capacity=1.0, time_limit=1):
    # 配置 HGS Solver
    ap = hgs.AlgorithmParameters(timeLimit=time_limit, targetFeasible=0.01, nbElite=75, lambda_=1)
    hgs_solver = hgs.Solver(parameters=ap, verbose=False)

    results = []
    costs = []
    for i in range(len(batch["depot"])):
        depot = batch["depot"][i]
        node_coords = batch["loc"][i]
        node_demand = batch["demand"][i]

        # 准备 VRP 数据
        vrp_data = prepare_vrp_data(depot, node_coords, node_demand, num_vehicles, vehicle_capacity)

        # 求解 VRP
        result = hgs_solver.solve_cvrp(vrp_data)
        results.append({
            "cost": -result.cost,
            "routes": result.routes
        })
        costs.append(-result.cost)
    costs = torch.tensor(costs, dtype=torch.float32).view(-1, 1).to(device)  # 转为 [batch, 1]
    return costs

import torch

# 保存hgs_costs数据
def save_hgs_costs(file_path, epoch, episode, hgs_costs):
    """
    保存 (epoch, episode) 和 hgs_costs 到文件中，数据以 Tensor 格式保存。
    """
    try:
        existing_data = torch.load(file_path)
    except FileNotFoundError:
        existing_data = {}

    # 使用 (epoch, episode) 作为键
    existing_data[(epoch, episode)] = hgs_costs

    torch.save(existing_data, file_path)


# 加载hgs_costs数据
def load_hgs_costs(file_path, epoch, episode):
    """
    根据 (epoch, episode) 从文件中加载对应的 hgs_costs，数据为 Tensor 格式。
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