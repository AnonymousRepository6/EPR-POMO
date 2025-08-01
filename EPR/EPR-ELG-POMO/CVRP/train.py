import torch
from torch.optim import Adam as Optimizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import yaml
import wandb
import datetime
import os
import json
from tqdm import trange

from generate_data import generate_vrp_data, VRPDataset
from CVRPModel import CVRPModel, CVRPModel_local
from CVRPEnv import CVRPEnv
from utils import rollout, check_feasible, Logger, seed_everything, hgs_solution, save_hgs_costs, load_hgs_costs, compute_cost_difference


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def test_rollout(loader, env, model):
    avg_cost = 0.
    num_batch = 0.
    for batch in loader:
        env.load_random_problems(batch)
        reset_state, _, _ = env.reset()
        model.eval()
        # greedy rollout
        with torch.no_grad():
            model.pre_forward(reset_state)
            solutions, probs, rewards = rollout(model=model, env=env, eval_type='greedy')
        # check feasible
        check_feasible(solutions[0:1], reset_state.node_demand[0:1])
        batch_cost = -rewards.max(1)[0].mean()
        avg_cost += batch_cost
        num_batch += 1.
    avg_cost /= num_batch

    return avg_cost

def validate(model, multiple_width, device, mixed=True):
    # Initialize env
    env = CVRPEnv(multi_width=multiple_width, device=device)

    if mixed:
        # validation dataset
        val_u = VRPDataset('CVRP/EPR-master/CVRP/data/vrp_uniform100_1000_seed1234.pkl', num_samples=1000)
        val_u_loader = DataLoader(val_u, batch_size=1000)
        val_c = VRPDataset('CVRP/EPR-master/CVRP/data/vrp_cluster100_1000_seed1234.pkl', num_samples=1000)
        val_c_loader = DataLoader(val_c, batch_size=1000)
        val_m = VRPDataset('CVRP/EPR-master/CVRP/data/vrp_mixed100_1000_seed1234.pkl', num_samples=1000)
        val_m_loader = DataLoader(val_m, batch_size=1000)

        # validate
        val_u_cost = test_rollout(val_u_loader, env, model)
        val_c_cost = test_rollout(val_c_loader, env, model)
        val_m_cost = test_rollout(val_m_loader, env, model)

        avg_cost_list = [val_u_cost.cpu().numpy().tolist(), 
                        val_c_cost.cpu().numpy().tolist(), 
                        val_m_cost.cpu().numpy().tolist()]
    else:
        # validation dataset
        val_100 = VRPDataset('CVRP/EPR-master/CVRP/data/vrp100_val.pkl', num_samples=1000)
        val_100_loader = DataLoader(val_100, batch_size=1000)
        val_200 = VRPDataset('CVRP/EPR-master/CVRP/data/vrp200_val.pkl', num_samples=1000)
        val_200_loader = DataLoader(val_200, batch_size=1000)
        val_500 = VRPDataset('CVRP/EPR-master/CVRP/data/vrp500_val.pkl', num_samples=100)
        val_500_loader = DataLoader(val_500, batch_size=10)

        # validate
        val_100_cost = test_rollout(val_100_loader, env, model)
        val_200_cost = test_rollout(val_200_loader, env, model)
        val_500_cost = test_rollout(val_500_loader, env, model)

        avg_cost_list = [val_100_cost.cpu().numpy().tolist(), 
                        val_200_cost.cpu().numpy().tolist(), 
                        val_500_cost.cpu().numpy().tolist()]
    return avg_cost_list
    

def train(model, training, T, start_steps, train_steps, mixed, train_batch_size, problem_size, distribution, multiple_width, lr, device, logger, scale_norm, fileLogger, dir_path, log_step, hgs_file_path):
    # Initialize env
    env = CVRPEnv(multi_width=multiple_width, device=device)
    distribution_ = distribution.copy()
    gaps = np.array([1, 1, 1])
    optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=1e-6)
    hgs_file_exists  = os.path.exists(hgs_file_path)
    # REINFORCE training
    for i in trange(train_steps - start_steps + 1):
        model.train()
        # Enable joint training
        if i == T - start_steps and training == 'joint':
            print("Enable joint training.")
            model.decoder.add_local_policy(device)
            optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=1e-6)
        
        if mixed:
            dis = np.random.choice(['uniform', 'cluster', 'mixed'], size=1, p=softmax(gaps))
            distribution_['data_type'] = dis
        else:
            distribution_['data_type'] = 'uniform'
        batch = generate_vrp_data(batch_size=train_batch_size, problem_size=problem_size, distribution=distribution_)

        env.load_random_problems(batch)
        reset_state, _, _ = env.reset()

        model.pre_forward(reset_state)
        solutions, probs, rewards = rollout(model=model, env=env, eval_type='sample')
        # solutions的shape为[train_batch, multiple_width100, 路径节点]
        # rewards的shape为[train_batch, multiple_width100
        # probs的shape为[train_batch, 路径节点， multiple_width100]

        # check feasible
        check_feasible(solutions[0:1], reset_state.node_demand[0:1])

        optimizer.zero_grad()
        # POMO
        bl_val = rewards.mean(dim=1)[:, None]  # [train_batch, 1]
        log_prob = probs.log().sum(dim=1)      # [train_batch, mul]
        advantage = rewards - bl_val           # [train_batch, mul]

        # HGS优化损失函数
        if i > 0 and (i == 1 or i % 1000 == 0):
            # if hgs_file_exists:
            #     hgs_costs = load_hgs_costs(hgs_file_path, i)
            # else:
            hgs_costs = hgs_solution(batch, device=device)  # hgs_costs为[train_batch, 100]并且是负数
            cost_difference, mask1 = compute_cost_difference(hgs_costs, rewards, advantage, weight=1.0)
            J = - (advantage + cost_difference * 2) * log_prob * mask1 +  (-advantage * log_prob) * ~mask1
        else:
            J = - advantage * log_prob
        if scale_norm:
            # J = J / (advantage ** 2).mean(dim=1)
            J = J / advantage.max(dim=1)[0][:, None]
        J = J.mean()

        # print("training length: {:.4f}".format(-rewards.max(1)[0].mean()))
        J.backward()
        optimizer.step()

        # validation and log
        if (i + 1) % log_step == 0:
            val_info = validate(model, multiple_width, device, mixed)
            fileLogger.log(val_info)
            if logger is not None:
                logger.log({'val_100_cost': val_info[0],
                            'val_300_cost': val_info[1],
                            'val_500_cost': val_info[2]},
                        step=i)   

            checkpoint_dict = {
                    'step': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
            torch.save(checkpoint_dict, dir_path + '/model_epoch_{}.pt'.format(int((i + 1) / log_step)))

            if mixed:
                # sample dis
                opts = np.array([15.740834, 7.909336, 14.294179])
                vals = np.array(val_info)
                gaps = (vals - opts) / opts

if __name__ == "__main__":
    with open('CVRP/EPR-master/CVRP/config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)

    # params
    name = config['name']
    device = "cuda:{}".format(config['cuda_device_num']) if config['use_cuda'] else 'cpu'
    logger_name = config['logger']
    load_checkpoint = config['load_checkpoint']
    hgs_file_path = config['hgs_file_path']
    problem_size = config['params']['problem_size']
    T = config['params']['T']
    distribution = config['distribution']
    multiple_width = config['params']['multiple_width']
    scale_norm = config['params']['scale_norm']
    start_steps = config['params']['start_steps']
    train_steps = config['params']['train_steps']
    mixed = config['params']['mixed']   
    train_batch_size = config['params']['train_batch_size']
    lr = config['params']['learning_rate']
    log_step = config['params']['log_step']
    model_params = config['model_params']

    seed_everything(config['seed'])
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f'-ts{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}'
    dir_path = 'CVRP/EPR-master/CVRP/weights/{}_{}_{}_{}'.format(name, ts_name, config['seed'], train_batch_size)
    os.mkdir(dir_path)

    log_config = config.copy()
    param_config = log_config['params'].copy()
    log_config.pop('params')
    model_params_config = log_config['model_params'].copy()
    log_config.pop('model_params')
    log_config.pop('distribution')
    log_config.update(param_config)
    log_config.update(model_params_config)
    # Initialize logger
    if(logger_name == 'wandb'):
        logger = wandb.init(project="EPR-CVRP",
                         name=name + ts_name,
                         config=log_config)
    else:
        logger = None
    # Initialize fileLogger
    filename = 'CVRP/EPR-master/CVRP/log/{}_{}'.format(name, ts_name)
    fileLogger = Logger(filename, config)

    # Initialize model and baseline
    model = CVRPModel(**model_params)
    if config['training'] == 'only_local':
        model = CVRPModel_local(**model_params)        
    
    if load_checkpoint is not None:
        checkpoint = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # model.decoder.change_local_policy()
    model.to(device)
    
    # Training
    train(model=model,
          training=config['training'],
          T=T,
          start_steps=start_steps,
          train_steps=train_steps,
          mixed=mixed,
          train_batch_size=train_batch_size, 
          problem_size=problem_size, 
          distribution=distribution,
          multiple_width=multiple_width, 
          lr=lr,
          device=device,
          logger=logger,
          scale_norm=scale_norm,
          fileLogger=fileLogger,
          dir_path=dir_path,
          log_step=log_step,
          hgs_file_path=hgs_file_path)