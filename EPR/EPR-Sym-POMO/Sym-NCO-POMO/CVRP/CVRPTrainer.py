import torch
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model
from CVRPModel_ours import CVRPModel as Model_ours

from CVRPTester import CVRPTester as Tester

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from HGS import hgs_solution, save_hgs_costs, load_hgs_costs

from utils.utils import *
# import wandb

# wandb.init(project="cvrp_ablation", entity="alstn12088")



class CVRPTrainer:
    def __init__(self,
                 env_params, env_test_params,
                 model_params,
                 optimizer_params,
                 trainer_params, tester_params):

        # save arguments
        self.env_params = env_params
        self.env_test_params = env_test_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.tester_params = tester_params
        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components

        if self.trainer_params['is_pomo']:
            self.model = Model(**self.model_params)
        else:
            self.model = Model_ours(**self.model_params)
        self.env = Env(**self.env_params)
        self.env_test = Env(**self.env_test_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch'] - 1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']


            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))
            a,b,c = self._test()

            # wandb.log({"greedy": a})
            # wandb.log({"pomo": b})
            # wandb.log({"pomo_aug": c})


    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            hgs_file_path = self.trainer_params['hgs_file_path']
            cuda_device_num = self.trainer_params['cuda_device_num']

            avg_score, avg_loss = self._train_one_batch(batch_size=batch_size, hgs_file_path=hgs_file_path, epoch=epoch, episode=episode, cuda_device_num=cuda_device_num)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size, hgs_file_path, epoch, episode, cuda_device_num):

    # Prep
    ###############################################
        self.model.train()
        hgs_file_exists = os.path.exists(hgs_file_path)
        batch = self.env.load_problems(batch_size, self.env_params['sr_size'])  
        reset_state, _, _ = self.env.reset()
        proj_nodes = self.model.pre_forward(reset_state,return_h_mean=True)
 
        prob_list = torch.zeros(size=(batch_size*self.env_params['sr_size'], self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        i=0
        while not done:

            selected, prob = self.model(state=state)
            # if i==1:
            #     entropy = -prob * torch.log(prob)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

            i = i + 1
    #prob_list = prob_list.reshape(self.env_params['sr_size'],batch_size,self.env.pomo_size, -1).permute(1,0,2,3).reshape(batch_size,self.env.pomo_size*self.env_params['sr_size'],-1)
    #reward = reward.reshape(self.env_params['sr_size'],batch_size, self.env.pomo_size).permute(1,0,2).reshape(batch_size,self.env.pomo_size*self.env_params['sr_size'])
    #entropy = entropy.reshape(self.env_params['sr_size'],batch_size, self.env.pomo_size).permute(1,0,2).reshape(batch_size,self.env.pomo_size*self.env_params['sr_size'])


        # ours
        if self.env_params['sr_size']>1:

            # Rotational Invariant
            ###############################################
            proj_nodes = proj_nodes.reshape(self.env_params['sr_size'], batch_size, proj_nodes.shape[1],-1)
            cos = torch.nn.CosineSimilarity(dim=-1)
            similarity = 0
            for i in range(self.env_params['sr_size']-1):
                similarity = similarity + cos(proj_nodes[0],proj_nodes[i+1])

            similarity /= (self.env_params['sr_size']-1)
       
            # Problem Symmetricity
            ###############################################
            prob_list_sr \
                = prob_list.view(self.env_params['sr_size'], batch_size, self.env.pomo_size, -1).permute(1, 2, 0,3).reshape(batch_size,self.env_params['sr_size']*self.env.pomo_size,-1)
            reward_sr \
                = reward.view(self.env_params['sr_size'], batch_size, self.env.pomo_size).permute(1, 2, 0).reshape(batch_size,self.env_params['sr_size']*self.env.pomo_size)

            # shape: (batch,pomo,sr_size)
            advantage_sr = reward_sr - reward_sr.float().mean(dim=1,keepdims=True)

            # shape: (batch,pomo,sr_size)W
            log_prob_sr = prob_list_sr.log().sum(dim=2)
            loss_sr = -advantage_sr*log_prob_sr
            loss_sr_mean = loss_sr.mean()

            # Solution Symmetricity
            ###############################################
            prob_list_pomo \
                = prob_list.view(self.env_params['sr_size'], batch_size, self.env.pomo_size, -1).permute(1, 0, 2,3)
            reward_pomo \
                = reward.view(self.env_params['sr_size'], batch_size, self.env.pomo_size).permute(1, 0, 2)
            # shape: (batch,sr_size,pomo)
            advantage_pomo = reward_pomo - reward_pomo.float().mean(dim=2, keepdims=True)

            # shape: (batch,sr_size,pomo)
            log_prob_pomo = prob_list_pomo.log().sum(dim=3)
            loss_pomo = -advantage_pomo*log_prob_pomo
            loss_pomo_mean = loss_pomo.mean()
            # wandb.log({"similarity": similarity.mean()})
            # wandb.log({"reward": reward.mean()})    
            # Sum of two symmetric loss

            # HGS loss
            advantage_hgs = reward - reward.float().mean(dim=1, keepdims=True)
            # shape: (batch, pomo)
            log_prob_hgs = prob_list.log().sum(dim=2)
            # size = (batch, pomo)
            if episode > 0 and (episode == 24):
                # if hgs_file_exists:
                #     hgs_costs = load_hgs_costs(hgs_file_path, epoch)
                hgs_costs = hgs_solution(batch, device = torch.device(f"cuda:{cuda_device_num}")) 
                save_hgs_costs(hgs_file_path, epoch, episode, hgs_costs)
                weight = 1.0
                cost_difference = hgs_costs - reward
                cost_difference = weight / (cost_difference + 1e-6)
                mask = (advantage_hgs > 0) & (cost_difference > 0)
                loss_hgs = - (advantage_hgs + cost_difference) * log_prob_hgs * mask + (-advantage_hgs * log_prob_hgs) * ~mask
            else:
                loss_hgs = - advantage_hgs * log_prob_hgs
            loss_hgs_mean = loss_hgs.mean()

            loss_mean = loss_sr_mean + loss_pomo_mean + loss_hgs_mean


            reward \
                = reward.reshape(self.env_params['sr_size'],batch_size, self.env.pomo_size).permute(1,0,2).reshape(batch_size,self.env.pomo_size*self.env_params['sr_size'])

        else:

            proj_nodes = proj_nodes.reshape(self.env_params['sr_size'], batch_size, proj_nodes.shape[1],-1)
            cos = torch.nn.CosineSimilarity(dim=-1)


            similarity = cos(proj_nodes[0],proj_nodes[0])
            # wandb.log({"similarity": similarity.mean()})
            # wandb.log({"reward": reward.mean()}) 
            # Loss
            ###############################################
            advantage = reward - reward.float().mean(dim=1, keepdims=True)
            # shape: (batch, pomo)
            log_prob = prob_list.log().sum(dim=2)
            
            # HGS优化损失函数
            if episode > 0 and (episode == 24):
                # if hgs_file_exists:
                #     hgs_costs = load_hgs_costs(hgs_file_path, epoch, episode)
                hgs_costs = hgs_solution(batch, device = torch.device(f"cuda:{cuda_device_num}")) 
                save_hgs_costs(hgs_file_path, epoch, episode, hgs_costs)
                weight = 1.0
                print(hgs_costs.shape)
                cost_difference = hgs_costs - reward
                cost_difference = weight / (cost_difference + 1e-6)
                mask = (advantage > 0) & (cost_difference > 0)
                loss = - (advantage + cost_difference) * log_prob * mask + (-advantage * log_prob) * ~mask
            else:
                loss = - advantage * log_prob

            # shape: (batch, pomo)
            loss_mean = loss.mean()


        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()

    def _test(self):

        no_pomo_score_AM = AverageMeter()
        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()


        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            no_pomo_score,score, aug_score = self._test_one_batch(batch_size)

            no_pomo_score_AM.update(no_pomo_score, batch_size)
            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size
        return no_pomo_score_AM.avg, score_AM.avg, aug_score_AM.avg


    def _test_one_batch(self, batch_size):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

  

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        no_pomo_score = -aug_reward[0, :, 0].mean()

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_pomo_score.item(), no_aug_score.item(), aug_score.item()
