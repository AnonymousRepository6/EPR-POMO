##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 2

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "../..")  # for problem_def
sys.path.insert(0, "../../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSPTrainer import TSPTrainer as Trainer


##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
    'sr_size':2,
}

env_test_params = {
    'problem_size': 100,
    'pomo_size': 100,
}


model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [2001,],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 2000,
    'train_episodes': 10 * 1000,
    'train_batch_size': 32,
    'lkh_file_path':'lkh_costs/lkh_costs.pt',
    'is_pomo':False,
    'alpha':0.1,
    'wandb':True,
    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 10,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_20.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': True,  # enable loading pre-trained model
        'path': './result/20250609_122605_train__tsp_n20',  # directory path of pre-trained model and log files saved.
        'epoch': 1150,  # epoch version of pre-trained model to laod.

    }
}



tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'test_episodes': 10*1000,
    'test_batch_size': 32,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 32
}




logger_params = {
    'log_file': {
        'desc': 'train__tsp_n20',
        'filename': 'run_log'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params,tester_params=tester_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    global tester_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4
    tester_params['test_episodes'] = 10
    tester_params['aug_factor'] = 2
    tester_params['test_batch_size'] = 4
    tester_params['aug_batch_size'] = 4

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
