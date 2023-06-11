import logging
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options

import numpy as np
import time

def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    '''
    num_list = np.arange(520000, 1400000, 20000)
    fname = opt['name']
    fpretrain = opt['path']['pretrain_network_g']
    resultdir = opt['path']['results_root']
    for num in num_list:
        opt['name'] = fname + str(num) + 'itr'
        opt['path']['pretrain_network_g'] = fpretrain + str(num) + '.pth'
        opt['path']['results_root'] = resultdir + str(num) + 'itr'
        opt['path']['log'] = resultdir + str(num) + 'itr'
        opt['path']['visualization'] = resultdir + str(num) + 'itr\\visualization'

        print(opt['name'])
        print(opt['path']['pretrain_network_g'])
    '''
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    # [YNU] added calculating elapsed time
    torch.cuda.synchronize()
    start = time.time()

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])

    torch.cuda.synchronize()
    elapsed_time = time.time() - start

    return elapsed_time

if __name__ == '__main__':

    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    elapsed_time = test_pipeline(root_path)

    print(f'Elapsed time:{elapsed_time} sec.')