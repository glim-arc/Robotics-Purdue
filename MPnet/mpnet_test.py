from __future__ import print_function
from Model.end2end_model import End2EndMPNet, End2EndMPNet3D
import Model.model as model
import Model.model_nd as modelnd
import Model.AE.CAE as CAE_2d
import Model.AE.CAE3D as CAE_3d

import numpy as np
import argparse
import os
import torch
from plan_general import *
import plan_s2d  # planning function specific to s2d environment (e.g.: collision checker, normalization)
import plan_s3d
import data_loader_2d
import data_loader_r3d
from torch.autograd import Variable
import copy
import os
import random
from utility import *
import utility_s2d
import utility_s3d
import progressbar
import logging

def main(args):
    # set seed
    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)
    py_seed = np.random.randint(low=0, high=1000)
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)
    # Build the models
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    # setup evaluation function and load function
    if args.env_type == 's2d':
        total_input_size = 2800+4
        AE_input_size = 2800
        mlp_input_size = 28+4
        output_size = 2
        IsInCollision = plan_s2d.IsInCollision
        load_test_dataset = data_loader_2d.load_test_dataset
        normalize = utility_s2d.normalize
        unnormalize = utility_s2d.unnormalize
        CAE = CAE_2d
        MLP = model.MLP

        if args.drop == False:
            MLP = modelnd.MLP

        mpNet = End2EndMPNet(total_input_size, AE_input_size, mlp_input_size, \
                    output_size, CAE, MLP)

    elif args.env_type == 's3d':
        total_input_size = 6000+6
        AE_input_size = 6000
        mlp_input_size = 30+6
        output_size = 3
        load_test_dataset = data_loader_r3d.load_test_dataset
        IsInCollision = plan_s3d.IsInCollision
        normalize = utility_s3d.normalize
        unnormalize = utility_s3d.unnormalize
        CAE = CAE_3d
        MLP = model.MLP

        if args.drop == False:
            MLP = modelnd.MLP

        mpNet = End2EndMPNet3D(total_input_size, AE_input_size, mlp_input_size, \
                            output_size, CAE, MLP)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    # load previously trained model if start epoch > 0
    model_path='mpnet_epoch_%d.pkl' %(args.epoch)
    if args.epoch > 0:
        load_net_state(mpNet, os.path.join(args.model_path, model_path))
        if args.reproducible:
            # set seed from model file
            torch_seed, np_seed, py_seed = load_seed(os.path.join(args.model_path, model_path))
            torch.manual_seed(torch_seed)
            np.random.seed(np_seed)
            random.seed(py_seed)
    if torch.cuda.is_available():
        mpNet.cuda()
        mpNet.mlp.cuda()
        mpNet.encoder.cuda()
    if args.epoch > 0:
        load_opt_state(mpNet, os.path.join(args.model_path, model_path))


    # load test data
    print('loading...')
    test_data = load_test_dataset(N=args.N, NP=args.NP, s=args.s, sp=args.sp, folder=args.data_path)
    obc, obs, paths, path_lengths = test_data
    print("paths: ", len(paths))
    print(path_lengths) 

    normalize_func=lambda x: normalize(x, args.world_size)
    unnormalize_func=lambda x: unnormalize(x, args.world_size)

    # test on dataset
    test_suc_rate = 0.
    DEFAULT_STEP = 0.01
    # for statistics

    n_valid_total = 0
    n_successful_total = 0
    sum_time = 0.0
    sum_timesq = 0.0
    min_time = float('inf')
    max_time = -float('inf')

    for i in range(len(paths)):
        logging.info(f'planning start: env={args.s + i}')

        n_valid_cur = 0
        n_successful_cur = 0

        widgets = [
            f'planning: env={args.s + i}, path=',
            progressbar.Variable('path_number', format='{formatted_value}', width=1), ' ',
            progressbar.Bar(),
            ' (', progressbar.Percentage(), ' complete)',
            ' success rate = ', progressbar.Variable('success_rate', format='{formatted_value}', width=4, precision=3),
            ' planning time = ', progressbar.Variable('planning_time', format='{formatted_value}sec', width=4, precision=3),
        ]
        bar = progressbar.ProgressBar(widgets = widgets)

        # save paths to different files, indicated by i
        # feasible paths for each env
        for j in bar(range(len(paths[0]))):
            time0 = time.time()
            if path_lengths[i][j]<2:
                # the data might have paths of length smaller than 2, which are invalid
                continue

            found_path = False
            n_valid_cur += 1
            path = [torch.from_numpy(paths[i][j][0]).type(torch.FloatTensor),\
                    torch.from_numpy(paths[i][j][path_lengths[i][j]-1]).type(torch.FloatTensor)]
            step_sz = DEFAULT_STEP
            MAX_NEURAL_REPLAN = 11
            for t in range(MAX_NEURAL_REPLAN):
                path = neural_plan(mpNet, path, obc[i], obs[i], IsInCollision, \
                                    normalize_func, unnormalize_func, t==0, step_sz=step_sz)
                
                if args.lvc == True:
                    path = lvc(path, obc[i], IsInCollision, step_sz=step_sz)
                
                if feasibility_check(path, obc[i], IsInCollision, step_sz=step_sz):
                    found_path = True
                    n_successful_cur += 1
                    break

            time1 = time.time() - time0
            sum_time += time1
            sum_timesq += time1 * time1
            min_time = min(min_time, time1)
            max_time = max(max_time, time1)

            # write the path
            if type(path[0]) is not np.ndarray:
                # it is torch tensor, convert to numpy
                path = [p.numpy() for p in path]
            path = np.array(path)
            path_file = args.result_path+'env_%d/' % (i+args.s)
            if not os.path.exists(path_file):
                # create directory if not exist
                os.makedirs(path_file)

            if found_path:
                filename = f'path_{j+args.sp}.txt'
            else:
                filename = f'path_{j+args.sp}-fail.txt'
            np.savetxt(path_file + filename, path, fmt='%f')

            success_rate = n_successful_cur / n_valid_cur if n_valid_cur > 0 else float('nan')

            if found_path:
                bar.update(path_number=j+args.sp, success_rate=success_rate, planning_time=time1)

        n_valid_total += n_valid_cur
        n_successful_total += n_successful_cur
        if n_valid_total == 0:
            success_rate = avg_time = stdev_time = float('nan')
        else:
            success_rate = n_successful_total / n_valid_total if n_valid_total > 0 else float('nan')
            avg_time = sum_time / n_valid_total
            stdev_time = np.sqrt((sum_timesq - sum_time * avg_time) / (n_valid_total - 1)) if n_valid_total > 1 else 0
        print(f'cumulative: success rate={success_rate:.2f}, runtime (min/avg/max/stdev) = {min_time:.2f}/{avg_time:.2f}/{max_time:.2f}/{stdev_time:.2f}s')
        logging.info(f'cumulative: success rate={success_rate:.2f}, runtime (min/avg/max/stdev) = {min_time:.2f}/{avg_time:.2f}/{max_time:.2f}/{stdev_time:.2f}s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lvc', type=bool, default=True, help='lvc on off')
    parser.add_argument('--drop', type=bool, default=True, help='drop out on off')

    parser.add_argument('--model-path', type=str, default='./models/',help='folder of trained model')
    parser.add_argument('--N', type=int, default=10, help='number of environments')
    parser.add_argument('--NP', type=int, default=100, help='number of paths per environment')
    parser.add_argument('--s', type=int, default=0, help='start of environment index')
    parser.add_argument('--sp', type=int, default=2001, help='start of path index')

    # Model parameterss
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--data-path', type=str, default='./data/', help='path to dataset')
    parser.add_argument('--result-path', type=str, default='./results/', help='folder to save paths computed')
    parser.add_argument('--epoch', type=int, default=500, help='epoch of trained model to use')
    parser.add_argument('--env-type', type=str, default='s3d', help='s2d for simple 2d')
    parser.add_argument('--world-size', nargs='+', type=float, default=20., help='boundary of world')
    parser.add_argument('--reproducible', default=False, action='store_true', help='use seed bundled with trained model')

    args = parser.parse_args()
    # Initialize Logging
    logging.basicConfig(filename="log.txt", level=logging.INFO)
    logging.info(args)


    print(args)
    main(args)
