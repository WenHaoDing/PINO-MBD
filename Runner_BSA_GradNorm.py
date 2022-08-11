from argparse import ArgumentParser
import yaml
import torch
from models.fourier1d import FNN1d_BSA_GradNorm
from train_utils import Adam
from train_utils.datasets import BSA_Loader_WithVirtualData, FES_Loader
from train_utils.train_2d import train_BSA_WithGradNorm
from train_utils.eval_2d import eval_burgers
from train_utils.solution_extension import FDD_Extension
import matplotlib.pyplot as plt
import os
from train_utils.losses_BSA import BSA_PINO_loss
import h5py
from train_utils.losses import LpLoss
from Defination_Experiments import Experiments_GradNorm_BSA, Experiments_Virtual_BSA
from scipy.io import savemat
# import spicy.io as io
import numpy as np

f = open(r'configs/BSA/BSA_PINO-MBD.yaml')
BSA_config = yaml.load(f)


def run(config, args=False):
    data_config = config['data']
    ComDevice = torch.device('cuda:0')
    dataset = BSA_Loader_WithVirtualData(data_config['datapath'], data_config['weights_datapath'],
                                         data_config['test_datapath'], data_config['weights_datapath_test'],
                                         data_config['virtual_datapath'], data_config['weights_datapath_virtual'],
                                         data_config['Structure_datapath'],
                                         nt=data_config['nt'], nSlice=data_config['nSlice'],
                                         sub_t=data_config['sub_t'],
                                         new=False, inputDim=data_config['inputDim'],
                                         outputDim=data_config['outputDim'],
                                         ComDevice=ComDevice)

    # Manual:Change new to False(from new)
    train_loader, test_loader, virtual_loader, PDE_weights_virtual, ToOneV, W2_CX, W2_CY, W2_CZ, Eigens2, TrackDOFs, Nloc = dataset.make_loader(
        n_sample=data_config['n_sample'], n_sample_virtual=data_config['n_sample_virtual'],
        batch_size=config['train']['batchsize'],
        batch_size_virtual=config['train']['batchsize_virtual'],
        start=data_config['offset'])
    if data_config['OperatorType'] == 'PINO-MBD' or data_config['OperatorType'] == 'PINO':
        if data_config['NoData'] == 'On':
            task_number = 1
        else:
            task_number = 2
            if data_config['DiffLossSwitch'] == 'On':
                task_number += 1
            if data_config['VirtualSwitch'] == 'On':
                task_number += 1
    else:
        task_number = 1
    print('This mission will have {} task(s)'.format(task_number))
    if data_config['GradNorm'] == 'On' and task_number != 1:
        print('GradNorm will be launched with alpha={}.'.format(data_config['GradNorm_alpha']))
    else:
        print('GradNorm will not be launched for this mission.')
    model = FNN1d_BSA_GradNorm(modes=config['model']['modes'],
                               width=config['model']['width'], fc_dim=config['model']['fc_dim'],
                               inputDim=data_config['inputDim'],
                               outputDim=data_config['outputDim'],
                               task_number=task_number).to(ComDevice)

    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=config['train']['base_lr'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=config['train']['base_lr'], momentum=0.95, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                  milestones=config['train']['milestones'],
    #                                                  gamma=config['train']['scheduler_gamma'])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=0.1 * config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    train_BSA_WithGradNorm(model,
                           train_loader, test_loader, virtual_loader, PDE_weights_virtual,
                           optimizer, scheduler,
                           config,
                           ToOneV,
                           W2_CX, W2_CY, W2_CZ,
                           Eigens2, TrackDOFs, Nloc,
                           inputDim=data_config['inputDim'], outputDim=data_config['outputDim'], D=data_config['D'],
                           ComDevice=ComDevice,
                           rank=0, log=False,
                           project='PINO-BSA',
                           group='default',
                           tags=['default'],
                           use_tqdm=True
                           )

    # for x, y in train_loader:
    #     x, y = x.cuda(), y.cuda()
    #     batch_size = config['train']['batchsize']
    #     nt = data_config['nt']
    #     out = model(x)
    #     print('Shape of x={}; Shape of y={}; Shape of out={}'.format(x.shape, y.shape, out.shape))
    #     device2 = torch.device('cpu')
    #     plt.figure(1)
    #     plt.plot(y.to(device2).permute([0, 2, 1])[0, 0, :].detach().numpy(), color='red')
    #     plt.plot(out.to(device2).permute([0, 2, 1])[0, 0, :].detach().numpy(), linestyle='--', color='black')
    #     plt.figure(2)
    #     plt.plot(y.to(device2).permute([0, 2, 1])[0, 0+20, :].detach().numpy(), color='red')
    #     plt.plot(out.to(device2).permute([0, 2, 1])[0, 0+20, :].detach().numpy(), linestyle='--', color='black')
    #     plt.figure(3)
    #     plt.plot(y.to(device2).permute([0, 2, 1])[0, 0+21, :].detach().numpy(), color='red')
    #     plt.plot(out.to(device2).permute([0, 2, 1])[0, 0+21, :].detach().numpy(), linestyle='--', color='black')
    #     plt.figure(4)
    #     plt.plot(y.to(device2).permute([0, 2, 1])[0, 0+22, :].detach().numpy(), color='red')
    #     plt.plot(out.to(device2).permute([0, 2, 1])[0, 0+22, :].detach().numpy(), linestyle='--', color='black')
    #     plt.figure(5)
    #     plt.plot(y.to(device2).permute([0, 2, 1])[0, 0+23, :].detach().numpy(), color='red')
    #     plt.plot(out.to(device2).permute([0, 2, 1])[0, 0+23, :].detach().numpy(), linestyle='--', color='black')
    #     plt.figure(6)
    #     plt.plot(y.to(device2).permute([0, 2, 1])[0, 0+26, :].detach().numpy(), color='red')
    #     plt.plot(out.to(device2).permute([0, 2, 1])[0, 0+26, :].detach().numpy(), linestyle='--', color='black')
    #     plt.show()
    #
    #     # dy, ddy = FDD_Extension(y, dt=0.5)
    #     # dout, ddout = FDD_Extension(out, dt=0.5)
    #     # # plt.figure(1)
    #     # # plt.plot(dy.to(device2).permute([0, 2, 1])[0, 0, :].detach().numpy(), color='red')
    #     # # plt.plot(dout.to(device2).permute([0, 2, 1])[0, 0, :].detach().numpy(), linestyle='--', color='black')
    #     # # plt.show()
    #     #
    #     # # dy = np.mat(dy.to(device2).detach().numpy())
    #     # # ddy = np.mat(ddy.to(device2).detach().numpy())
    #     # # dout = np.mat(dout.to(device2).detach().numpy())
    #     # # ddout = np.mat(ddout.to(device2).detach().numpy())
    #     # # io.savemat('PythonData.mat', {'dy': dy, 'ddy': ddy, 'dout': dout, 'ddout': ddout})
    #     #
    #     np.savetxt('y.txt', y.to(device2).permute([0, 2, 1])[0, 0+20, :].detach().numpy())
    #     np.savetxt('out.txt', out.to(device2).permute([0, 2, 1])[0, 0+20, :].detach().numpy())
    return model


Style = 'Train'
Multiple = 'Yes'
Clip = 5
File = './configs/BSA/BSA_PINO-MBD.yaml'
if Style == 'Train':
    Experiments_GradNorm_BSA(Multiple, Clip, File, run)
    # Experiments_Virtual_BSA(Multiple, Clip, File, run)
elif Style == 'eval':
    device = torch.device('cpu')
    BSA_data_config = BSA_config['data']
    model = FNN1d_BSA_GradNorm(modes=BSA_config['model']['modes'],
                               width=BSA_config['model']['width'], fc_dim=BSA_config['model']['fc_dim'],
                               inputDim=BSA_data_config['inputDim'],
                               outputDim=BSA_data_config['outputDim'], task_number=4).to(device)
    ckpt_path = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/BSARunner/Experiment1/BSA_405.pt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    virtual_datapath = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/data/Project_BSA/PDEM/VirtualData_499.mat'
    input_eval = torch.tensor(h5py.File(virtual_datapath)['input'][:, 40:, :]).permute([2, 1, 0]).to(torch.float32)
    # eval_dataset = torch.utils.data.TensorDataset(input_eval)
    # eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1000, shuffle=False)
    index = 1
    SavePath = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/BSARunner/Experiment1/eval/'

    out = model(input_eval).detach().numpy()
    mdic = {"output": out}
    FileName = SavePath + 'eval_499.mat'
    savemat(FileName, mdic)
elif Style == 'eval_batch':
    device = torch.device('cpu')
    BSA_data_config = BSA_config['data']
    model = FNN1d_BSA_GradNorm(modes=BSA_config['model']['modes'],
                               width=BSA_config['model']['width'], fc_dim=BSA_config['model']['fc_dim'],
                               inputDim=BSA_data_config['inputDim'],
                               outputDim=BSA_data_config['outputDim'], task_number=4).to(device)
    ckpt_path = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/BSARunner/Experiment1/BSA_405.pt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    virtual_datapath = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/data/Project_BSA/PDEM/VirtualData_50000_Pack2.mat'
    input_eval = torch.tensor(h5py.File(virtual_datapath)['input'][:, 40:, :]).permute([2, 1, 0]).to(torch.float32)
    eval_dataset = torch.utils.data.TensorDataset(input_eval)
    batch_size = 5000
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    eval_iter = iter(eval_loader)
    index = 1
    SavePath = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/BSARunner/Experiment1/eval/EV1/'

    for i in range(0, int(input_eval.size(0)/batch_size)+1):
        print('Now operating batch No.{}'.format(i+1))
        x = next(eval_iter)[0].to(device)
        out = model(x).detach()

        Name = SavePath + 'eval' + str(input_eval.size(0)) + '_' + str(i+1) + '.pt'
        torch.save(out, Name)

