
import os
import random

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

from config import args
from utils import *
from Solver import Solver


os.environ['CUDA_VISIBLE_DEVICES'] = str(args.id)
assert torch.cuda.is_available() # "Currently, we only support CUDA version"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

subdir = args.Trainset + '-' + args.meta + '-' + str(args.para) + '/'

###########################################################################
### File Folder Preparing

Base_Dir = os.path.dirname(os.path.abspath(__file__))
Dir = Base_Dir + args.Task + '/' + subdir + '/'
Pth_Dir = Dir + '/Pth/'
History_Dir = Dir + '/History/'
Results_Dir = Dir + '/Results/'

if not os.path.exists(Pth_Dir):
    os.makedirs(Pth_Dir)
if not os.path.exists(History_Dir):
    os.makedirs(History_Dir)
if not os.path.exists(Results_Dir):
    os.makedirs(Results_Dir)

###########################################################################
### Data Preparing

sample_list = sio.loadmat(Base_Dir + '/Sample_List.mat') # we pre-stratified the ADNI-1 dataset into training-set and validation-set
ADNI = pd.read_excel(Base_Dir + '/ADNI.xlsx')
ADNI = ADNI[ADNI.DX != -10]
ADNI2 = ADNI[ADNI.Phase == 'ADNI 2']
ADNIGO = ADNI[ADNI.Phase == 'ADNI GO']
ADNI3 = ADNI[ADNI.Phase == 'ADNI 3']

if args.Task == 'ADNC':
    Train_list = list(np.squeeze(sample_list['Train']['AD'][0, 0])) + list(np.squeeze(sample_list['Train']['NC'][0, 0]))
    Valid_list = list(np.squeeze(sample_list['Val']['AD'][0, 0])) + list(np.squeeze(sample_list['Val']['NC'][0, 0]))
    Test1_list = list(ADNI2[ADNI2.DX == 3].RID) + list(ADNI2[ADNI2.DX == 1].RID)
    Test2_list = list(ADNI3[ADNI3.DX == 3].RID) + list(ADNI3[ADNI3.DX == 1].RID)
elif args.Task == 'MCIc':
    Train_list = list(np.squeeze(sample_list['Train']['pMCI'][0, 0])) + list(np.squeeze(sample_list['Train']['sMCI'][0, 0]))
    Valid_list = list(np.squeeze(sample_list['Val']['pMCI'][0, 0])) + list(np.squeeze(sample_list['Val']['sMCI'][0, 0]))
    Test1_list = list(ADNI2[ADNI2.DX == 4].RID) + list(ADNI2[ADNI2.DX == 2].RID) + list(ADNIGO[ADNIGO.DX == 4].RID) + list(ADNIGO[ADNIGO.DX == 2].RID)
    Test2_list = list(ADNI3[ADNI3.DX == 4].RID) + list(ADNI3[ADNI3.DX == 2].RID)

sio.savemat(Dir + '/Dataset.mat',{
    'Train_list': Train_list,
    'Valid_list': Valid_list,
    'Test1_list': Test1_list,
    'Test2_list': Test2_list
})

###########################################################################
### Model Training
"""
DataLoader Preparing
"""
# print(Train_list)
random.shuffle(Train_list)
train_set = LONI_Loader(Train_list, root = Base_Dir + 'Data/ADNI/', mode = 'train', task = args.Task)
train_loader = DataLoader(dataset=train_set, batch_size=args.Batch_size, shuffle=True)
valid_set = LONI_Loader(Valid_list, root = Base_Dir + 'Data/ADNI/', mode = 'valid', task = args.Task)
valid_loader = DataLoader(dataset=valid_set, batch_size=args.Batch_size, shuffle=False)
# test1_set = LONI_Loader(Test1_list, root = Base_Dir + 'Data/ADNI/', mode = 'valid', task = args.Task)
# test1_loader = DataLoader(dataset=test1_set, batch_size=args.Batch_size, shuffle=False)
# test2_set = LONI_Loader(Test2_list, root = Base_Dir + 'Data/ADNI/', mode = 'valid', task = args.Task)
# test2_loader = DataLoader(dataset=test2_set, batch_size=args.Batch_size, shuffle=False)

solver = Solver(args, dir = Dir)
min_val_loss = np.inf
args.Epoches = int(args.Epoches)
for epoch in range(args.Epoches):
    
    solver.train_epoch(train_loader, epoch, prefix = args.Task + '_train')
    cls_loss, acc = solver.test_epoch(valid_loader, epoch, prefix = args.Task + '_Val')
    # solver.test_epoch(test1_loader, epoch, prefix = args.Task + '_Test1')
    # solver.test_epoch(test2_loader, epoch, prefix = args.Task + '_Test2')
    
    if cls_loss<min_val_loss:
        torch.save(solver.E.state_dict(), Pth_Dir + 'E_Checkpoint_' + args.Task + '.pth')
        torch.save(solver.D.state_dict(), Pth_Dir + 'D_Checkpoint_' + args.Task + '.pth')
        torch.save(solver.C.state_dict(), Pth_Dir + 'C_Checkpoint_' + args.Task + '.pth')
        torch.save(solver.R.state_dict(), Pth_Dir + 'R_Checkpoint_' + args.Task + '.pth')
        torch.save(solver.Gen.state_dict(), Pth_Dir + 'Gen_Checkpoint_' + args.Task + '.pth')
        torch.save(solver.Dis.state_dict(), Pth_Dir + 'Dis_Checkpoint_' + args.Task + '.pth')
        torch.save(solver.MI.state_dict(), Pth_Dir + 'MI_Checkpoint_' + args.Task + '.pth')
        torch.save(solver.Proj0.state_dict(), Pth_Dir + 'Proj0_Checkpoint_' + args.Task + '.pth')
        torch.save(solver.Proj1.state_dict(), Pth_Dir + 'Proj1_Checkpoint_' + args.Task + '.pth')
        min_val_loss = cls_loss
