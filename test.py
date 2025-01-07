
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import scipy.io as sio

from config import args
from model import Classifier, Disentanglement, Encoder
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.id)
assert torch.cuda.is_available() # "Currently, we only support CUDA version"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

###########################################################################

args.Batch_size = 12
Base_Dir = os.path.dirname(os.path.abspath(__file__))
subdir = args.Trainset + '-' + args.meta + '-' + str(args.para) + '/'

Dir = Base_Dir + args.Task + '/' + subdir + '/'
Pth_Dir = Dir + '/Pth/'
History_Dir = Dir + '/History/'
Results_Dir = Dir + '/Results/'

###########################################################################
### Data Preparing
data = sio.loadmat(Dir + '/Dataset.mat')
Train_list = list(data['Train_list'].squeeze())
Valid_list = list(data['Valid_list'].squeeze())
Test1_list = list(data['Test1_list'].squeeze())
Test2_list = list(data['Test2_list'].squeeze())

###########################################################################

"""
DataLoader Preparing
"""
# print(Train_list)

train_set = LONI_Loader(Train_list, root = Base_Dir + 'Data/ADNI/', mode = 'test', task = args.Task)
train_loader = DataLoader(dataset=train_set, batch_size=args.Batch_size, shuffle=True)
valid_set = LONI_Loader(Valid_list, root = Base_Dir + 'Data/ADNI/', mode = 'test', task = args.Task)
valid_loader = DataLoader(dataset=valid_set, batch_size=args.Batch_size, shuffle=False)
test1_set = LONI_Loader(Test1_list, root = Base_Dir + 'Data/ADNI/', mode = 'test', task = args.Task)
test1_loader = DataLoader(dataset=test1_set, batch_size=args.Batch_size, shuffle=False)
test2_set = LONI_Loader(Test2_list, root = Base_Dir + 'Data/ADNI/', mode = 'test', task = args.Task)
test2_loader = DataLoader(dataset=test2_set, batch_size=args.Batch_size, shuffle=False)

model_E = Encoder().cuda()
model_D = Disentanglement().cuda()
model_C = Classifier().cuda()

def test_step(dataset):
    model_E.eval()
    model_D.eval()
    model_C.eval()
    score_all = []
    label_all = []
    RID_all = []
    feat_c = []
    feat_m = []
    meta = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataset):
            images, Age, Gender, Edu, label, RID = data
            #print(Age.shape, Gender.shape, Edu.shape)
            Age = Age.cpu().numpy()
            Gender = Gender.cpu().numpy()
            Edu = Edu.cpu().numpy()
            label = label.squeeze(-1).long().cpu().numpy()
            RID = RID.squeeze(-1).long().cpu().numpy()
            batch_score = []
            batch_feat_c = []
            batch_feat_m = []
            for image in images:
                feat = model_E(image.cuda())
                dis0, dis1  = model_D(feat)
                out = model_C(dis0)
                out = out.softmax(-1).unsqueeze(1).detach().cpu().numpy()
                batch_score.append(out)
                batch_feat_c.append(dis0.unsqueeze(1).detach().cpu().numpy())
                batch_feat_m.append(dis1.unsqueeze(1).detach().cpu().numpy())

            score_all.append(np.concatenate(batch_score, axis=1))
            feat_c.append(np.concatenate(batch_feat_c, axis = 1))
            feat_m.append(np.concatenate(batch_feat_m, axis = 1))
            meta.append(np.concatenate([Age, Gender, Edu], axis = 1))
            label_all.append(label)
            RID_all.append(RID)
            
    return np.concatenate(feat_c, axis=0), np.concatenate(feat_m, axis=0), np.concatenate(meta, axis=0), np.concatenate(score_all, axis=0), np.concatenate(label_all, axis=0), np.concatenate(RID_all, axis=0)

model_E.load_state_dict(torch.load(Pth_Dir + 'E_Checkpoint_' + args.Task  + '.pth'))
model_D.load_state_dict(torch.load(Pth_Dir + 'D_Checkpoint_' + args.Task  + '.pth'))
model_C.load_state_dict(torch.load(Pth_Dir + 'C_Checkpoint_' + args.Task  + '.pth'))

train_feat_c, train_feat_m, train_meta, train_score, train_label, train_rid = test_step(train_loader)
valid_feat_c, valid_feat_m, valid_meta, valid_score, valid_label, valid_rid = test_step(valid_loader)
test1_feat_c, test1_feat_m, test1_meta, test1_score, test1_label, test1_rid = test_step(test1_loader)
test2_feat_c, test2_feat_m, test2_meta, test2_score, test2_label, test2_rid = test_step(test2_loader)
sio.savemat(Results_Dir + '/' + args.Task + '_Prediction_Checkpoint.mat', {
    'train_feat_c':train_feat_c, 'train_feat_m':train_feat_m, 'train_meta':train_meta, 'train_score':train_score, 'train_label':train_label, 'train_rid':train_rid,
    'valid_feat_c':valid_feat_c, 'valid_feat_m':valid_feat_m, 'valid_meta':valid_meta, 'valid_score':valid_score, 'valid_label':valid_label, 'valid_rid':valid_rid,
    'test1_feat_c':test1_feat_c, 'test1_feat_m':test1_feat_m, 'test1_meta':test1_meta, 'test1_score':test1_score, 'test1_label':test1_label, 'test1_rid':test1_rid,
    'test2_feat_c':test2_feat_c, 'test2_feat_m':test2_feat_m, 'test2_meta':test1_meta, 'test2_score':test2_score, 'test2_label':test2_label, 'test2_rid':test2_rid    
})
            
