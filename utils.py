import logging
import os
import pickle
import random

import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset


Base_Dir = os.path.dirname(os.path.abspath(__file__))

# AD:   3
# pMCI: 4
# sMCI: 2
# NC:   1
Binary_list = {
    'ADNC':{'3': 1, '1': 0}, 
    'MCIc':{'4': 1, '2': 0}
    }

Categorical_list = {
    '3':    {'3': 2, '4': 1, '2': 1, '1': 0},
}

Gender_list = {'F': 0, 'M': 1, '1': 1, '2': 0}

def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class LONI_Loader(Dataset):
    def __init__(self, RID_list, root, mode='train', task_num_classes = 2, task = 'ADNC'):
        self.lines = []
        paths, names = [], []
        for id in RID_list:
            names.append(id)
            path = os.path.join(root, str(id))
            paths.append(path)

        self.mode = mode
        self.names = names
        self.paths = paths
        self.task_num_classes = task_num_classes
        self.task = task

    def __getitem__(self, item):
        path = self.paths[item]
        tid = list(range(12))
        index0 = np.random.randint(0,12)
        tid.pop(index0)
        index1 = np.random.randint(0,11)
        index1 = tid[index1]
        image0, Age, Gender, Edu, label, RID = pkload(path + '/Flirt2Template' + str(index0) + '.pkl')
        # print(RID)
        if self.task_num_classes == 2:
            label = Binary_list[self.task][str(label)]
        else :
            label = Categorical_list[str(self.task_num_classes)][str(label)]
        # print(label)
        label = torch.from_numpy(np.ascontiguousarray(label))
        Age = torch.from_numpy(np.ascontiguousarray((Age - 55.1)/(91.5 - 55.1))).float()
        Gender = torch.from_numpy(np.ascontiguousarray(Gender)).long()
        Edu = torch.from_numpy(np.ascontiguousarray((Edu - 4)/(20 - 4))).float()
        RID = torch.from_numpy(np.ascontiguousarray(RID)).float()
        
        if self.mode == 'train':

            image1 = pkload(path + '/Flirt2Template' + str(index1) + '.pkl')[0]
            if random.random() < 0.5:
                image0 = np.flip(image0, 1)
            if random.random() < 0.5:
                image1 = np.flip(image1, 1)
    
            image0 = torch.from_numpy(np.ascontiguousarray(image0)).float()
            image1 = torch.from_numpy(np.ascontiguousarray(image1)).float()
            return image0, image1, Age, Gender, Edu, label, RID
        
        elif self.mode == 'valid':
            images = []
            image0 = pkload(path + '/Flirt2Template0.pkl')[0]
            images.append(torch.from_numpy(np.ascontiguousarray(image0)).float())
            return images, Age, Gender, Edu, label, RID
        
        else:
            images = []
            for i in range(12):
                img = pkload(path + '/Flirt2Template' + str(i) + '.pkl')[0]
                images.append(torch.from_numpy(np.ascontiguousarray(img)).float())
            return images, Age, Gender, Edu, label, RID

    def __len__(self):  
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

def ToPickle(excel_data, data = 'ADNI'):

    for s in range(excel_data.shape[0]):
        
        pkl_path = Base_Dir + '/Data/'
        nii_dir = Base_Dir + data + '-Flirt/'
        
        if data == 'NACC':
            Label   = excel_data.NACCUDSD[s]
            Age     = excel_data.Age[s]
            Gender  = Gender_list[str(excel_data.Sex[s])]
            Edu     = excel_data.Edu[s]
               
        else:
            Label   = excel_data.DX[s]
            Age     = excel_data.Age[s]
            Gender  = Gender_list[excel_data.Sex[s]]
            Edu     = excel_data.Education[s]
            
        nii_name = ['norm_brain_flirt.nii.gz'] +  ['norm_brain_flirt2T' + str(i) + '.nii.gz' for i in range(11)]
        
        for t in range(len(nii_name)):
                
            if data != 'NACC':
                filename = nii_dir + str(excel_data.ImageID[s]) + '/' + nii_name[t]
                pkl_out_dir = pkl_path + data + '/' + str(excel_data.RID[s]) + '/'
                
            else:
                filename = nii_dir + str(excel_data.NACCID[s]) + '/' + nii_name[t]
                pkl_out_dir = pkl_path + data + '/' + str(excel_data.NACCID[s]) + '/'
                
            images = nib.load(filename).get_fdata()
            images = np.expand_dims(images, 0) # [1, 154, 190, 148]
            images = images/255.0 # another choice is Min-Max normalization, all are okay in this work
            
            print(filename, Age, Gender, Edu, Label)
            if not os.path.exists(pkl_out_dir):
                os.makedirs(pkl_out_dir)
            pkl_out = pkl_out_dir + 'Flirt2Template' + str(t) + '.pkl'
            with open(pkl_out, 'wb') as f:
                if data == 'NACC':
                    pickle.dump((images, Age, Gender, Edu, Label, int(excel_data.NACCID[s][4:])), f)
                else:
                    pickle.dump((images, Age, Gender, Edu, Label, excel_data.RID[s]), f)
                    

if __name__ == '__main__':

    ADNI = pd.read_excel(Base_Dir + '/ADNI.xlsx')
    ToPickle(ADNI)

