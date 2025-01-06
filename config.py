import  argparse

"""
    description and super parameter
"""

args = argparse.ArgumentParser()
args.add_argument('--Dataset', default='ADNI') # Preprocessed Data from ADNI
args.add_argument('--Task', default='ADNC') # ADNC, pMCIsMCI
args.add_argument('--Filters', default= [16, 32, 64, 96, 80, 64])
args.add_argument('--Batch_size', default= 6)
args.add_argument('--Latent', default= 64)
args.add_argument('--dropout', default= 0.3)
args.add_argument('--learning_rate', default= 0.0001)
args.add_argument('--Epoches', default= 100)
args.add_argument('--task_num_classes', default= 2)
args.add_argument('--Trainset', default= 'ADNI1') # We only trained the model on ADNI1, tested on ADNI2 and NACC
args.add_argument('--meta', default= 'All') # Age, Gender, Education, All
args.add_argument('--id', default= 2)
args.add_argument('--mi_coef', default= 1)
args.add_argument('--mi_iter', default= 5)
args.add_argument('--para', default= 0.5)
args = args.parse_args()
