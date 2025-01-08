## Multi-Template Meta-Information Regularized Network for Alzheimer’s Disease Diagnosis Using Structural MRI<br><sub>Official PyTorch Implementation</sub> 
### [Paper](https://ieeexplore.ieee.org/document/10365189) 
![Framework](/images/framework.png)

## Template Seclection
First, all T1 images are aligned to [Colin27 Template](https://www.mcgill.ca/bic/software/tools-data-analysis/anatomical-mri/atlases/colin-27) using affine registration ([FLIRT](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/))
```bash
flirt -in invol -ref refvol -out outvol -omat invol2refvol.mat
```
Second, regarding the transformation matrix as features, affinity propogation is then applied for template selection.
```bash
from sklearn.cluster import AffinityPropagation
import scipy.io as sio

transformation = sio.loadmat('./transformation.mat')['transformation']
ap = AffinityPropagation(preference=10, random_state=42) 
ap.fit(transformation)
cluster_centers_indices = ap.cluster_centers_indices_
```

## Model Training and Testing
All parameters are saved in [config.py](/config.py). 
To train and test the model with defaulted parameters, run:
```bash
python train.py
python test.py
```
To train the model with personalized parameters (such as classification weight $\alpha$ as 0.3 in the task of AD vs. NC Classification), run:
```bash
python train.py --para 0.3 --Task ADNC
python test.py --para 0.3 --Task ADNC
```

## Datasets
[ADNI](https://ida.loni.usc.edu)
[NACC](https://naccdata.org/)

## BibTeX
Please cite this paper if using the code.
```bibtex
@ARTICLE{10365189,
  author={Han, Kangfu and Li, Gang and Fang, Zhiwen and Yang, Feng},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Multi-Template Meta-Information Regularized Network for Alzheimer’s Disease Diagnosis Using Structural MRI}, 
  year={2024},
  volume={43},
  number={5},
  pages={1664-1676},
  keywords={Feature extraction;Metadata;Self-supervised learning;Mutual information;Alzheimer's disease;Aging;Minimization;Alzheimer’s disease;multi-template;meta-information;mutual information},
  doi={10.1109/TMI.2023.3344384}}
```
