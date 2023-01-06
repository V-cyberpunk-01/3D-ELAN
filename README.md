# Code for paper "3D-ELAN: 3D Equivalent Local Augmentation Deep Graph Neural Networks for Formation Energy Prediction on Doping Crystal Structure"

This is a PyTorch implementation of 3D-ELAN model and discussion experiments proposed by our paper "3D-ELAN: 3D Equivalent Local Augmentation Deep Graph Neural Networks for Formation Energy Prediction on Doping Crystal Structure"

## Data Preparation

```
│3D-ELAN
├──0916/  # Processed data for the main experiment
│  ├── composition
│  │   ├── composition_feature_a_pca.csv
│  │   ├── composition_feature_b_pca.csv
│  │   ├── ......
│  ├── NbSi_a_c0_f85_k15
│  │   ├── all_connect
│  │   │   ├── adj
│  │   │   ├── feature
│  ├── ......
├──0925/  # Processed data for the discussion experiment 1
│  ├── composition
│  ├── NbSi_a_f85_k53   # Data containing all atoms in a cell
│  ├── NbSi_b_f85_k53
│  ├── ......
├──1018/  # Processed data for the discussion experiment 2
│  ├── composition
│  ├── NbSi_a_c0_f50_k15   # Data containing all atoms in a cell
│  ├── NbSi_b_c21_f50_k17
│  ├── ......
```

```
│CSA
├──cif/  # material structure example for CSA 
```

## CSA Algorithm

You can get shell positions from a Crystal via this code.This code requires a target Crystal .cif file.And use command like `python.py csa.py -f ./cif/Nb5Si3.cif` to run it.And you will get the corresponding shell positions,as well as a picture to show.  

This is an example to show shell structure via CSA algorithm in the file named as image.png.

## 3D-ELAN Model Training Results

We uploaded the results and related metrics obtained during the experimental training of the model, and the suffixes in the saved pth file denote R2/MAE/RMSE in order.

```
│3D-ELAN
├──model_basic  # Results for the main experiment
├──model_(whole)cell  # Results for the discussion  experiment 1
├──model_feature50  # Results for the discussion  experiment 2
├──model_ablation  # Results for the discussion  experiment 3
```

- note: "rs" in the file name means "random seed","wo_i" means "without the part i"

## Requirements

update in 2023-01-01, please use the newest package 

## Model training and validation

We put the training section and validation section in the train.py. To ensure that our experimental results can be read in and reproduced, and run the model on different shell type , you have to change the path in train.py file from line 16 to line 24:

```
data_path = './1018/NbSi_c_c43_f50_k10/all_connect'  # data path for graph adj and node feature

adj_all_path = str(data_path + '/adj/')  
feature_all_path = str(data_path + '/feature/')  
com_feature_path_g1 = './1018/composition/composition_feature_c_pca.csv' # data path for composition  

adj_shell_path = str(data_path + '/adj/')  
feature_shell_path = str(data_path + '/feature/')  
com_feature_path_g2 = './1018/composition/composition_feature_c_pca.csv'
```

And you have to change the corresponding input node number for different datasets:

| Dataset | Node num | com_size |
|:-------:|:--------:|:--------:|
| (a)     | 15       | [26, 20] |
| (b)     | 17       | [30, 20] |
| (c)     | 10       | [16, 20] |
| (d)     | 11       | [18, 20] |

## Acknowledgment

    The author Chen Shuizhou would like to thank Yi Liu and Yuchao Tang for the material data. And thank Wenbin Ye for the CSA programming, Wenya Hu for the explanation experiment using PGexplainer.

    This work was sponsored by the National Key Research and Development Program of China (No. 2018YFB0704400), Key Program of Science and Technology of Yunnan Province (No. 202002AB080001-2, 202102AB080019-3),Key Research Project of Zhejiang Laboratory (No.2021PE0AC02), Key Project of Shanghai Zhangjiang National Independent Innovation Demonstration Zone(No. ZJ2021-ZD-006). The authors gratefully appreciate the anonymous reviewers for their valuable comments.
