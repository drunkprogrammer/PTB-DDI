# PTB-DDI

![](https://github.com/drunkprogrammer/PTB-DDI/blob/main/Overall-Architecture.png)

## Contents

- [Overview](#overview)
- [Config PTB-DDI Environment](#Config PTB-DDI Environment)
- [Train & Test on the BIOSNAP dataset](#Train & Test on the BIOSNAP dataset)
- [Train & Test on the DrugBank dataset](#Train & Test on the DrugBank dataset)
- [Notice](#Notice)

## Overview 

PTB-DDI: Accurate and Simple Framework for Drug-Drug Interaction Prediction Based on Pre-trained Tokenizer and BiLSTM Model

## Config PTB-DDI Environment
```
conda create -n PTB-DDI
conda activate PTB-DDI
conda install python==3.10.0
conda install scikit-learn
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia 
conda install -c conda-forge matplotlib==3.5.1
conda install -c conda-forge numpy==1.22.0
conda install -c conda-forge pandas==1.3.5 tqdm==4.62.3 
conda install -c conda-forge transformers
pip install torch_geometric
pip install rdkit
pip install protobuf untangle deepchem bertviz
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
```

## Train & Test on the BIOSNAP dataset
** Parameter-sharing **
```
python3 main.py --train_root './datasets/BIOSNAP/biosnap_train/' --train_path 'train_val_biosnap_smiles_new.csv' --test_root './datasets/BIOSNAP/biosnap_test/' --test_path 'test_ biosnap_smiles_new.csv' --batch_size 8 --epochs 30 --lr 2e-5 --weight_decay 2e-4 --gamma 0.8 --dropout 0 --mode train --shared True --model_name biosnap --saved_root './trained_record/biosnap/'
```
** Parameter-independent **
```
python3 main.py --train_root './datasets/BIOSNAP/biosnap_train/' --train_path 'train_val_biosnap_smiles_new.csv' --test_root './datasets/BIOSNAP/biosnap_test/' --test_path 'test_ biosnap_smiles_new.csv' --batch_size 8 --epochs 30 --lr 2e-5 --weight_decay 2e-4 --gamma 0.8 --dropout 0 --mode train --shared False --model_name biosnap --saved_root './trained_record/biosnap/'
```
### Test using our best model
```
python3 test.py --test_root './datasets/BIOSNAP/biosnap_test/' --test_path 'test_ biosnap_smiles_new.csv' --batch_size 8 --epochs 30 --lr 2e-5 --weight_decay 2e-4 --gamma 0.8 --dropout 0 --mode test --shared True --model_name biosnap --saved_root './trained_record/biosnap/' --load_model_path './trained_record/biosnap/2024-01-13-12:47:42/'
```


## Train & Test on the DrugBank dataset
** Parameter-sharing **
```
python3 main.py --train_root './datasets/drugbank/drugbank_train/' --train_path 'train_ drugbank_smiles_new.csv' --test_root './datasets/drugbank/drugbank_test/' --test_path 'test_ drugbank_smiles_new.csv' --batch_size 16 --epochs 30 --lr 2e-5 --weight_decay 1e-2 --gamma 0.8 --dropout 0 --mode train --shared True --model_name drugbank --saved_root './trained_record/drugbank/'
```
** Parameter-independent **
```
python3 main.py --train_root './datasets/drugbank/drugbank_train/' --train_path 'train_ drugbank_smiles_new.csv' --test_root './datasets/drugbank/drugbank_test/' --test_path 'test_ drugbank_smiles_new.csv' --batch_size 16 --epochs 30 --lr 2e-5 --weight_decay 1e-2 --gamma 0.8 --dropout 0 --mode train --shared False --model_name drugbank --saved_root './trained_record/drugbank/'
```

### Test using our best model
```
python3 ddi2013.py --test_root './datasets/drugbank/drugbank_test/' --test_path 'test_ drugbank_smiles_new.csv' --batch_size 16 --lr 2e-5 --weight_decay 1e-2 --gamma 0.8 --dropout 0 --mode test --shared True --model_name drugbank --saved_root './trained_record/drugbank/' --load_model_path './trained_record/drugbank/2024-01-12-22:02:36/'
```

## Notice
If you use this code, please cite our paper:
```
@article{Qiu2024ptb-ddi,
  title={PTB-DDI: Accurate and Simple Framework for Drug-Drug Interaction Prediction Based on Pre-trained Tokenizer and BiLSTM Model},
  author={Qiu, Jiayue and Liu, Huanxiang},
  journal={***},
  year={2024}
}
```


The BIOSNAP dataset and DrugBank dataset are from the following paper:
```
@article{huang2019caster,
  title={CASTER: Predicting Drug Interactions with Chemical Substructure Representation},
  author={Huang, Kexin and Xiao, Cao and Hoang, Trong Nghia and Glass, Lucas M and Sun, Jimeng},
  journal={AAAI},
  year={2020}
}
```
