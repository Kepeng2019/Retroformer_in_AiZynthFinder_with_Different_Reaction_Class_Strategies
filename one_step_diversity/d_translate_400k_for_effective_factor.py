import sys,os
sys.path.append(os.getcwd())
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from one_step_diversity.utils.utils_reactions_400k_version2 import load_model, load_dataloader, predict
import pandas as pd
import time
import numpy as np
# RDKit相关导入
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.getcwd())

def argments_one_step_diversity():
    parser = ArgumentParser('LocalRetro testing arguements')
    parser.add_argument('-g', '--gpu', default='cuda:0', help='GPU device to use')
    parser.add_argument('-d', '--dataset', default='USPTO_400k', help='Dataset to use')
    parser.add_argument('-b', '--batch-size', default=128, help='Batch size of dataloader')
    parser.add_argument('-n', '--num-epochs', type=int, default=50, help='Maximum number of epochs for training')
    parser.add_argument('-p', '--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('-cl', '--max-clip', type=int, default=20, help='Maximum number of gradient clip')
    parser.add_argument('-clu', '--cluster', type=int, default=113, help='the cluster num of molecules')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='Learning rate of optimizer')
    parser.add_argument('-l2', '--weight-decay', type=float, default=1e-4, help='Weight decay of optimizer')
    parser.add_argument('-ss', '--schedule_step', type=int, default=10, help='Step size of learning scheduler')
    parser.add_argument('-nw', '--num-workers', type=int, default=0, help='Number of processes for data loading')
    parser.add_argument('-do', '--drop_out', type=int, default=0.8, help='dropout')
    parser.add_argument('-nb', '--nbit', type=int, default=8192, help='the nbits of fingerprint model')
    parser.add_argument('-nm', '--name', type=str, default="fp_FC2_r1_cutoff_0.6", help='the nbits of fingerprint model')
        # parser.add_argument('-dt', '--data_test_csv', type=str, default="data_for_translate_400k/data_class_3_multi_beam_10_diversity_1_processed.csv", help='the test data csv file')
    args = parser.parse_args().__dict__
    args['device'] = torch.device(args['gpu']) if torch.cuda.is_available() else torch.device('cpu')


    # load model
    model_name = 'models_to_predict_cluster_reaction_class_by_reaction' 
    args['model_path'] = 'one_step_diversity/models_by_reactions/' + model_name+'_400k.pt'

    return args


def USPTOTestDataset(product_smile, reactant_smiles):

    smi = product_smile
    f1=np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),2,nBits=8192))
    fps = []
    react_smi = reactant_smiles
    
    for j in range(len(react_smi)):
        react_smi_j = react_smi[j]
        try:
            if react_smi_j.startswith('.'):
                react_smi_j = react_smi_j[1:]
            f2=np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(react_smi_j),2,nBits=8192))
            fps.append(f1-f2)
        except Exception as e:
            continue
        
    return react_smi, fps  

def dedup_by_similarity(probs, threshold):
    """策略: 基于概率分布的余弦相似度"""
    # 计算余弦相似度矩阵
    sim_matrix = cosine_similarity(probs.cpu().numpy())
    
    # 贪心聚类
    n = len(probs)
    used = [False] * n
    unique_indices = []
    
    for i in range(n):
        if used[i]:
            continue
        unique_indices.append(i)
        # 标记所有与i相似度超过阈值的样本
        for j in range(i+1, n):
            if not used[j] and sim_matrix[i, j] >= threshold:
                used[j] = True
    
    return unique_indices, sim_matrix


def diversity_deduplicate(model, learning_rate, weight_decay, schedule_step, drop_out, args, product_smile, reactant_smiles, threshold=0.85):
    react_smi, fps = USPTOTestDataset(product_smile, reactant_smiles,)

    model.eval()
    with torch.no_grad():
        rxn_classes = []
        rxn_class_probs = []
   
        preds = predict(model, torch.tensor(fps).to(args["device"]))
        # rxn_class_prob,rxn_class = torch.topk(preds, k=10, dim=-1)
        ids, sim_matrix = dedup_by_similarity(preds,threshold=threshold)
        react_smi = [react_smi[i] for i in ids]
        # print(sim_matrix)
        # print(react_smi)
        return react_smi

def diversity_400k(product_smile='CN(C)Cc1cc2c(C3=C(c4cn(C)c5ccccc45)C(=O)NC3=O)c(Cl)ccc2cn1',
        reactant_smiles=['Cn1cc(C2=C(c3c(Cl)ccc4cnc(CF)cc34)C(=O)NC2=O)c2ccccc21', 'Cn1cc(C2=C(c3c(Cl)ccc4cnc(CO)cc34)C(=O)NC2=O)c2ccccc21'], 
        threshold=0.85):

    args = argments_one_step_diversity()

    model, _, _, _ = load_model(args, args['learning_rate'], args['weight_decay'], int(args['schedule_step']), args['drop_out'])
    checkpoint = torch.load(args['model_path'])
    model.load_state_dict(checkpoint['net'])

    return diversity_deduplicate(model,args['learning_rate'], args['weight_decay'], args['schedule_step'], args['drop_out'], args,
        product_smile=product_smile,
        reactant_smiles=reactant_smiles,
        threshold=threshold)

