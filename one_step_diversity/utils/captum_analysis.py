from argparse import ArgumentParser
import torch
import torch.nn as nn
from utils import load_model
import pandas as pd
import time
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem,Draw
import os
import pickle
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import re
from PIL import Image
import io

def main(learning_rate, weight_decay, schedule_step, drop_out, args, test_loader):
    print("recommeding begin")
    print(time.ctime())
    model_name = "%s_optimizer_original_%s_fp.pth" % (args['dataset'],args["name"])
    args['model_path'] = '../models/'+model_name
    import os
    print(os.getcwd())
    # Classifier/models/USPTO_50k_optimizer_original_fp_FC2_r1_cutoff_0.6_fp.pth
    model, _, _, _ = load_model(args, learning_rate, weight_decay, int(schedule_step), drop_out)
    checkpoint = torch.load(args['model_path'])
    model.load_state_dict(checkpoint['net'])
    return model
 
parser = ArgumentParser('LocalRetro testing arguements')
parser.add_argument('-g', '--gpu', default='cuda:0', help='GPU device to use')
parser.add_argument('-d', '--dataset', default='USPTO_50k', help='Dataset to use')
parser.add_argument('-b', '--batch-size', default=128, help='Batch size of dataloader')
parser.add_argument('-n', '--num-epochs', type=int, default=50, help='Maximum number of epochs for training')
parser.add_argument('-p', '--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('-cl', '--max-clip', type=int, default=20, help='Maximum number of gradient clip')
parser.add_argument('-clu', '--cluster', type=int, default=77, help='the cluster num of molecules')
parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='Learning rate of optimizer')
parser.add_argument('-l2', '--weight-decay', type=float, default=1e-4, help='Weight decay of optimizer')
parser.add_argument('-ss', '--schedule_step', type=int, default=10, help='Step size of learning scheduler')
parser.add_argument('-nw', '--num-workers', type=int, default=0, help='Number of processes for data loading')
parser.add_argument('-do', '--drop_out', type=int, default=0.8, help='dropout')
parser.add_argument('-nb', '--nbit', type=int, default=8192, help='the nbits of fingerprint model')
parser.add_argument('-nm', '--name', type=str, default="fp_FC2_r1_cutoff_0.6", help='the nbits of fingerprint model')
args, unknown = parser.parse_known_args()
args = args.__dict__
args['mode'] = 'test'
args['device'] = torch.device(args['gpu']) if torch.cuda.is_available() else torch.device('cpu')
print('Using device %s' % args['device'])
args['data_dir'] = '../data/%s' % args['dataset']

df = pd.read_csv("../../clustering/scripts/data/cluster_test_50k.csv")
smiles = df['prod_smiles'].tolist()
pickle_name = "../data/fp_test_%s_for_vis.pkl" % (args["cluster"])

if os.path.exists(pickle_name):
    with open(pickle_name,'rb') as f:
        fps = pickle.load(f)
else:
    fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=8192)) for smi in df['prod_smiles']]
    with open(pickle_name, 'wb') as f:  
        pickle.dump(fps, f)

model = main(args["learning_rate"], args["weight_decay"], args["schedule_step"], args["drop_out"], args,fps)

def func(m,attributions_index,attributions_weight,target_class,output_path):  
        plt.bar(range(len(attributions_index)), attributions_weight)
        plt.xlabel('Fingerprint Bit')
        plt.ylabel('Attribution Value')
        plt.title(f'Feature Attribution for Class {target_class}')
        # plt.show()
        plt.savefig(output_path+'/molecule_{:04d}_bar_ditributions.png'.format(m))
        plt.close()

        smi = smiles[m]
        mol = Chem.MolFromSmiles(smi)  # 例如乙醇 (C2H5OH)

        radius = 2 
        info = {}
        fp_explain = AllChem.GetMorganFingerprintAsBitVect(mol, radius, bitInfo=info,nBits=8192)


        ECFP_tuples = [(mol, bit, info) for bit in list(info.keys())]

        ECFP_part1, ECFP_part2 = [],[]
        legend1,legend2 = [],[]
        for i in range(len(attributions_weight)):
            if attributions_weight[i]>0:
                ECFP_part1.append(ECFP_tuples[i])
                legend1.append(str(list(info.keys())[i])+': {:4.2f}'.format(attributions_weight[i]))
            else:
                ECFP_part2.append(ECFP_tuples[i])
                legend2.append(str(list(info.keys())[i])+': {:4.2f}'.format(attributions_weight[i]))


        print('positive substructures')
        img = Draw.DrawMorganBits(ECFP_part1, molsPerRow=4, legends=legend1)
        # 确保 img 是图像对象
        if isinstance(img, Draw.MolDraw2DCairo):
            img = img.GetDrawingText()  # 转换为图像对象，避免直接用 PIL Image 方法
        img.save(output_path+'/molecule_{:04d}_of_positive_substructures.png'.format(m))

        print('negtive substructures')
        img = Draw.DrawMorganBits(ECFP_part2, molsPerRow=4, legends=legend2)
        img.save(output_path+'/molecule_{:04d}_of_negtive_substructures.png'.format(m))

model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 禁用梯度计算
    for m in range(len(df)):
        ig = IntegratedGradients(model)
        fp = torch.tensor(fps[m],dtype=torch.float).unsqueeze(0)
        fp = fp.requires_grad_()
        fp = fp.to(args['device'])
        print(fp)
        target_class = df['class'].values.tolist()[0] # 实际聚类分类为5，预测的结果是9
        pred_target_classes = model(fp).squeeze().detach().cpu()
        _,pred_topk = torch.topk(pred_target_classes,10)
        print(pred_topk)
        pred_class = pred_topk.numpy().tolist()[0]
        attr, delta = ig.attribute(fp,target=target_class, return_convergence_delta=True)
        attributions = attr.squeeze().detach().cpu().numpy()
        pred_attr, pred_delta = ig.attribute(fp,target=pred_class, return_convergence_delta=True)
        pred_attributions = pred_attr.squeeze().detach().cpu().numpy()
        print(attr)

        attributions_index = [i for i in range(len(attributions)) if attributions[i]!=0]
        print([str(i)+':'+str(attributions_index[i]) for i in range(len(attributions_index))])
        attributions_weight = [attributions[i] for i in range(len(attributions)) if attributions[i]!=0]

        pred_attributions_index = [i for i in range(len(pred_attributions)) if pred_attributions[i]!=0]
        print([str(i)+':'+str(pred_attributions_index[i]) for i in range(len(pred_attributions_index))])
        pred_attributions_weight = [pred_attributions[i] for i in range(len(pred_attributions)) if pred_attributions[i]!=0]

        path1 = './analysis_result_by_captum/true_cluster_class'
        os.makedirs(path1,exist_ok=True)
        path2 = './analysis_result_by_captum/top1_pred_class'
        os.makedirs(path2,exist_ok=True)
                
        
        func(m+1,attributions_index,attributions_weight,target_class,output_path=path1)
        func(m+1,pred_attributions_index,pred_attributions_weight,pred_class,output_path=path2)
        break


    