import time
import os

from argparse import ArgumentParser
import torch
import torch.nn as nn
from Classifier.fp.class_utils import load_model, predict

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 确保指定正确的 GPU 设备



def get_top_k_result(logits, k=3, sorted=True):
    indices = np.argsort(logits, axis=-1)[:, -k:]
    if sorted:
        tmp = []
        for item in indices:
            tmp.append(item[::-1])
        indices = np.array(tmp)
    values = []
    for idx, item in zip(indices, logits):
        p = item.reshape(1, -1)[:, idx].reshape(-1)
        values.append(p)
    values = np.array(values)
    return values, indices


class LabelClass():
    
    def __init__(self,select_smile,args=None):
        super().__init__()
        class_parser = ArgumentParser('Label class arguements')
        class_parser.add_argument('-g', '--gpu', default='cuda:0', help='GPU device to use')
        class_parser.add_argument('-d', '--dataset', default='USPTO_50k', help='Dataset to use')
        class_parser.add_argument('-b', '--batch-size', default=128, help='Batch size of dataloader')
        class_parser.add_argument('-n', '--num-epochs', type=int, default=50, help='Maximum number of epochs for training')
        class_parser.add_argument('-p', '--patience', type=int, default=10, help='Patience for early stopping')
        class_parser.add_argument('-cl', '--max-clip', type=int, default=20, help='Maximum number of gradient clip')
        class_parser.add_argument('-clu', '--cluster', type=int, default=77, help='the cluster num of molecules')
        class_parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='Learning rate of optimizer')
        class_parser.add_argument('-l2', '--weight-decay', type=float, default=1e-4, help='Weight decay of optimizer')
        class_parser.add_argument('-ss', '--schedule_step', type=int, default=10, help='Step size of learning scheduler')
        class_parser.add_argument('-nw', '--num-workers', type=int, default=0, help='Number of processes for data loading')
        class_parser.add_argument('-do', '--drop_out', type=int, default=0.8, help='dropout')
        class_parser.add_argument('-nb', '--nbit', type=int, default=8192, help='the nbits of fingerprint model')
        class_parser.add_argument('-nm', '--name', type=str, default="fp_FC2_r1_cutoff_0.6", help='the nbits of fingerprint model')
        class_parser.add_argument('--start', '-s',type=int, default=0, help='the start number of target smiles')
        class_parser.add_argument('--end','-e', type=int, default=1000, help='the end number of target smiles')
        class_args = class_parser.parse_args().__dict__
        class_args['mode'] = 'test'
        class_args['device'] = torch.device(class_args['gpu']) if torch.cuda.is_available() else torch.device('cpu')
        print('Using device %s' % class_args['device'])
        class_args['data_dir'] = '../data/%s' % class_args['dataset']
        class_args['cluster'] = args.classes
        # select_smile = smi
        self.class_labels,self.logits = self.run(class_args["learning_rate"], class_args["weight_decay"], class_args["schedule_step"], class_args["drop_out"], class_args, select_smile,args)
        self.smiles = [select_smile]*10
        self.data = pd.DataFrame({
            'reactants>reagents>production':self.smiles,
            'class':self.class_labels[0]
        })
        self.props = self.logits[0] 
    def run(self,learning_rate, weight_decay, schedule_step, drop_out, class_args, test_smile,args):
        # print("recommeding begin")
        # print(time.ctime())
        # model_name = "%s_optimizer_original_%s_fp.pth" % (class_args['dataset'],class_args["name"])
        class_args['model_path'] = args.class_prediction_model
        # print(os.getcwd())
        # Classifier/models/USPTO_50k_optimizer_original_fp_FC2_r1_cutoff_0.6_fp.pth
        model, _, _, _ = load_model(class_args, learning_rate, weight_decay, int(schedule_step), drop_out)
    
        checkpoint = torch.load(class_args['model_path'])
        model.load_state_dict(checkpoint['net'])
        top_10_indices,top_10_logists = self.write_edits(class_args, model, test_smile)
     
        # print(time.ctime())
        # print("recommending done")
        return top_10_indices,top_10_logists

    def write_edits(self,class_args, model, test_smile):
        model.eval()
        with torch.no_grad():
            fp = torch.tensor([np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(test_smile), 2, nBits=8192))],dtype=torch.int64)
            fp = fp.to(class_args["device"])
            ori_logits = predict(model, fp)
            batch_logits = nn.Softmax(dim=1)(ori_logits)
            a = batch_logits.cpu().numpy()
            top_10_logits, top_10_indices = get_top_k_result(a, k=10)
            return top_10_indices,top_10_logits

 

if __name__ == "__main__":
    smi = 'CCO'
    result = LabelClass(smi)
    print(result.class_labels)
    print(result.smiles)



