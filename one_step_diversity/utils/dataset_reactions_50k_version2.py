# 标准库导入
import gzip
import math
import os
import random
from collections import defaultdict
import pickle
import _pickle as cPickle

# 第三方库导入
import numpy as np
import pandas as pd

# RDKit相关导入
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import DataStructs


class USPTODataset(object):
    def __init__(self, args,rank):
        df = pd.read_csv("data/paroutes_data/cluster_train_valid_50k_version2.csv")
        # a = 'clustering/scripts/data_paroutes_50k/cluster_train_valid_50k.csv'
        if rank:
            df = df.loc[df["Rank"]==rank].reset_index()
        self.train_ids = df.index[df['split'] == 'train'].values
        self.val_ids = df.index[df['split'] == 'valid'].values
        self.smiles = df['prod_smiles'].tolist()
        self.reactant_smiles = df['reac_smiles'].tolist()
        #cluster_name = str(args["cluster"])+"_"+"class"
        self.labels = [[t] for t in df[args["cluster_name"]]]
        self.cluster = int(max(self.labels)[0]+1)
        import random
        a = random.randint(1,100)
        pickle_name = "data/train_valid_product_smiles_to_%s_50k_%d.pkl"%(args["cluster_name"], a)
        def func(smi,nBits = 8192):
            try:
                mol = Chem.MolFromSmiles(smi)
                mol.UpdatePropertyCache(strict=False)
                Chem.GetSSSR(mol)
                fp = AllChem.GetTopologicalTorsionFingerprint(mol=mol)
                # print(np.array(fp))
                fp_array = set()
                for idx in fp.GetNonzeroElements().keys():
                    fp_array.add(idx)
                
                
                bit_vector = np.zeros(nBits, dtype=int)

                # 哈希函数：将每个数映射到二进制位上
                def hash_to_bit_position(value):
                    return hash(value) % nBits

                # 将每个十进制数映射到相应的二进制位置
                for num in fp_array:
                    bit_pos = hash_to_bit_position(num)
                    bit_vector[bit_pos] = 1
                return bit_vector
            except:
                return np.zeros(nBits, dtype=int)
        if os.path.exists(pickle_name):
            with  open(pickle_name, 'rb')  as f1:
                self.fps = pickle.load(f1)
        else:
            # self.fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),2,nBits=8192)) for smi in df['prod_smiles']]
            self.fps = []
            for i in range(len(self.smiles)):
                smi = self.smiles[i]
                react_smi = self.reactant_smiles[i]
                # print("Processing SMILES:", smi)
                f1=np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),2,nBits=8192))
                f2=np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(react_smi),2,nBits=8192))
                self.fps.append( f1-f2)
        # self.fps = [func(smi) for smi in df['prod_smiles']]
        # self.fps = [np.concatenate((func(smi),np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),2,nBits=8192)))) for smi in df['prod_smiles']]
            
            pickle.dump(self.fps, open(pickle_name, "wb"))

    def __getitem__(self, item):
        return self.smiles[item], self.fps[item], self.labels[item]  # remove the test

    def __len__(self):
            return len(self.smiles)

class USPTOTestDataset(object):
    def __init__(self, args,rank):
            df = pd.read_csv("data/paroutes_data/cluster_class_test_50k_version2.csv")
            # a = 'clustering/scripts/data_paroutes_50k/cluster_train_valid_50k.csv'
            if rank:
                df = df.loc[df["Rank"]==rank].reset_index()
            # self.train_ids = df.index[df['split'] == 'train'].values
            # self.val_ids = df.index[df['split'] == 'valid'].values
            # self.test_ids = df.index[df['split'] == 'test'].values
            self.smiles = df['prod_smiles'].tolist()
            self.reactant_smiles = df['reac_smiles'].tolist()
            #cluster_name = str(args["cluster"])+"_"+"class"
            self.labels = [[t] for t in df[args["cluster_name"]]]
            self.cluster = int(max(self.labels)[0]+1)
            import random
            a = random.randint(1,100)
            pickle_name = "data/test_react_product_smiles_to_%s_50k_%d.pkl"%(args["cluster_name"], a)
            def func(smi,nBits = 8192):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    mol.UpdatePropertyCache(strict=False)
                    Chem.GetSSSR(mol)
                    fp = AllChem.GetTopologicalTorsionFingerprint(mol=mol)
                    # print(np.array(fp))
                    fp_array = set()
                    for idx in fp.GetNonzeroElements().keys():
                        fp_array.add(idx)
                    
                    
                    bit_vector = np.zeros(nBits, dtype=int)

                    # 哈希函数：将每个数映射到二进制位上
                    def hash_to_bit_position(value):
                        return hash(value) % nBits

                    # 将每个十进制数映射到相应的二进制位置
                    for num in fp_array:
                        bit_pos = hash_to_bit_position(num)
                        bit_vector[bit_pos] = 1
                    return bit_vector
                except:
                    return np.zeros(nBits, dtype=int)
            if os.path.exists(pickle_name):
                with  open(pickle_name, 'rb')  as f1:
                    self.fps = pickle.load(f1)
            else:
                # self.fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),2,nBits=8192)) for smi in df['prod_smiles']]
                self.fps = []
                for i in range(len(self.smiles)):
                    smi = self.smiles[i]
                    react_smi = self.reactant_smiles[i]
                    # print("Processing SMILES:", smi)
                    f1=np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),2,nBits=8192))
                    f2=np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(react_smi),2,nBits=8192))
                    self.fps.append( f1-f2)
            # self.fps = [func(smi) for smi in df['prod_smiles']]
            # self.fps = [np.concatenate((func(smi),np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),2,nBits=8192)))) for smi in df['prod_smiles']]
                
                pickle.dump(self.fps, open(pickle_name, "wb"))

    def __getitem__(self, item):
        return self.smiles[item], self.fps[item], self.labels[item]  # remove the test

    def __len__(self):
            return len(self.smiles)


class USPTOvisDataset(object):
    def __init__(self, args):
        df = pd.read_csv("../../clustering/scripts/data/50k_small/cluster_train_valid_50k.csv")
        template_csv = pd.read_csv("../../clustering/scripts/data/50k_small/uspto_50k_preparation.csv")
        self.prod_center = [a.split(">>")[0] for a in template_csv["template_r3"][template_csv['split'] == 'valid']]

        self.smiles = df['prod_smiles'][df['split'] == 'valid'].tolist()

        #cluster_name = str(args["cluster"]) + "_" + "class"
        pickle_name = "../data/fp_vis_%s_clean.pkl" % (args["cluster_name"])
        self.labels = [[t] for t in df[args["cluster_name"]][df['split'] == 'valid']]
        self.cluster = int(max(self.labels)[0]+1)

        if os.path.exists(pickle_name):
            with  open(pickle_name, 'rb')  as f1:
                self.fps = pickle.load(f1)
        else:
            self.fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=8192)) for smi
                        in self.smiles]
            pickle.dump(self.fps, open(pickle_name, "wb"))

    def __getitem__(self, item):
        return self.smiles[item], self.fps[item], self.labels[item],self.prod_center[item]

    def __len__(self):
        return len(self.smiles)

class USPTOKDDataset(object):
    def __init__(self, args,epoch):
        df = pd.read_csv('%s/55_out.csv' % args['data_dir'])
        self.high_id = df.index[(df['split'] == 'train') & (df['Rank']=='high')].values.tolist()
        self.middle_id = df.index[(df['split'] == 'train') & (df['Rank']=='middle')].values.tolist()
        self.low_id = df.index[(df['split'] == 'train') & (df['Rank']=='low')].values.tolist()
        count_low = len(self.low_id)
        high_k = (len(self.high_id)-count_low)/math.log(args["num_epochs"]+1)
        middle_k = (len(self.middle_id)-count_low)/math.log(args["num_epochs"]+1)
        choose_middle_id = self.middle_id[:int(middle_k*math.log(epoch+1)+count_low)+1]
        choose_high_id = self.high_id[:int(high_k*math.log(epoch+1)+count_low)+1]
        choose_low_id = self.low_id
        self.train_ids = np.array(choose_high_id+choose_middle_id+choose_low_id)
        self.val_ids = df.index[df['split'] == 'val'].values
        self.smiles = df['Products'].tolist()
        self.kd_logits = [a for a in df["logits"]]
        self.label_pos = df["label_pos"].to_list()
        self.masks = df["Rank"].to_list()
        cluster_name = str(args["cluster"])+"_"+"class"
        self.labels = [[t-1] for t in df[cluster_name]]
        pickle_name = "../data/fp_%s_%s_KD_utils.pkl"%(args["cluster"],args["nbit"])
        if os.path.exists(pickle_name):
            with  open(pickle_name, 'rb')  as f1:
                self.fps = pickle.load(f1)
        else:
            self.fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),2,nBits=args["nbit"])) for smi in df['Products']]
            pickle.dump(self.fps, open(pickle_name, "wb"))

    def __getitem__(self, item):
        return self.smiles[item], self.fps[item], self.labels[item],  self.kd_logits[item],  self.label_pos[item], self.masks[item]# remove the test

    def __len__(self):
            return len(self.smiles)
