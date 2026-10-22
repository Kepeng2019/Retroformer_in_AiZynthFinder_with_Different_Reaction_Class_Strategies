"""
Wrapper for SingleStepModelZoo for MHNreact (https://github.com/ml-jku/mhn-react).

Code and logic for model setup and model call is adapted from the original source code.
Where possible the original code is used.

P. Seidl, P. Renz, N. Dyubankova, P. Neves, J. Verhoeven, J. K. Wegner, M. Segler, S. Hochreiter, and G. Klambauer, 
“Improving Few- and Zero-Shot Reaction Template Prediction Using Modern Hopfield Networks,” 
Journal of Chemical Information and Modeling, vol. 62, no. 9, pp. 2111–2120, 2022, publisher: American Chemical Society. [Online]. 
Available: https://doi.org/10.1021/acs.jcim.1c01065
"""

import argparse
import os
import pickle as pkl
import sys
import time
import uuid
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import Memory
from pyexpat import model
from rdchiral.template_extractor import extract_from_reaction
from rdkit import Chem
from external.modelsmatter_modelzoo.ssbenchmark.ssmodels.base_model import Registries, SSMethod
from external.modelsmatter_modelzoo.ssbenchmark.utils import canonicalize_smiles


@Registries.ModelChoice.register(key="mhn_react")
class model_mhn_react(SSMethod):
    def __init__(self, module_path=None, max_workers=2, batch_size=1):
        self.model_name = "mhn_react"
        self.max_workers = max_workers  # 限制并行worker数量，避免内存爆炸
        self.batch_size = batch_size    # 批处理大小
        if module_path is not None:
            sys.path.insert(len(sys.path), os.path.abspath(module_path))

    def _extract_from_reaction(
        self,
        reaction_dicts,
    ):  # Added by Paula: some non-conventional reactions fail and since were using pool makes the whole batch fail
        try:
            re = extract_from_reaction(reaction_dicts)
            if "reaction_smarts" not in re:
                re["reaction_smarts"] = ""
            return re
        except:
            return {}

    def _process_batch(self, df, reaction_col):
        """处理单个批次，避免多进程内存开销"""
        import hashlib
        
        def create_hash(pd_row):
            return hashlib.md5(pd_row.to_json().encode()).hexdigest()

        if "reaction_type" in df.columns:
            df = df.drop(columns=["reaction_type"])

        if "_id" not in df.columns:
            df["_id"] = df.apply(create_hash, axis=1)

        df.rename(columns={reaction_col: "reaction_smiles"}, inplace=True)
        if "product" in df.columns:
            df.rename(columns={"product": "prod_smiles"}, inplace=True)

        reactants, spectators, products = list(
            zip(*[s.split(">") for s in df["reaction_smiles"]])
        )
        df["reactants"] = reactants
        df["spectators"] = spectators
        df["products"] = products

        df["prod_smiles"] = df.products.apply(canonicalize_smiles)

        # 串行处理，避免多进程内存开销
        reaction_dicts = [row.to_dict() for i, row in df.iterrows()]
        res = []
        total = len(reaction_dicts)
        for i, r_dict in enumerate(reaction_dicts):
            if i % 100 == 0:
                print(f"  Processing template {i}/{total}")
            res.append(self._extract_from_reaction(r_dict))

        reaction_smarts = [
            r["reaction_smarts"]
            if (r is not None) and ("reaction_smarts" in r.keys())
            else np.nan
            for r in res
        ]
        df["reaction_smarts"] = reaction_smarts
        df = df.dropna(axis=0, subset=["reaction_smarts"])

        def canonicalize_reactants(smiles, can_steps=2):
            if can_steps == 0:
                return smiles

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.nan
                
            for a in mol.GetAtoms():
                a.ClearProp("molAtomMapNumber")

            smiles = Chem.MolToSmiles(mol, True)
            if can_steps == 1:
                return smiles

            mol2 = Chem.MolFromSmiles(smiles)
            if mol2 is None:
                return smiles
            smiles = Chem.MolToSmiles(mol2, True)
            if can_steps == 2:
                return smiles

            raise ValueError("Invalid can_steps")

        df["reactants_can"] = [
            canonicalize_reactants(r, can_steps=2) if r is not None else np.nan
            for r in df["reactants"]
        ]
        
        # 清理临时变量
        del reaction_dicts, res
        import gc
        gc.collect()
        
        return df

    def _preprocess(self, df, reaction_col):
        """预处理，支持分批处理避免内存爆炸"""
        import psutil
        
        # 检查内存使用情况
        available_memory = psutil.virtual_memory().available / 1024**3
        df_size_gb = df.memory_usage(deep=True).sum() / 1024**3
        
        print(f"Available memory: {available_memory:.1f}GB, Data size: {df_size_gb:.1f}GB")
        
        # 如果数据太大，分批处理
        if len(df) > self.batch_size or df_size_gb > (available_memory / 4):
            batch_size = min(self.batch_size, max(500, int(len(df) / 10)))
            print(f"Processing {len(df)} rows in batches of {batch_size}")
            all_results = []
            num_batches = (len(df) - 1) // batch_size + 1
            
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size].copy()
                print(f"Processing batch {i//batch_size + 1}/{num_batches} ({len(batch_df)} rows)")
                
                # 处理当前批次
                batch_result = self._process_batch(batch_df, reaction_col)
                all_results.append(batch_result)
                
                # 清理内存
                del batch_df
                gc.collect()
                
                # 检查内存
                current_memory = psutil.virtual_memory().available / 1024**3
                print(f"  Memory after batch: {current_memory:.1f}GB available")
            
            # 合并结果
            df = pd.concat(all_results, ignore_index=True)
            del all_results
            gc.collect()
        else:
            df = self._process_batch(df, reaction_col)

        # 后续处理
        def filter_by_dict(df, fil):
            for col, value in fil.items():
                if not isinstance(value, list):
                    value = [value]
                df = df[df[col].isin(value)]
            return df

        import re

        mre = ":\d+(?=])"
        df["reaction_smarts"] = df["reaction_smarts"].astype(str)
        df["unmapped_template"] = df["reaction_smarts"].apply(
            lambda x: re.sub(mre, "", x)
        )

        unmapped2idx = {}
        labels = []
        for split in ["train", "valid", "test"]:
            sub = filter_by_dict(df, {"split": split})
            for u in sub["unmapped_template"]:
                if u not in unmapped2idx:
                    label = len(unmapped2idx)
                    unmapped2idx[u] = label

        df["label"] = [unmapped2idx[u] for u in df["unmapped_template"]]
        if "class" in df.columns:
            df = df.drop(columns=["class"])
            
        return df[["prod_smiles", "reactants_can", "split", "reaction_smarts", "label"]]

    def preprocess(self, df, reaction_col, batched=False):
        if not batched:
            df = self._preprocess(df, reaction_col)
        return df

    def process_input(self, data, reaction_col, chunk_size=None):
        print("preprocess input")
        data = self.preprocess(data, reaction_col, chunk_size)
        return data

    def preprocess_store(self, data, preprocess_root, instance_name):
        print("preprocess store")
        opath = os.path.abspath(preprocess_root)
        if not self.check_path(opath):
            self.make_path(opath)
        data = data.reset_index(drop=True)
        data.to_csv(os.path.join(opath, f"{instance_name}_MHN_prepro.csv.gz"), 
                   compression='gzip', index=False)

    def process_output(self, data_root, instance_name, k):
        with open(
            data_root,
            "rb",
        ) as f:
            data = pkl.load(f)
        data = [d[:k] for d in data]
        return [[canonicalize_smiles(p) for p in pred] for pred in data]

    def sort_by_template_and_flatten(
        self, template_scores, prod_idx_reactants, agglo_fun=sum
    ):  # Adapted from mhnreact
        flat_results = []
        flat_priors = []
        for ii in range(len(template_scores)):
            idx_prod_reactants = defaultdict(list)
            for k, v in prod_idx_reactants[ii].items():
                for iv in v:
                    idx_prod_reactants[iv].append(template_scores[ii, k])
            d2 = {k: agglo_fun(v) for k, v in idx_prod_reactants.items()}
            if len(d2) == 0:
                flat_results.append([])
            else:
                res = pd.DataFrame.from_dict(d2, orient="index").sort_values(
                    0, ascending=False
                )
                flat_results.append(res.index.values.tolist())
                flat_priors.append(res[0].values.tolist())
        return flat_results, flat_priors

    def model_setup(
        self,
        use_gpu=False,
        model_fn="",
        model_path="",
        template_path="",
        large_dataset=False,
        **kwargs,
    ):
        import pandas as pd
        from external.modelsmatter_modelzoo.external_models.modelsmatter_mhn_react_hpc.mhnreact.data import load_dataset_from_csv
        from external.modelsmatter_modelzoo.external_models.modelsmatter_mhn_react_hpc.mhnreact.inspect import load_clf
        from external.modelsmatter_modelzoo.external_models.modelsmatter_mhn_react_hpc.mhnreact.molutils import LargeDataset
        
        self.model = load_clf(model_fn, model_path, model_type="mhn", device="cpu")
        self.bs = kwargs["bs"] if "bs" in kwargs else 512  # 减小batch size
        self.model.eval()
        
        X, y, template_list, test_reactants_can = load_dataset_from_csv(
            csv_path=template_path,
            split_col="split",
            input_col="prod_smiles",
            ssretroeval=True,
            reactants_col="reactants_can",
            ret_df=False,
        )
        self.templates = list(template_list.values())
        self.model.set_templates(self.templates)
        self.template_product_smarts = [str(s).split(">")[0] for s in self.templates]
        
        if large_dataset:
            self.large_dataset = LargeDataset(
                kwargs["fp_type"] if "fp_type" in kwargs.keys() else "morgan",
                kwargs["template_fp_type"]
                if "template_fp_type" in kwargs.keys()
                else "rdk",
                kwargs["fp_size"] if "fp_size" in kwargs.keys() else 4096,
                kwargs["fp_radius"] if "fp_radius" in kwargs.keys() else 2,
                template_list,
                kwargs["njobs"] if "njobs" in kwargs.keys() else 1,  # 限制为1
                kwargs["only_templates_in_batch"]
                if "only_templates_in_batch" in kwargs.keys()
                else False,
                kwargs["verbose"] if "verbose" in kwargs.keys() else False,
            )
        
        # 修改缓存设置，限制大小
        cachedir = kwargs.get("cachedir", "./data/cache")
        self.cachedir = os.path.join(cachedir, str(uuid.uuid4()))
        self.cachedir_template = kwargs["cachedir_template"] if "cachedir_template" in kwargs else cachedir
        
        # 限制模板缓存数量，避免过多文件
        cache_limit = kwargs.get("template_cache_limit", 8)  # 限制缓存数量
        self.model.template_cache = {i: os.path.join(self.cachedir_template, f"tmp_{i}.pt") 
                                     for i in range(cache_limit)}

    def _model_call(self, X):
        from external.modelsmatter_modelzoo.external_models.modelsmatter_mhn_react_hpc.mhnreact.molutils import smarts2appl
        from external.modelsmatter_modelzoo.external_models.modelsmatter_mhn_react_hpc.mhnreact.retroeval import run_templates

        # 限制缓存大小
        cache_limit_gb = 5  # 限制为5GB
        memory = Memory(self.cachedir, verbose=0, bytes_limit=cache_limit_gb * 1024**3)
        
        X = [[x] for x in X]
        
        # 如果输入太大，分批处理
        max_batch_size = 2
        if len(X) > max_batch_size:
            print(f"Processing {len(X)} inputs in batches of {max_batch_size}")
            all_flat_results = []
            all_flat_priors = []
            
            for i in range(0, len(X), max_batch_size):
                batch_X = X[i:i+max_batch_size]
                print(f"Processing batch {i//max_batch_size + 1}/{(len(X)-1)//max_batch_size + 1}")
                
                flat_results, flat_priors = self._process_batch_call(batch_X, memory)
                all_flat_results.extend(flat_results)
                all_flat_priors.extend(flat_priors)
                
                # 清理内存
                import gc
                gc.collect()
            
            return all_flat_results, all_flat_priors
        else:
            return self._process_batch_call(X, memory)
    
    def _process_batch_call(self, X, memory):
        """处理单批次的模型调用"""
        from external.modelsmatter_modelzoo.external_models.modelsmatter_mhn_react_hpc.mhnreact.molutils import smarts2appl
        from external.modelsmatter_modelzoo.external_models.modelsmatter_mhn_react_hpc.mhnreact.retroeval import run_templates

        print("execute all templates")
        test_product_smarts = [xi[0] for xi in X]
        
        # 使用缓存但要限制
        smarts2appl_cached = memory.cache(smarts2appl, ignore=["njobs", "nsplits"])
        memory.reduce_size()
        
        appl = smarts2appl_cached(test_product_smarts, self.template_product_smarts)
        
        if (appl[0].size | appl[1].size) == 0:
            print(f"No templates were applicable for {X}, returning empty lists")
            return [], []
            
        y_preds = None
        y = np.zeros(len(X)).astype(np.int)
        
        if y_preds is None:
            y_preds = self.model.evaluate(
                X,
                X,
                y,
                is_smiles=True,
                split="test",
                only_loss=True,
                wandb=None,
                bs=self.bs,
                cachedir=self.cachedir_template,
            )

        template_scores = y_preds

        # 限制top-k数量，减少内存
        if y_preds.shape[1] > 10000:  # 降低阈值
            kth = 100  # 降低k值
            print(f"only evaluating top {kth} applicable predicted templates")
            appl_mtrx = np.zeros_like(y_preds, dtype=bool)
            appl_mtrx[appl[0], appl[1]] = 1

            appl_and_topkth = ([], [])
            for row in range(len(y_preds)):
                argpreds = np.argpartition(
                    -(y_preds[row] * appl_mtrx[row]), min(kth, y_preds.shape[1]-1), axis=0
                )[:kth]
                mask = appl_mtrx[row][argpreds]
                argpreds = argpreds[mask]

                appl_and_topkth[0].extend([row for _ in range(len(argpreds))])
                appl_and_topkth[1].extend(list(argpreds))

            appl = appl_and_topkth

        print("running the templates")
        run_templates_func = run_templates
        prod_idx_reactants, prod_temp_reactants = run_templates_func(
            test_product_smarts, self.templates, appl
        )

        flat_results, flat_priors = self.sort_by_template_and_flatten(
            y_preds, prod_idx_reactants, agglo_fun=sum
        )
        
        # 限制输出大小
        beam_size = 20  # 减小beam size
        if len(flat_results) > beam_size:
            flat_results = flat_results[:beam_size]
            flat_priors = flat_priors[:beam_size]
        
        # 清理缓存
        memory.clear(warn=False)
        
        return flat_results, flat_priors

    def model_call(self, X):
        return self._model_call(X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_path", type=str, default=None)
    parser.add_argument("--preprocess_batch", type=str, default=None)
    parser.add_argument("--preprocess_join", type=str, default=None)
    parser.add_argument("--reaction_col", type=str, default="reaction_smiles")
    parser.add_argument("--preprocess_output", type=str, default=None)
    parser.add_argument("--instance_name", type=str, default=None)
    parser.add_argument("--max_workers", type=int, default=2, help="Maximum parallel workers")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")

    args = parser.parse_args()

    model_instance = model_mhn_react(module_path=args.module_path, 
                                     max_workers=args.max_workers,
                                     batch_size=args.batch_size)
    
    if args.preprocess_batch is not None:
        df = model_instance.read_csv(args.preprocess_batch)
        df = model_instance._preprocess(df, args.reaction_col)
        model_instance.write_csv(df, args.preprocess_output, str(uuid.uuid4()))

    if args.preprocess_join is not None:
        df = model_instance.gather_batches(args.preprocess_join)
        df = model_instance.preprocess(df, args.reaction_col, batched=True)
        model_instance.preprocess_store(df, args.preprocess_output, args.instance_name)