import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from retroformer.utils.smiles_utils import *
from retroformer.utils.translate_utils_one_step import translate_batch_original, translate_batch_stepwise
from retroformer.utils.build_utils_one_step import build_model, build_iterator, load_checkpoint

# import re
# import os
import copy
import math
# import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import argparse


parser1 = argparse.ArgumentParser()

parser1.add_argument('--device', type=str, default='cuda:0', choices=['cuda', 'cpu'], help='device GPU/CPU')
parser1.add_argument('--batch_size_val', type=int, default=1, help='batch size')
parser1.add_argument('--batch_size_trn', type=int, default=1, help='batch size')
parser1.add_argument('--total_beam_size', type=int, default=30, help='beam size for one reaction')
parser1.add_argument('--class_size', type=int, default=1, help='the number of classes for one reaction')
parser1.add_argument('--beam_size', type=int, default=10, help='beam size for one class of one reaction')
parser1.add_argument('--stepwise', type=str, default='False', choices=['True', 'False'], help='')
parser1.add_argument('--use_template', type=str, default='False', choices=['True', 'False'], help='')

parser1.add_argument('--encoder_num_layers', type=int, default=4, help='number of layers of transformer')
parser1.add_argument('--decoder_num_layers', type=int, default=4, help='number of layers of transformer')
parser1.add_argument('--d_model', type=int, default=256, help='dimension of model representation')
parser1.add_argument('--heads', type=int, default=8, help='number of heads of multi-head attention')
parser1.add_argument('--d_ff', type=int, default=2048, help='')
parser1.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser1.add_argument('--known_class', type=str, default='True', help='with reaction class known/unknown')
parser1.add_argument('--shared_vocab', type=str, default='True', choices=['True', 'False'], help='whether sharing vocab')
parser1.add_argument('--shared_encoder', type=str, default=False, choices=['True', 'False'],
                    help='whether sharing encoder')

parser1.add_argument('--data_dir', type=str, default='external/modelsmatter_modelzoo/external_models/Retroformer_KD_as_one_step/data_random', help='base directory')
parser1.add_argument('--intermediate_dir', type=str, default='external/modelsmatter_modelzoo/external_models/Retroformer_KD_as_one_step/intermediate_FC2_r1_paroutes_50k_typed', help='intermediate directory')
parser1.add_argument('--checkpoint_dir', type=str, default='external/modelsmatter_modelzoo/external_models/Retroformer_KD_as_one_step/checkpoint_FC2_r1_paroutes_50k_typed', help='checkpoint directory')
parser1.add_argument('--checkpoint', type=str,default='std_model_10.pt', help='checkpoint model file')
parser1.add_argument('--mode', type=str, default='test', help='test file is large for lmdb, so divide in parts')
parser1.add_argument('--classes', type=int, default=77, help='the classes of fingerprint templates')

parser1.add_argument('--gpu_mode', type=bool, default=True, help='the classes of fingerprint templates')
parser1.add_argument('--real_beam_size', type=str, default='', help='a illustration of real_beam_size')
parser1.add_argument('--class_prediction_model', type=str, default='external/modelsmatter_modelzoo/external_models/Retroformer_as_one_step/Classifier/models/select_fp_iter2_USPTO_50k_optimizer_original_FC2_r1_cutoff_0.6.pt', help='a illustration of real_beam_size')
parser1.add_argument('--start', '-s',type=int, default=0, help='the start number of target smiles')
parser1.add_argument('--end','-e', type=int, default=1000, help='the end number of target smiles')

one_step_args = parser1.parse_args()


# one_step_args.encoder_num_layers=4
# one_step_args.decoder_num_layers=4
# one_step_args.batch_size_val=1
# one_step_args.shared_vocab='False'
# one_step_args.shared_encoder=False
# one_step_args.data_dir='data_random'
# one_step_args.intermediate_dir='external/modelsmatter_modelzoo/external_models/Retroformer_as_one_step/intermediate_FC2_r1_paroutes_400k'
# one_step_args.checkpoint_dir='external/modelsmatter_modelzoo/external_models/Retroformer_as_one_step/checkpoint_FC2_r1_paroutes_400k_untyped'
# one_step_args.checkpoint='std_model_775000.pt'
# one_step_args.known_class='False'
# # one_step_args.beam_size=3  control by total_beam_size and class_size
# one_step_args.total_beam_size=30
# one_step_args.class_size=3
# one_step_args.use_template=False
# one_step_args.mode='test'
# one_step_args.classes=113
# one_step_args.class_prediction_model='external/modelsmatter_modelzoo/external_models/Retroformer_as_one_step/Classifier/models/select_fp_iter2_USPTO_400k_optimizer_original_FC2_r1_cutoff_0.6.pt'


def translate(iterator, model, dataset):
    ground_truths = []
    generations = []
    value_scores = []
    if one_step_args.shared_vocab == 'True':
        invalid_token_indices = [dataset.tgt_stoi['<RX_{}>'.format(i)] for i in range(0,one_step_args.classes)]
    else:
        invalid_token_indices = []
    invalid_token_indices += [dataset.tgt_stoi['<unk>']]
    # Translate:
    for batch in tqdm(iterator, total=len(iterator)):
        src, tgt, _, _, _ = batch
        # print(iterator.dataset.data,flush=True)
        if one_step_args.stepwise == 'False':
            # Original Main:
            pred_tokens, pred_scores = translate_batch_original(model, batch, beam_size=one_step_args.beam_size,
                                                                invalid_token_indices=invalid_token_indices)
            for idx in range(batch[0].shape[1]):
                gt = ''.join(dataset.reconstruct_smi(tgt[:, idx], src=False))
                hypos = np.array([''.join(dataset.reconstruct_smi(tokens, src=False)) for tokens in pred_tokens[idx]])
                hypo_len = np.array([len(smi_tokenizer(ht)) for ht in hypos])
                new_pred_score = copy.deepcopy(pred_scores[idx]).cpu().numpy() / hypo_len
                ordering = np.argsort(new_pred_score)[::-1]

                ground_truths.append(gt)
                generations+=hypos[ordering].tolist()
                original_beam_size = pred_tokens.shape[1]
                value_scores+=new_pred_score[ordering].tolist()
        else:
            # Stepwise Main:
            # untyped: T=10; beta=0.5, percent_aa=40, percent_ab=40
            # typed: T=10; beta=0.5, percent_aa=40, percent_ab=55
            if one_step_args.known_class == 'True':
                percent_ab = 55
            else:
                percent_ab = 40
            try:
                pred_tokens, pred_scores, predicts = \
                    translate_batch_stepwise(model, batch, beam_size=one_step_args.beam_size,
                                            invalid_token_indices=invalid_token_indices,
                                            T=10, alpha_atom=-1, alpha_bond=-1,
                                            beta=0.5, percent_aa=40, percent_ab=percent_ab, k=3,
                                            use_template=one_step_args.use_template == 'True',
                                            factor_func=dataset.factor_func,
                                            reconstruct_func=dataset.reconstruct_smi,
                                            rc_path=one_step_args.intermediate_dir + '/rt2reaction_center.pk')
            except:
                return None

            original_beam_size = pred_tokens.shape[1]
            current_i = 0
            for batch_i, predict in enumerate(predicts):
                gt = ''.join(dataset.reconstruct_smi(tgt[:, batch_i], src=False))
                remain = original_beam_size
                beam_size = math.ceil(original_beam_size / len(predict))

                # normalized_reaction_center_score = np.array([pred[1] for pred in predict]) / 10
                hypo_i, hypo_scores_i = [], []
                for j, (rc, rc_score) in enumerate(predict):
                    # rc_score = normalized_reaction_center_score[j]

                    pred_token = pred_tokens[current_i + j]

                    sub_hypo_candidates, sub_score_candidates = [], []
                    for k in range(pred_token.shape[0]):
                        hypo_smiles_k = ''.join(dataset.reconstruct_smi(pred_token[k], src=False))
                        hypo_lens_k = len(smi_tokenizer(hypo_smiles_k))
                        hypo_scores_k = pred_scores[current_i + j][k].cpu().numpy() / hypo_lens_k + rc_score / 10

                        if hypo_smiles_k not in hypo_i:  # only select unique entries
                            sub_hypo_candidates.append(hypo_smiles_k)
                            sub_score_candidates.append(hypo_scores_k)

                    ordering = np.argsort(sub_score_candidates)[::-1]
                    sub_hypo_candidates = list(np.array(sub_hypo_candidates)[ordering])[:min(beam_size, remain)]
                    sub_score_candidates = list(np.array(sub_score_candidates)[ordering])[:min(beam_size, remain)]

                    hypo_i += sub_hypo_candidates
                    # print('The length of hypo_i {}'.format(len(hypo_i)),flush=True) 
                    hypo_scores_i += sub_score_candidates

                    remain -= beam_size

                current_i += len(predict)  ## point
                
                ground_truths += gt
                generations += hypo_i
                value_scores += hypo_scores_i



    print('The length of generations {}'.format(len(generations)),flush=True)  
    results = []
    class_nums = one_step_args.class_size
    # class_nums = len(dataset.props)//one_step_args.beam_size
    for m in range(len(dataset.props)//class_nums):
        props = np.zeros((class_nums*original_beam_size),dtype=float)
        for i in range(original_beam_size):
            props[i::original_beam_size] = dataset.props[m*class_nums:(m+1)*class_nums]
        value_scores_ = value_scores[m*class_nums*original_beam_size:(m+1)*class_nums*original_beam_size]
        value_scores_ = np.array(value_scores_)*props
        ordering = np.argsort(value_scores_)[::-1]
        value_scores_ = list(np.array(value_scores_)[ordering])
        generations_ = generations[m*class_nums*original_beam_size:(m+1)*class_nums*original_beam_size]
        generations_ = [generations_[i] for i  in ordering]

        # generations_ = generations_[:15]
        # value_scores_ = value_scores_[:15]

        unique_reason_smiles = []
        unique_reason_value_scores = []
        for i in range(len(generations_)):
            if generations_[i] not in unique_reason_smiles:
                reac = generations_[i].split('.')
                count = 0
                for smi in reac:
                    try:
                        if Chem.MolFromSmiles(smi) != None:
                            count += 1
                        if count == len(reac):
                            unique_reason_smiles.append(generations_[i])
                            unique_reason_value_scores.append(value_scores_[i])
                    except:
                        print(i)  # beam_research 确定了分子的合理性
                        # break
                
        # print(unique_reason_smiles)
        try:
            unique_reason_value_scores = torch.tensor(unique_reason_value_scores)
            unique_reason_value_scores = torch.softmax(unique_reason_value_scores, dim=0)
            unique_reason_value_scores = unique_reason_value_scores.tolist()
            # unique_reason_value_scores = unique_reason_value_scores/sum(unique_reason_value_scores)
        except:
            return None
        print('the unique length of smiles: {}'.format(len(unique_reason_smiles)),flush=True)
        result = pd.DataFrame({
            'reactants':unique_reason_smiles,
            'scores':unique_reason_value_scores,
            'templates':['temp']*len(unique_reason_value_scores)
        })
        print(result.to_string(),flush=True)
        results.append(result)
    return results

class OneStepRetro():
    def __init__(self,args=one_step_args,one_step_args=one_step_args):
        for key, value in vars(args).items():
            setattr(one_step_args, key, value)
        one_step_args.batch_size_val= one_step_args.class_size
        one_step_args.beam_size = one_step_args.total_beam_size//one_step_args.class_size
        one_step_args.select_smiles = ['CCO','C[C@H](c1ccccc1)N1C[C@]2(C(=O)OC(C)(C)C)C=CC[C@@H]2C1=S']
        
        self.args = one_step_args
        _, dataset = build_iterator(self.args, train=False)
    
        # Load Checkpoint Model:
        model = build_model(self.args, dataset.src_itos, dataset.tgt_itos)
        _, _, self.model = load_checkpoint(self.args, model)
        
    def run(self,smis_list=['CCO','C[C@H](c1ccccc1)N1C[C@]2(C(=O)OC(C)(C)C)C=CC[C@@H]2C1=S']):  
        args = self.args 
        args.select_smiles = smis_list
        iterator, dataset = build_iterator(args, train=False)
        return translate(iterator, self.model, dataset)


if __name__ == "__main__":
    
    model = OneStepRetro(args=one_step_args,one_step_args=one_step_args)
    path = "external/modelsmatter_modelzoo/external_models/Retroformer_as_one_step/data_random/test_10_smiles.csv"
    smiles_list = pd.read_csv(path,usecols=['prod_smiles']).values.flatten().tolist()[:2]
    
    results = model.run(smis_list=smiles_list)
    # results[0].to_csv('./result_beam_3_plus.csv')
