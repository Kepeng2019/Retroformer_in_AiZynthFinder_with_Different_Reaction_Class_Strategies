import os
import sys
import argparse
import pandas as pd

from external.modelsmatter_modelzoo.ssbenchmark.ssmodels.base_model import Registries, SSMethod
from external.modelsmatter_modelzoo.ssbenchmark.utils import canonicalize_smiles, split_reactions


@Registries.ModelChoice.register(key="retroformer")
class model_retroformer(SSMethod):
    def __init__(self, module_path=''):
        module_path = '/home/kara_chen/workspace/'+\
                 'Retrosynthesis/modelsmatter-main_ver1/external/'+\
                'modelsmatter_modelzoo/external_models/Retroformer_as_one_step'
        self.model_name = "retroformer" # no limit to this name
        print(module_path)
        if module_path is not None:
            sys.path.insert(len(sys.path), os.path.abspath(module_path))
            sys.path.insert(len(sys.path), os.path.abspath((os.path.join(module_path, "retroformer"))))

    def preprocess(self):
        pass
    def process_input(self):
        pass

    def preprocess_store(self):
        pass

    def process_output(self):
        pass

    def model_setup(self, use_gpu=False, **kwargs):
        from external.modelsmatter_modelzoo.external_models.Retroformer_as_one_step.retroformer.one_step_model import OneStepRetro
        self.device = "cuda:0" if use_gpu else "cpu"
        # beam_size = kwargs.get('num_beams', 30) # config setting is None, equal as expanding_size
        # class_size = kwargs.get('class_size', 3)

        from argparse import Namespace
        kwargs_obj = Namespace(**kwargs)
        
        self.model = OneStepRetro(args=kwargs_obj)

    #  gpu_mode: True
    #   module_path: external/modelsmatter_modelzoo/external_models/Retroformer_KD_as_one_step/
    #   model_path: external/modelsmatter_modelzoo/external_models/Retroformer_KD_as_one_step/checkpoint_FC2_r1_paroutes_50k_typed/std_model_10.pt
    #   vocab_path: external/modelsmatter_modelzoo/external_models/Retroformer_KD_as_one_step/intermediate_FC2_r1_paroutes_50k_typed
    #   known_class: 'True'
    #   classes: 77
    #   shared_vocab: 'False'
    #   encoder_layers: 4
    #   decoder_layers: 4
    #   classifier_model: external/modelsmatter_modelzoo/external_models/Retroformer_KD_as_one_step/Classifier/models/USPTO_50k_optimizer_original_fp_FC2_r1_cutoff_0.6_fp.pth
    #   beam_size: 30
    #   class_size: 3
    #   real_beam_size: beam_size//class_size____not_a_parameter_but_a_illustration

#   python retroformer/translate_400k.py \
#     --batch_size_val 10 \
#     --shared_vocab 'False' \
#     --shared_encoder 'False' \
#     --data_dir './data_FC2_r1_paroutes_50k' \
#     --data_file 'FC2_r1_paroutes_50k_no_class' \
#     --intermediate_dir './intermediate_FC2_r1_paroutes_50k' \
#     --checkpoint_dir './checkpoint_FC2_r1_paroutes_50k' \
#     --checkpoint 'model_200000.pt' \
#     --known_class 'False' \
#     --beam_size 50 \
#     --stepwise False \
#     --use_template False \
#     --mode "$mode" \
#     --classes 0

    def _model_call(self, X):
      
        try:
            result = self.model.run(X)[0]
        except:
            result = None
        

        if result is not None and (len(result['scores']) > 0):
            smiles = result['reactants'].values.flatten().tolist()
            scores = result['scores'].values.flatten().tolist()
            prob = scores
        else:
            smiles,prob = [],[]

        return [smiles], [prob]
    
    def model_call(self, X):
        return self._model_call(X)