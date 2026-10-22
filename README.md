# Retroformer_in_AiZynthFinder_with_Different_Reaction_Class_Strategies

Our package is built upon [modelsmatter](https://github.com/AlanHassen/modelsmatter), which extends [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder) to integrate Retroformer as the single-step retrosynthesis model, as described in our paper.

## Overview
modelsmatter_with_retroformer
├─aizynthfinder
│  ├─...
├─configs
├─data
│  └─...
├─download
├─external
│  └─modelsmatter_modelzoo
│      ├─external_models
│      │  ├─Chemformer
│      │  │  ├─example_scripts
│      │  │  │  └─finetune_regression
│      │  │  ├─molbart
│      │  │  │  ├─data
│      │  │  │  │  └─__pycache__
│      │  │  │  ├─models
│      │  │  │  │  └─__pycache__
│      │  │  │  └─__pycache__
│      │  │  ├─molbart.egg-info
│      │  │  └─test
│      │  ├─modelsmatter_localretro_hpc
│      │  ├─modelsmatter_mhn_react_hpc
│      │  ├─molbart.egg-info
│      │  └─Retroformer_as_one_step
│      │      ├─.vscode
│      │      ├─checkpoint_FC2_r1_paroutes_400k_typed
│      │      ├─checkpoint_FC2_r1_paroutes_400k_untyped
│      │      ├─checkpoint_FC2_r1_paroutes_50k_typed
│      │      ├─checkpoint_FC2_r1_paroutes_50k_untyped
│      │      ├─Classifier
│      │      ├─data_random
│      │      ├─intermediate_FC2_r1_paroutes_400k
│      │      ├─intermediate_FC2_r1_paroutes_50k
│      │      ├─log
│      │      ├─retroformer
│      │      ├─Retroformer_as_one_step.egg-info
│      │      └─__pycache__
│      └─ssbenchmark
├─one_step_diversity
│  ├─models_by_reactions
│  ├─one_step_diversity.egg-info
│  ├─utils
├─other_depends
└─z_scripts






## Installation Procedure

To install the package, follow the sequential steps below:

```bash
# Clone this repository and all submodules
git clone --recursive https://github.com/AlanHassen/modelsmatter

# Create and initialize a conda environment
conda env create -f environments/environment.yml -n ssbenchmark
conda activate ssbenchmark

# Transition to the SSBenchmark directory
cd external/modelsmatter_modelzoo/

# First the installation of the ModelZoo
poetry install

# Navigate back to the models matter directory
cd ../..

# Finalize the installation process
poetry install

# Subsequently, install the appropriate single-step models or the necessary libraries.
```

## Usage Guidelines

Adaptations to AiZynthFinder config are necessary to accommodate the different single-step models (examples provided in `config/`). The configurable settings include:

- **gpu_mode**: Enable GPU mode for accelerated inference.
  
- **module_path**: The path of the single-step retrosynthesis model repository.
  
- **model_path**: The location of the trained model.

- Additional parameters can be set, such as specifying the vocabulary for Chemformer via `vocab_path`.

Utilize the extended functionalities of AiZynthFinder through Models Matter for a comprehensive synthesis prediction experience.

All scripts for retrosynthesis planning are located in z_scripts.

## Datasets & Models

Datasets and models that are not proprietary are available in the [figshare repository](https://figshare.com/s/2eab4132b322229c1efc).

- ZINC molecules file (paroutes_stock.hdf5)​ is included in the download. The file path within the download is download\paroutes_stock.hdf5.
- 
- Eight checkpoint files of Retroformer​ are located in external/modelsmatter_modelzoo/external_models/Retroformer_as_one_step/. The file path within the download is Retroformer_as_one_step/*.
- 
- The checkpoint file of the product-based class prediction model​ is located in external/modelsmatter_modelzoo/external_models/Retroformer_as_one_step/Classifier/models/. The file path within the download is Classifier/models/*.
- 
- The data and checkpoint file of the reaction-based class prediction model​ is located in one_step_diversity/models_by_reactions. The file path within the download is models_by_reactions/*.
- 
- The data and checkpoints of Chemformer for forward reaction prediction​ are also included. The file path within the download is chemformer/*.



## others

For success route diversity, the procedure is：
```
z_process1_for_success_paths_with_diversity.py
z_process1_for_success_paths_with_diversity.ipynb
```

For all trajectory diversity, the procedure is:

```
z_process1_for_all_tragictorics_with_diversity.py
z_process1_for_all_tragictorics_with_diversity.ipynb
```

Note：the detailed training process of Retroformer is available in the [GitHub repository](https://github.com/Kepeng2019/Retroformer.git)




## References
1. Torren-Peraire P, Hassen AK, Genheden S, et al (2024) Models Matter: the impact of single-step retrosynthesis on synthesis planning. Digital Discovery. https://doi.org/10.1039/D3DD00213G
2. Hassen AK, Torren-Peraire P, Genheden S, et al (2022) Mind the Retrosynthesis Gap: Bridging the Divide between Single-step and Multi-step Retrosynthesis Prediction. arXiv:2212.11809. https://doi.org/10.48550/arXiv.2212.11809
3. Irwin R, Dimitriadis S, He J, et al (2022) Chemformer: a pre-trained transformer for computational chemistry. Mach Learn Sci Technol 3:015022. https://doi.org/10.1088/2632-2153/ac3ffb
4. Wan Y, Hsieh C-Y, Liao B, et al (2022) Retroformer: Pushing the Limits of End-to-end Retrosynthesis Transformer. In: Proceedings of the 39th International Conference on Machine Learning (ICML 2022), PMLR 162:22475–22490. https://proceedings.mlr.press/v162/wan22a.html
5. Liu T, Cao Z, Huang Y, et al (2023) SynCluster: Reaction Type Clustering and Recommendation Framework for Synthesis Planning. JACS Au 3(12):3446–3461. https://doi.org/10.1021/jacsau.3c00607
6. Thakkar A, Kogej T, Reymond J-L, et al (2019) Datasets and their influence on the development of computer assisted synthesis planning tools in the pharmaceutical domain. Chem Sci. https://doi.org/10.1039/C9SC04944D
7. Genheden S, Thakkar A, Chadimova V, et al (2020) AiZynthFinder: a fast, robust and flexible open-source software for retrosynthetic planning. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.12465371.v1

