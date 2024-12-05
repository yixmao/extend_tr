# Extended Topological Regression

## Introduction
This is the Python package of an extended version of Topological Regression (TR), a similarity-based regression framework, that is statistically grounded, computationally fast, and interpretable. This package offers flexible options for descriptor calculation, distance calculation, anchor point selection, and model configuration, allowing users to fine-tune the method to their specific research needs. For more information, please see
1. Zhang, Ruibo, et al. "Topological regression as an interpretable and efficient tool for quantitative structure-activity relationship modeling." Nature Communications 15.1 (2024): 5072. 

## Installation
**Step 1: Creat a conda environment (optional)**
```bash
conda create --name extendtr python=3.8
conda activate extendtr
```
**Step 2: Install the extendtr package**
Clone the github repo:
```bash
git clone git@github.com:yixmao/extend_tr.git
cd extend_tr
```
Install the package
```bash
pip install .
```
**Step 3: Verify installation**
```bash
python tr_examples.py
```
Expected output:
```
----------------- Running different descriptors -----------------
Performance of ensemble predictions:
Spearman: 0.8048832322935336
R2: 0.8187727046698461
RMSE: 0.7894710288568683
NRMSE: 0.4257079930306147
Performance of stacking predictions:
Spearman: 0.7736080001537748
R2: 0.8180014941571891
RMSE: 0.7911490376625655
NRMSE: 0.4266128289712007
----------------- Running different anchor selections -----------------
Performance of ensemble predictions:
Spearman: 0.8280874367843225
R2: 0.869484364577384
RMSE: 0.6699704746442829
NRMSE: 0.3612694775684988
Performance of stacking predictions:
Spearman: 0.7875305228482481
R2: 0.8355063934636877
RMSE: 0.7521403782978013
NRMSE: 0.405578113975979
----------------- Running different distance calculations -----------------
Performance of ensemble predictions:
Spearman: 0.825262577107183
R2: 0.8559124650619488
RMSE: 0.7039431684702607
NRMSE: 0.3795886391056128
Performance of stacking predictions:
Spearman: 0.8621875286012207
R2: 0.8960827625479393
RMSE: 0.5978169015645128
NRMSE: 0.32236196650979265
----------------- Running different models ----------------------
Performance of ensemble predictions:
Spearman: 0.825262577107183
R2: 0.8559124650619488
RMSE: 0.7039431684702607
NRMSE: 0.3795886391056128
Performance of stacking predictions:
Spearman: 0.8621875286012207
R2: 0.8960827625479393
RMSE: 0.5978169015645128
NRMSE: 0.32236196650979265
----------------- All configurations run successfully! -----------------
```

## Example usage
Detailed example usage of different TR configurations can be found in `tr_examples.ipynb`. Here, we go through a simply example that runs ensemble TR on the CHEMBL dataset 278.
**Step 1: Prepare the data**
To use TR functions, first we need to prepare the data, including the descriptors, targets, train and test indices and validation indices (optional). Note that the descriptors and targets need to be ```pdDataFrame```, and the indices need to be ```list```.
```python
import pandas as pd
# load the descriptor - the indices of desc will be used later
desc = pd.read_parquet(f'./SampleDatasets/CHEMBL278/data_ECFP4.parquet', engine='fastparquet').astype('bool')
# load targets
data = pd.read_csv(f'./SampleDatasets/CHEMBL278/data_cp.csv', index_col=0)
target = data["pChEMBL Value"]
```
As a sanity check, make sure that ```desc``` and ```target``` have the same indices.
```python
# make sure that the indices of desc and target match
desc = desc.loc[target.index]
target = target.loc[desc.index]
```
Then, we define the indices for training, test and validation (option) samples.
```python
import json
from sklearn.model_selection import train_test_split
# load indicies for scaffold split
with open(f'./SampleDatasets/CHEMBL278/scaffold_split_index.json', 'r') as f:
    index = json.load(f)  
train_idx = index['train_idx']
test_idx = index['test_idx']
# make sure that train and test indices are included in target.index
train_idx = [idx for idx in train_idx if idx in target.index]
test_idx = [idx for idx in test_idx if idx in target.index]

##### alternatively, you can ranomly split train and test idx
# dataset_idx = target.index.tolist()
# train_idx, test_idx = train_test_split(dataset_idx, test_size=0.2, random_state=args.seed)

# set validation index if necessary
val_set = 0.2 # a fraction number to use [val_set] percent samples from the train set as val set, or None for no validation
if val_set is not None: # if we want to test on the validation set
    train_idx, val_idx = train_test_split(train_idx, test_size=val_set, random_state=2021)
else: # no validation
    val_idx = None
```

**Step 2: Model and predict**
Set the arguments and random seed. You can use ```args = TopoRegArgs()``` to receive command-line input.
```python
from extendtr.utils.args import TopoRegArgs
from extendtr.utils.utils import set_seed
# get the args
args = TopoRegArgs('-ensemble 1') # ensemble TR
# set random seed
set_seed(args.seed)
```
Train the TR model(s) and get the predictions using ```TopoReg```. ```mdl``` will be a ```list``` of models if ```ensemble``` is enabled. Note that ```pred_val``` will be ```None``` if ```val_idx``` is ```None```.
```python
from extendtr.TR.topoReg import TopoReg
# train and get the predictions
mdl, pred_test, pred_val, train_time, test_time = TopoReg(desc, target, train_idx, test_idx, val_idx, args)
```
**Step 3: Evaluate the predictions**
We provide a simple function that can calculate Spearman's correlation, R2, root mean square error (RMSE) and normalized RMSE (NRMSE)
```python
from extendtr.utils.utils import metric_calc
# evaluate the resuls
scorr, r2, rmse, nrmse = metric_calc(pred_test, target.loc[test_idx], True)
```
### Parameters
Table below shows the parameters and their possible values and descriptions.
| Parameter Name         | Possible Values/Range                  | Default Value | Description                                                                 |
|------------------------|-----------------------------------------|---------------|-----------------------------------------------------------------------------|
| `-anchorselection`     | `'random'`, `'maximin'`, `'maximin_density'` | `'random'`    | Anchor selection strategy.                                                  |
| `-desc_norm`           | `True`, `False`                        | `False`       | Normalize the descriptors or not.                                           |
| `-distance`            | `'jaccard'`, `'tversky'`, `'euclidean'`, `'cosine'` | `'jaccard'`  | Distance metric for calculations.                                           |
| `-model`               | `'LR'`, `'LR_L1'`, `'RF'`, `'ANN'`     | `'LR'`        | Model type for training.                                                    |
| `-anchor_percentage`   | `[0, 1]` (float)                       | `0.5`         | Percentage of anchors for training.                                         |
| `-max_num_anchors`     | Integer                                | `2000`        | Maximum number of anchor points.                                            |
| `-ensemble`            | `True`, `False`                        | `False`       | Enable ensemble TR.                                |
| `-mean_anchor_percentage` | `[0, 1]` (float)                    | `0.6`         | Mean anchor percentage for ensemble  TR.                                     |
| `-std_anchor_percentage`  | `[0, 1]` (float)                    | `0.2`         | Standard deviation of anchor percentage for ensemble  TR.                    |
| `-min_anchor_percentage`  | `[0, 1]` (float)                    | `0.3`         | Minimum anchor percentage for ensemble  TR.                                  |
| `-max_anchor_percentage`  | `[0, 1]` (float)                    | `0.9`         | Maximum anchor percentage for ensemble  TR.                                  |
| `-num_TR_models`       | Integer                                | `15`          | Number of TR models included in the ensemble TR.                               |
| `-seed`                | Integer                                | `2021`        | Random seed for reproducibility.                                            |
| `-rbf_gamma`           | Float                                  | `0.5`         | Gamma parameter for the RBF function.                                       |
| `-verbose`             | `True`, `False`                        | `False`       | Report the metrics or not.                                                  |
| `-check_duplicates`    | `True`, `False`                        | `False`       | Check for duplicate samples in the dataset.                                 |
| `-tversky_alpha`       | `[0, 1]` (float)                       | `0.5`         | Alpha parameter for Tversky distance.                                       |
| `-tversky_beta`        | `[0, 1]` (float)                       | `0.5`         | Beta parameter for Tversky distance.                                        |
| `-refine_anchors_lasso`| `True`, `False`                        | `False`       | Enable L1-norm regularization to refine the anchors.                        |
| `-lasso_alpha`         | Float                                  | `0.05`        | Alpha parameter for Lasso regularization.                                   |
| `-lasso_thres`         | Float                                  | `1e-6`        | Threshold for Lasso coefficient filtering.                                  |
| `-weight_density`      | `[0, 1]` (float)                       | `0.5`         | Weight for the `'maximin_density'` anchor selection approach.                   |
| `-ann_cp_dir`          | String (directory path)                | `'./results/ann_cp/'` | Directory to save the ANN checkpoint.                                      |
| `-ann_act`             | `'tanh'`, `'relu'`, `'sigmoid'`, `'linear'` | `'tanh'`    | Activation function for ANN.                                                |
| `-ann_lr`              | Float                                  | `0.001`       | Learning rate for ANN training.                                             |
| `-ann_num_layers`      | Integer                                | `1`           | Number of hidden layers in the ANN.                                                |
| `-ann_epochs`          | Integer                                | `50`          | Number of training epochs for ANN.                                          |
| `-ann_batch_size`      | Integer                                | `256`         | Batch size for ANN training.                                                |
| `-ann_batch_norm`      | `True`, `False`                        | `True`        | Enable batch normalization in ANN.                                          |
| `-ann_init_wts`        | `True`, `False`                        | `True`        | Enable weight initialization in ANN.                                        |
| `-ann_early_stop`      | `True`, `False`                        | `True`        | Enable early stopping for ANN.                                              |
| `-ann_patience`        | Integer                                | `3`           | Number of steps to wait before stopping ANN training.                       |
| `-ann_min_delta`       | Float                                  | `1e-3`        | Minimum change in NRMSE to qualify as an improvement for early stopping.    |
