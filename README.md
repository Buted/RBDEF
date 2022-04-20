# A Dual-Expert Framework for Event Argument Extraction

## Requirements

To run this repo, you need to install pytorch>=1.4, transformers, learn2learn.

## Run code

First,  set the corresponding files in`config` and `scripts`.

For `scripts`, select the mode, see Table 1 for more details. For `config`,  it's enough to adjust *model* most times.

### Train base model

```bash
set model of `ace.json` in `config` to "Main model"
set mode of `train.sh` in `scripts` to "train"
run the command "bash scripts/train.sh"
```

### Run RBDEF

Train base model first, then train routing, head expert and tail expert.

Train routing:

```bash
set model of `ace.json` in `config` to "Selector"
set mode of `train.sh` in `scripts` to "train"
run the command "bash scripts/train.sh"
```

Train head expert:

```bash
set model of `ace.json` in `config` to "Head"
set mode of `train.sh` in `scripts` to "train"
run the command "bash scripts/train.sh"
```

Train tail expert:

```bash
# meta-train first
set model of `ace.json` in `config` to "Meta"
set mode of `train.sh` in `scripts` to "meta"
run the command "bash scripts/train.sh"

# fine-tune
set model of `ace.json` in `config` to "FewRole"
set mode of `train.sh` in `scripts` to "train"
run the command "bash scripts/train.sh"
```

Evaluate RBDEF:

```bash
# no training, only evaluation
set model of `ace.json` in `config` to "Fuse"
set mode of `train.sh` in `scripts` to "threshold"
run the command "bash scripts/train.sh"
```

The result can be seen in `logs`.

For more details, see `code`.



Table 1

| mode       | sub_mode                             | argument                    | explanation                                                  |
| ---------- | ------------------------------------ | --------------------------- | ------------------------------------------------------------ |
| preprocess | -                                    | -                           | preprocess the data from sentence-level to entity-level      |
| train      | -                                    | load the saved model or not | training, set the configuration in `config`                  |
| evaluate   | -                                    | load the saved model or not | evaluation, set the configuration in `config`                |
| statistic  | -                                    | -                           | compute and save role2entity.json and role2event.json        |
| indicator  | filename of saved model              | -                           | test on devset, run this before *rank* mode                  |
| rank       | filename of the *indicator*'s result | -                           | rank roles by F1, run this after *indicator* mode            |
| important  | -                                    | -                           | set the flag whether or not the sample belong to $Q^{tar}_{b}$ |
| meta       | -                                    | -                           | meta-training, set the configuration in `config`             |
| threshold  | -                                    | -                           | evaluate the RBDEF                                           |
| save       | filename of saved model              | -                           | save the classifier from the saved model                     |
| fewshot    | -                                    | -                           | fine-tuning the initializations search by different meta learning algorithms, see sec 4.4 in paper |
| group      | -                                    | -                           | divide dataset into different groups for training Base+Fairness |
| parameter  | -                                    | -                           | compute the number of parameters of models                   |

