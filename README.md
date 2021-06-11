# CS60075-Team-2-Task-1
 SemEval Task 1 - Lexical Complexity Prediction (https://arxiv.org/abs/2106.02340)

## Pre-training models on the three corpora - BIBLE, BIONLP, EUROPARL

There were no off the shelf pre-trained RoBERTa Models available which were pre-trained on these corpora. So, we decided to pre-train on our own.

We trained on 2 Tesla V-100 GPUs.

### BIBLE

`pretrain_on_Bible` contains all the files as well as the corpus for pre-training. Look at `pretrain_on_Bible/pretrain_manuals.py` for training details, and  `pretrain_on_Bible/slurm-job-2-copy.sh` for the hyperparametrers and training conditions used.

The pretrained model can be found on HuggingFace (https://huggingface.co/abhi1nandy2/Bible-roberta-base).

### BIONLP

`pretrain_on_BioNLP` contains all the files as well as the corpus for pre-training. Look at `pretrain_on_BioNLP/pretrain_manuals.py` for training details, and  `pretrain_on_BioNLP/slurm-job-2-copy.sh` for the hyperparametrers and training conditions used.

The pretrained model can be found on HuggingFace (https://huggingface.co/abhi1nandy2/Craft-bionlp-roberta-base).

### EUROPARL

`pretrain_on_EuroParl` contains all the files as well as the corpus for pre-training. Look at `pretrain_on_EuroParl/pretrain_manuals.py` for training details, and  `pretrain_on_EuroParl/slurm-job-2-copy.sh` for the hyperparametrers and training conditions used.

The pretrained model can be found on HuggingFace (https://huggingface.co/abhi1nandy2/Europarl-roberta-base).

## Fine-tuning using tranformer based models, evaluation of their ensembles, and the baselines

All this can be found in `Main.ipynb`. Start srunning from top to bottom, except at one place - there is a `mode` variable in the `Baselines` section. Run once from that cell onwards for `mode = single` (sub-task 1), and then repeat the same for `mode = multi` (sub-task 2).

> `Main.ipynb` uses a python file `our_approach.py`, which contains the part for training the transformer models, and saving predictions on the test set.

## Results

- `our_approach_results` contains indivual predictions of 9 dine-tuned transformer models for the test set. The files of the form `test_sub_transf_{}_df.csv` are for sub-task l, and those with the format `test_sub_multi_transf_9_df.csv` are for sub-task 2. The 9 prediction csvs for each sub-task correspond to rhe following pre-trained models (huggingface model tags given here) in the same order -
"bert-base-uncased", "lukabor/europarl-mlm", "abhi1nandy2/Bible-roberta-base", "abhi1nandy2/Craft-bionlp-roberta-base", "ProsusAI/finbert", "allenai/scibert_scivocab_uncased", "xlm-roberta-base", "abhi1nandy2/Europarl-roberta-base", "emilyalsentzer/Bio_ClinicalBERT"

- `CodaLab_submissions` contains the `.csv` and the `.zip` formats of the submissions that our team made in CodaLab.
