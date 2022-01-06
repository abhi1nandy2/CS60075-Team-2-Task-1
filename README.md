# CS60075-Team-2-Task-1
 SemEval Task 1 - Lexical Complexity Prediction (https://arxiv.org/abs/2106.02340)

 Check out its explanation in the form of a video presentation - https://youtu.be/bEkuzyWItfI

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

## Citation

If you use this code in your work, please add the following citation -
```
@inproceedings{nandy-etal-2021-cs60075,
    title = "cs60075{\_}team2 at {S}em{E}val-2021 Task 1 : Lexical Complexity Prediction using Transformer-based Language Models pre-trained on various text corpora",
    author = "Nandy, Abhilash  and
      Adak, Sayantan  and
      Halder, Tanurima  and
      Pokala, Sai Mahesh",
    booktitle = "Proceedings of the 15th International Workshop on Semantic Evaluation (SemEval-2021)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.semeval-1.87",
    doi = "10.18653/v1/2021.semeval-1.87",
    pages = "678--682",
    abstract = "The main contribution of this paper is to fine-tune transformer-based language models pre-trained on several text corpora, some being general (E.g., Wikipedia, BooksCorpus), some being the corpora from which the CompLex Dataset was extracted, and others being from other specific domains such as Finance, Law, etc. We perform ablation studies on selecting the transformer models and how their individual complexity scores are aggregated to get the resulting complexity scores. Our method achieves a best Pearson Correlation of 0.784 in sub-task 1 (single word) and 0.836 in sub-task 2 (multiple word expressions).",
}
```
