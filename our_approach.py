import os
import numpy as np
import pandas as pd
import argparse
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
import copy
from transformers import BertModel, RobertaModel, BertTokenizer, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, random_split, DataLoader, IterableDataset, ConcatDataset
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score 
from tqdm import tqdm
import torch.nn as nn
# import demoji 
from scipy.stats import pearsonr 
from tqdm import tqdm
# demoji.download_codes() 

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required = True, type=str, help='whether simgle or multi')
args = parser.parse_args()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Model

class transformer_reg(torch.nn.Module):
    def __init__(self, model_name):
        super(transformer_reg, self).__init__()
        self.embeddings = AutoModel.from_pretrained(model_name, output_hidden_states = True)
        self.final = nn.Linear(self.embeddings.config.hidden_size, 1, bias = True)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_mask, token_type_ids):
        embed = self.embeddings(x,x_mask, token_type_ids = token_type_ids)[1]
        # print(out.shape)
        # break
        # mean_pooling = torch.mean(out, 1)
        # print(mean_pooling.shape)
        # max_pooling, _ = torch.max(out, 1)
        # print(max_pooling.shape)

        # embed = torch.cat((mean_pooling, max_pooling), 1)
        # print(embed.shape)
        embed = self.dropout(embed)
        # print(embed.shape)
        embed = self.final(embed)
        # print(embed.shape)
        y_pred = self.sigmoid(embed)
        return y_pred.view(y_pred.shape[0])


import pandas as pd
import csv

# set the mode as `single` or `multi` depending on
# whether somplexity is to be found for a 
# single word (subtask 1) or multi-word expression (task 2)

mode = args.mode

df1 = pd.read_csv('/content/train/lcp_{}_train.tsv'.format(mode), delimiter = "\t", quoting = csv.QUOTE_NONE, encoding = 'utf-8', keep_default_na=False)
df2 = pd.read_csv('/content/train/lcp_{}_trial.tsv'.format(mode), delimiter = "\t", quoting = csv.QUOTE_NONE, encoding = 'utf-8', keep_default_na=False)
df3 = pd.read_csv('/content/lcp_{}_test.tsv'.format(mode), delimiter = "\t", quoting = csv.QUOTE_NONE, encoding = 'utf-8', keep_default_na=False)

def get_dataset(df, tokenizer):
    sentences_1 = df.sentence.values
    sentences_2 = df.token.values
    labels = df.complexity.values
    input_ids = []
    attention_masks = []
    token_type_ids = []

    # For every sentence...
    for sent_idx in tqdm(range(len(sentences_1))):
        # inp = sentences_1[sent_idx] + '[SEP]'+ sentences_2[sent_idx]
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer(
                            sentences_1[sent_idx],                      # Input to encode.
                            sentences_2[sent_idx],
                            add_special_tokens = True, # To Add '[CLS]' and '[SEP]'
                            max_length = 256,           # tokenizer.model_max_length
                            truncation = True,
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_token_type_ids = True,
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])

        token_type_ids.append(encoded_dict['token_type_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences_1[0], sentences_2[0])
    # print('Token IDs:', input_ids[0])

    print(input_ids.shape, token_type_ids.shape, attention_masks.shape, labels.shape)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks,  token_type_ids, labels)
    return dataset



def evaluate(test_dataloader, model, is_test=False):
    model.eval()
    total_eval_accuracy=0
    y_preds = np.array([])
    y_test = np.array([])
    total_loss = 0
    criterion = nn.MSELoss()
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_token_type_ids = batch[2].to(device)
        b_labels = batch[3].float().to(device)
        with torch.no_grad():        
            ypred = model(b_input_ids, b_input_mask, b_token_type_ids)

        ypred = ypred.to('cpu').numpy()
        b_labels = b_labels.to('cpu').numpy()

        y_preds = np.hstack((y_preds, ypred))
        y_test = np.hstack((y_test, b_labels))

    print(y_preds.shape)
    print(y_test.shape)

    loss = np.mean((y_preds-y_test)**2)
    corr, _ = pearsonr(y_preds, y_test)
    if is_test:
        return y_preds, corr
    return loss, y_preds, y_test, corr
 
def train(training_dataloader, validation_dataloader, model, filename, epochs = 4):
    total_steps = len(training_dataloader) * epochs
    optimizer = AdamW(model.parameters(), lr=2e-5, eps = 1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    criterion = nn.MSELoss()
    best_model = copy.deepcopy(model)
    best_corr = 0
    # cur_epoch, best_corr = load_metrics(filename, model, optimizer)
    for epoch_i in tqdm(range(0, epochs)):
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(training_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_token_type_ids = batch[2].to(device)
            b_labels = batch[3].float().to(device)
            
            outputs = model(b_input_ids, b_input_mask, b_token_type_ids)
            # print(outputs, b_labels)
            loss = criterion(outputs, b_labels)
 
            if step%50 == 0:
                print(loss)
 
            total_train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
 
        print()
 
        print(f'Total Train Loss = {total_train_loss}')
        print('#############    Validation Set Stats')
        l2_loss, _, _ , corr = evaluate(validation_dataloader, model)
        print("  L2 loss: {0:.2f}".format(l2_loss))
        print("  Pearson Correlation: {0:.2f}".format(corr))
 
        if corr > best_corr:
            best_corr = corr
            best_model = model 
    print(best_corr)
    return best_model

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModel

models_ = ["bert-base-uncased", "lukabor/europarl-mlm", "abhi1nandy2/Bible-roberta-base", "abhi1nandy2/Craft-bionlp-roberta-base", "ProsusAI/finbert", "allenai/scibert_scivocab_uncased", "xlm-roberta-base", "abhi1nandy2/Europarl-roberta-base", "emilyalsentzer/Bio_ClinicalBERT"]

batch_size =  32

for model_idx_, model_name in tqdm(enumerate(models_)):

    if model_idx_ > 6:
        continue

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = get_dataset(df1, tokenizer)
    val_dataset = get_dataset(df2, tokenizer)
    test_dataset = get_dataset(df3, tokenizer)

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size, # Trains with this batch size.
                num_workers=8
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size, # Evaluate with this batch size.
                num_workers=8
            )

    test_dataloader = DataLoader(
                test_dataset, # The validation samples.
                sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                batch_size = batch_size, # Evaluate with this batch size.
                num_workers=8
            )
    
    model = transformer_reg(model_name).to(device)
    print(device)

    best_model = train(train_dataloader, validation_dataloader,model, model_name)

    _, corr = evaluate(test_dataloader, best_model, is_test=True)
    print(corr)

    test_sub_df = pd.DataFrame()
    test_sub_df['ID'] = df3['id']
    test_sub_df['SCORE'] = _

    results_dir = "our_approach_results"

    if os.path.exists(results_dir) == False:
    	os.mkdir(results_dir)

    if mode == 'single':
        test_sub_df.to_csv("{}/test_sub_transf_{}_df.csv".format(results_dir, model_idx_ + 1), index = False, header=False)
    else:
        test_sub_df.to_csv("{}/test_sub_multi_transf_{}_df.csv".format(results_dir, model_idx_ + 1), index = False, header=False)        