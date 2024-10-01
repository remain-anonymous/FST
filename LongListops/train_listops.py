import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
from time import time
from listops_model import *
import pickle
import os

# Tokenizer modified from from https://github.com/guy-dar/lra-benchmarks
def make_word_tokenizer(allowed_words, lowercase_input=False, allow_unk=False):
    # make distinct
    allowed_words = list(set(allowed_words))
    PAD_TOKEN = 0
    UNK_TOKEN =  -999999

    def _tokenizer(x_str, max_length):
        # note: x_str is not batched
        if lowercase_input:
            x_str = x_str.lower()
    
        x = x_str.split()
        x = x[:max_length]
        n = len(x)
        mask = ([1] * n) + ([0] * (max_length - n))
        ids = list(map(lambda c: allowed_words.index(c) + 1 if c in allowed_words else UNK_TOKEN, x)) + \
                  ([PAD_TOKEN] * (max_length - n))
        if not allow_unk:
            assert UNK_TOKEN not in ids, "unknown words are not allowed by this tokenizer"
        return {'input_ids': torch.LongTensor([ids]), 'attention_mask': torch.LongTensor([mask])}

    _tokenizer.vocab_size = len(allowed_words) + 1
    return _tokenizer

class ListOpsDataset:
    def __init__(self, MAX_LENGTH, split='train', data_number=0):

        # https://storage.googleapis.com/long-range-arena/lra_release.gz
        data_paths = {'train': f"listops_data/large_{data_number}_train.tsv",
                      'eval': "listops_data/basic_val.tsv"}
        self.data = pd.read_csv(data_paths[split], delimiter='\t')
        #self.tokenizer = make_word_tokenizer(list('0123456789') + [']', '(', ')', '[MIN', '[MAX', '[MED', '[SM'])
        self.tokenizer = make_word_tokenizer(list('0123456789') + [']', '[MIN', '[MAX', '[MED', '[SM'])
        self.max_length = MAX_LENGTH
        
    def __getitem__(self, i):
        data = self.data.iloc[i]
        source = data.Source.replace(')','').replace('(','')
        inputs = self.tokenizer(source, max_length=self.max_length) #return_tensors='pt', truncation=True, padding='max_length'
        target = data.Target
        return inputs, torch.LongTensor([target]), source
    
    def __len__(self):
        return len(self.data)

DEVICE = 0
MAX_LENGTH = 2000
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 128
features = 16
out_size = 10
hid_feature_size = 256
hid_seq_size = -1
nblocks = 6

model = Block_FST(features, MAX_LENGTH, out_size, MAX_LENGTH, hid_feature_size, nblocks, causal=False, dropout_rate=0.1).to(DEVICE)

if os.path.exists('FST_listops.pth'):
    model.load_state_dict(torch.load('FST_listops.pth', map_location='cpu'))
    
index_counts = Counter()

eval_every = 60*10

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

if os.path.exists('FST_optimizer.pth'):
    optimizer.load_state_dict(torch.load('FST_optimizer.pth', map_location='cpu'))

#lr_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
#if os.path.exists('FST_lr_sch.pth'):
#    lr_sch.load_state_dict(torch.load('FST_lr_sch.pth', map_location='cpu'))

criterion = nn.CrossEntropyLoss()
updates = 0
training_time = 0
n_evals = 0 
data_id = 0

n_params = 0
for param in model.parameters():
    n_params +=  param.nelement()
    
if os.path.exists('FST_listops.pkl'):
    results = pickle.load(open('FST_listops.pkl', 'rb'))
    training_time = results['training_time'][-1]
    updates= results['updates'][-1]
    n_evals = len(results['updates'])
else:
    results = {'updates': [], 
           'training_time' : [],
           'train_acc': [], 
           'train_loss' : [],
           'val_acc': [], 
           'val_loss' : [],
           'n_params' : n_params}

for epoch in range(1000):
    if not os.path.exists(f"listops-1000/large_{data_id}_train.tsv"):
        data_id = 0

    train_dataset = ListOpsDataset( MAX_LENGTH, split='train', data_number=data_id)
    evaL_dataset = ListOpsDataset( MAX_LENGTH, split='eval', data_number=data_id)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataloader = DataLoader(evaL_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    
    running_train_acc = 0
    running_train_loss = 0
    pre_eval_updates = 0
    for i, (src, tgt, source) in tqdm(enumerate(train_dataloader), total=len(train_dataloader) ):
        model.train()
        
        tgt = tgt.reshape(-1).to(DEVICE)
        src = src['input_ids'].transpose(1,2).squeeze(-1)
        src = nn.functional.one_hot(src, num_classes=features).to(torch.float32).to(DEVICE) 
        
        optimizer.zero_grad()
        start_time = time()
        output = model(src)[:,-1,:]
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        end_time = time()
        training_time+= end_time-start_time
        
        pred = torch.argmax(output, dim=-1)
        running_train_acc += torch.mean((pred==tgt).to(torch.float32)).cpu().numpy()
        running_train_loss += loss.item()
        
        updates+=1
        pre_eval_updates+=1
    
        if training_time >= eval_every*n_evals:
            model.eval()
            running_val_acc = 0
            running_val_loss = 0
            for i, (src, tgt, source) in enumerate(eval_dataloader):
                with torch.no_grad(): 
                    src = src['input_ids'].transpose(1,2).squeeze(-1)
                    src = nn.functional.one_hot(src, num_classes=features).to(torch.float32).to(DEVICE) 
                    tgt = tgt.reshape(-1).to(DEVICE)
                    output = model(src)[:,-1,:]
                    loss = criterion(output, tgt)
                    running_val_loss += loss.item()
                    pred = torch.argmax(output, dim=-1)
                    running_val_acc += torch.mean((pred==tgt).to(torch.float32)).cpu().numpy()
                
            print(f'Time: [{training_time/60. : .1f} mins], Updates [{updates}], T Loss: {running_train_loss/pre_eval_updates:.4f}, V Loss: {running_val_loss/len(eval_dataloader):.4f}, T Acc: {running_train_acc/pre_eval_updates:.4f},  V Acc: {running_val_acc/len(eval_dataloader):.4f}')

            #scheduler.step(val_loss)

            if results['val_loss']:
                if running_val_loss/len(eval_dataloader) < min(results['val_loss']):
                    torch.save(model.state_dict(), 'FST_listops_best_model.pth')
            
            results['updates'].append(updates)
            results['training_time'].append(training_time)
            results['val_acc'].append(running_val_acc/len(eval_dataloader))
            results['val_loss'].append(running_val_loss/len(eval_dataloader))
            results['train_acc'].append(running_train_acc/pre_eval_updates)
            results['train_loss'].append(running_train_loss/pre_eval_updates)   

            pre_eval_updates=0
            running_train_acc = 0
            running_train_loss = 0
            
            torch.save(model.state_dict(), 'FST_listops_model.pth')
            torch.save(optimizer.state_dict(), 'FST_optimizer.pth')
            #torch.save(lr_sch.state_dict(), 'FST_lr_sch.pth')
            pickle.dump(results, open(f'FST_listops.pkl', 'wb'))  
            n_evals+=1
    #lr_sch.step()
    data_id +=1
