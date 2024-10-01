# Code designed to be run once for each epoch on a cluster with a 2h job time limit 

import torch
from torch.utils.data import Dataset, DataLoader
from wikitext_model import *
import pickle
from tqdm import tqdm
from time import time
import os

ckp_pth = 'wikitext103_checkpoints/'

class WikiTextDataset(Dataset):
    def __init__(self, tokens, seq_len=512):
        self.seq_len = seq_len
        
        self.tokens = tokens
        
        self.num_sequences = (len(self.tokens) // self.seq_len) - 1
        self.random_truncate = torch.randint(self.seq_len, (1,))

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        
        start_idx = self.random_truncate + idx * self.seq_len
        end_idx = start_idx + self.seq_len
        
        # Input sequence is the first 512 tokens, target is the last token
        input_seq = self.tokens[start_idx:end_idx - 1]
        target_token = self.tokens[end_idx - 1]
        
        # Convert to tensors
        input_seq = torch.tensor(input_seq, dtype=torch.long)
        target_token = torch.tensor(target_token, dtype=torch.long)
        
        return input_seq, target_token

train_file_path = 'wikitext103_data/wiki.train.tokens.berttok.pkl'
test_file_path = 'wikitext103_data/wiki.test.tokens.berttok.pkl'
valid_file_path = 'wikitext103_data/wiki.valid.tokens.berttok.pkl'

with open(train_file_path, 'rb') as f:
        train_tokens = pickle.load(f)
    
with open(test_file_path, 'rb') as f:
        test_tokens = pickle.load(f)
    
with open(valid_file_path, 'rb') as f:
        valid_tokens = pickle.load(f)

seq_len = 512
hid_feature_size = 512
nblocks = 6
batch_size = 128
vocab_size = 30522
DEVICE = 0

# Create the dataset and dataloader
train_dataset = WikiTextDataset(train_tokens, seq_len=seq_len+1)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = WikiTextDataset(test_tokens, seq_len=seq_len+1)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

valid_dataset = WikiTextDataset(valid_tokens, seq_len=seq_len+1)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

def train_epoch(model, dataloader, criterion, optimizer, device='cpu'):
    model.train()  
    total_loss = 0.0
    total_tokens = 0
    training_time = 0
    
    for input_seq, target in tqdm(dataloader, total=len(dataloader)):

        input_seq, target = input_seq.to(device), target.to(device)
        optimizer.zero_grad()
        
        start_time = time()

        outputs = model(input_seq)[:, -1, :]  

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        end_time = time()
        training_time+= end_time-start_time
        
        total_loss += loss.item() * input_seq.size(0) 
        total_tokens += input_seq.size(0)  
    
    avg_loss = total_loss / total_tokens

    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity, training_time


def evaluate_perplexity(model, dataloader, criterion, device='cpu'):
    model.eval()  
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for input_seq, target in tqdm(dataloader, total=len(dataloader)):

            input_seq, target = input_seq.to(device), target.to(device)


            outputs = model(input_seq)[:, -1, :]  

            loss = criterion(outputs, target)
            total_loss += loss.item() * input_seq.size(0)  
            total_tokens += input_seq.size(0) 
    
    avg_loss = total_loss / total_tokens

    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

model = Block_FST(hid_feature_size, seq_len, vocab_size, seq_len, hid_feature_size, nblocks, use_embedding=True, vocab_size=vocab_size, causal=False).to(DEVICE)
n_params = 0
for param in model.parameters():
    n_params +=  param.nelement()

if os.path.exists(ckp_pth+'FST_model.pth'):
    model.load_state_dict(torch.load(ckp_pth+'FST_model.pth', map_location='cpu'), strict=False)

if os.path.exists(ckp_pth+'FST_results.pkl'):
    results = pickle.load(open(ckp_pth+'FST_results.pkl', 'rb'))
else:
    None
    results = {'train_loss': [], 'val_loss': [],  'train_perplex': [], 'val_perplex': [], 'training_time': [], 'n_params': n_params}
    
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss() 

if os.path.exists(ckp_pth+'FST_optimizer.pth'):
    optimizer.load_state_dict(torch.load(ckp_pth+'FST_optimizer.pth', map_location='cpu'))


train_loss, train_perplex, training_time = train_epoch(model, train_dataloader, criterion, optimizer, device=DEVICE)
val_loss, val_perplex = evaluate_perplexity(model, test_dataloader, criterion, device=DEVICE)

if results['val_loss']:
    if val_loss < min(results['val_loss']):
        torch.save(model.state_dict(), ckp_pth+'FST_best_model.pth')

results['train_loss'].append(train_loss)
results['val_loss'].append(val_loss)
results['train_perplex'].append(train_perplex)
results['val_perplex'].append(val_perplex)
results['training_time'].append(training_time)

pickle.dump(results, open(ckp_pth+'FST_results.pkl', 'wb'))
torch.save(model.state_dict(), ckp_pth+'FST_model.pth')
torch.save(optimizer.state_dict(), ckp_pth+'FST_optimizer.pth')
