import torch
from torch import nn
import torch.optim as optim
from time import time
import pickle 
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pixel_by_pixel_CIFAR100_task import PixelByPixelCIFAR100
from pix_by_pix_CIFAR100_model import *

criterion = nn.CrossEntropyLoss()

seq_len = 1024
features = 3 
out_dim = 100
batch_size = 16
DEVICE = 0


configs = [ 
            {'hid_feature_size': 512, 'nblocks': 6},
            {'hid_feature_size': 256, 'nblocks': 6},
            {'hid_feature_size': 64, 'nblocks': 6},
            {'hid_feature_size': 512, 'nblocks': 4},
            {'hid_feature_size': 256, 'nblocks': 4},
            {'hid_feature_size': 64, 'nblocks': 4},
            {'hid_feature_size': 512, 'nblocks': 2},
            {'hid_feature_size': 256, 'nblocks': 2},
            {'hid_feature_size': 64, 'nblocks': 2},
        ]

    
noise = 0.0
# Create a DataLoader for the dataset
train_dataset = PixelByPixelCIFAR100(train=True, noise=noise)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = PixelByPixelCIFAR100(train=False, noise=noise)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for config in configs:
    nblocks = config['nblocks'] 
    hid_feature_size = config['hid_feature_size']
    model = TransformerEncoder( input_dim=features, out_dim=out_dim, seq_len=seq_len, d_model=hid_feature_size, nhead=8, num_encoder_layers=nblocks).to(DEVICE)

    n_params = 0
    for param in model.parameters():
        n_params +=  param.nelement()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    updates = 0
    mean_eval_acc = 0
    training_time = 0
    eval_every = 30
    max_training_time = 60*60*3
    nevals= 0

    results = {'updates': [], 
               'training_time' : [],
               'acc': [], 
               'loss' : [],
               'config' : config,
               'n_params' : n_params,
               'train_gpu_mem': 0,
               'inf_gpu_mem': 0,}

    while True:
        for src, tgt in train_loader:
            model.train()
            src = src.to(torch.float32).to(DEVICE)
            tgt = tgt.to(DEVICE)[:, -1]

            optimizer.zero_grad()
            torch.cuda.empty_cache() 
            start_time = time()
            output = model(src)[:, -1, :]
            task_loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))
            loss = task_loss
            loss.backward()
            mem = torch.cuda.memory_allocated(DEVICE)
            results['train_gpu_mem'] = mem
            optimizer.step()
            end_time = time()
            training_time+= end_time-start_time

            if updates%100==0: #if training_time >= eval_every*nevals:
                model.eval()
                with torch.no_grad(): 
                    acc = 0
                    loss = 0
                    seq_acc = np.zeros(seq_len)
                    for src, tgt in val_loader:
                        src = src.to(torch.float32).to(DEVICE)
                        tgt = tgt.to(DEVICE)[:, -1]
                        
                        optimizer.zero_grad()
                        torch.cuda.empty_cache()
                        output = model(src)[:, -1, :]
                        mem = torch.cuda.memory_allocated(DEVICE)
                        results['inf_gpu_mem'] = mem
                        loss += criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))
                        pred = torch.argmax(output, dim=-1)
                        acc += torch.mean((pred==tgt).to(torch.float32)).cpu().numpy()
                        #seq_acc += torch.mean((pred==tgt).to(torch.float32), dim=0).cpu().numpy()
                acc = acc/len(val_loader)
                loss = loss/len(val_loader)
                print(f'Time: [{training_time/60. : .1f} mins], Updates [{updates}], Loss: {loss.item():.4f}, Acc: {acc:.4f}')

                #plt.plot(seq_acc/len(val_loader))
                #plt.show()
                results['updates'].append(updates)
                results['training_time'].append(training_time)
                results['acc'].append(acc.item())
                results['loss'].append(loss.item())

                nevals+=1
                pickle.dump(results, open(f'results/CIFAR100_noise_{noise}_transformer_nblocks_{nblocks}_hid_feature_size_{hid_feature_size}_results.pkl', 'wb'))
                
                if updates>=40000:
                    break
                    
            updates+=1

        if updates>=40000: #if training_time>=max_training_time:
            pickle.dump(results, open(f'results/CIFAR100_noise_{noise}_transformer_nblocks_{nblocks}_hid_feature_size_{hid_feature_size}_results.pkl', 'wb'))
            break
