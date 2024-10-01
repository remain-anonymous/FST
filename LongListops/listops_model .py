import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
DTYPE = torch.complex64

class MaskLinearWeights(nn.Module):
    # FOR CAUSAL MODEL, NOT USED IN PAPER
    def __init__(self, linear_layer, causal=True):
        super(MaskLinearWeights, self).__init__()
        self.linear_layer = linear_layer
        self.causal = causal

    def forward(self, x):
        if self.causal:
            with torch.no_grad():
                mask = torch.tril(torch.ones_like(self.linear_layer.weight), diagonal=0)
                self.linear_layer.weight.mul_(mask)
        return self.linear_layer(x)
    
    
class LRU(nn.Module):
    def __init__(self, input_size, hid_rnn_size, mask_inputs=False):
        super(LRU, self).__init__()

        self.input_size = input_size
        self.hid_rnn_size = hid_rnn_size

        B_init_real = torch.randn(input_size, hid_rnn_size)
        B_init_img = torch.randn(input_size, hid_rnn_size)        
    
        self.B_real = nn.Parameter(B_init_real)
        self.B_img = nn.Parameter(B_init_img)   
        
        if mask_inputs:
            # FOR CAUSAL MODEL, NOT USED IN PAPER
            self.B_mask = torch.triu(torch.ones_like(self.B_real), diagonal=0)
        else:
            self.B_mask = torch.Tensor([1.])
        
        nu_init = torch.log(-torch.log(torch.distributions.uniform.Uniform(0.999, 1).sample((hid_rnn_size,))))
        self.nu = nn.Parameter(nu_init)

        theta_init = torch.distributions.uniform.Uniform(0, math.pi / 10).sample((hid_rnn_size,))
        self.theta = nn.Parameter(theta_init)

        delta_init = torch.sqrt(1 - torch.exp(-torch.exp(nu_init)))
        self.delta = nn.Parameter(delta_init)

        h0_init_real = torch.randn(hid_rnn_size)
        h0_init_img = torch.randn(hid_rnn_size)
        
        self.h0_img = nn.Parameter(h0_init_real)
        self.h0_real = nn.Parameter(h0_init_img)
    
    def forward_recurrent(self, x):
        
        lam = torch.exp(-torch.exp(self.nu) + 1j * self.theta)
        
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        hidden_size = self.hid_rnn_size

        h0_real = self.h0_real.squeeze(0).repeat(batch_size, 1)
        h0_img = self.h0_img.squeeze(0).repeat(batch_size, 1)
        
        # Initial hidden state
        h = torch.zeros(batch_size, seq_len+1, hidden_size).to(torch.complex64).to(x.device)
        Bx = torch.zeros(batch_size, seq_len+1, hidden_size).to(torch.complex64).to(x.device)  
        
        
        B_real=self.B_real*(self.B_mask.to(x.device))
        B_img=self.B_img*(self.B_mask.to(x.device))
            
        Bx[:,1:,:] = torch.einsum('fh, bsf -> bsh', B_real, x)*self.delta + 1j*0
        Bx[:,1:,:] += 1j*torch.einsum('fh, bsf -> bsh', B_img, x)*self.delta
        h[:, 0, :] =  h0_real + 1j*0
        h[:, 0, :] +=  0 + 1j*h0_img
        
        for s in range(1, seq_len+1):
            h[:, s, :] = lam * h[:, s - 1, :] + Bx[:, s, :]
        
        return h[:,1:seq_len+1,:].real


    def forward_fullvec(self, x):

        seq_len = x.shape[1]
        batch_size = x.shape[0]
        hidden_size = self.hid_rnn_size

        lam = torch.exp(-torch.exp(self.nu) + 1j * self.theta)

        seq_len = x.size(1)
        
        B_real=self.B_real*(self.B_mask.to(x.device))
        B_img=self.B_img*(self.B_mask.to(x.device))
            
        Bx_real = torch.einsum('ij,bti->btj', B_real, x)*self.delta
        Bx_img = torch.einsum('ij,bti->btj', B_img, x)*self.delta
        Bx = Bx_real + 1j*Bx_img
        
        h0_real = self.h0_real.squeeze(0).repeat(batch_size, 1)
        h0_img = self.h0_img.squeeze(0).repeat(batch_size, 1)
        h0 = h0_real + 1j*h0_img
        
        lam_expanded = lam.unsqueeze(1).unsqueeze(2)
        
        # Compute the powers of lam
        time_diffs = torch.abs(torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)).to(x.device)
        
        lampow = torch.cumprod(lam.unsqueeze(0).expand(seq_len,-1), dim=0)
        h0_contrib = torch.einsum('sh,bh->bsh', lampow, h0)
        lampow = torch.cat([torch.ones(1,hidden_size).to(x.device), lampow[:-1, :]], dim = 0).T
        lampow = torch.index_select(lampow, 1, time_diffs.view(-1).to(x.device)).view(hidden_size, seq_len, seq_len)
        lampow = torch.triu(lampow)
        
        outputs = torch.einsum('hst,bsh->bth', lampow, Bx)
        outputs += h0_contrib
        
        return outputs.real

    def forward_last(self, x, T=None):

        seq_len = x.shape[1]
        batch_size = x.shape[0]
        hidden_size = self.hid_rnn_size

        if T == None: T = seq_len
        
        lam = torch.exp(-torch.exp(self.nu) + 1j * self.theta)

        seq_len = x.size(1)
        
        B_real=self.B_real*(self.B_mask.to(x.device))
        B_img=self.B_img*(self.B_mask.to(x.device))
            
        Bx_real = torch.einsum('ij,bti->btj', B_real, x[:,:T,:])*self.delta
        Bx_img = torch.einsum('ij,bti->btj', B_img, x[:,:T,:])*self.delta
        Bx = Bx_real + 1j*Bx_img
        
        h0_real = self.h0_real.squeeze(0).repeat(batch_size, 1)
        h0_img = self.h0_img.squeeze(0).repeat(batch_size, 1)
        h0 = h0_real + 1j*h0_img
        
        lam_expanded = lam.unsqueeze(1).unsqueeze(2)
        
        # Compute the powers of lam
        time_diffs = torch.arange(T).flip(0)
        
        lampow = torch.cumprod(lam.unsqueeze(0).expand(T,-1), dim=0)
        h0_contrib = h0*lampow[-1,:]
        lampow = torch.cat([torch.ones(1,hidden_size).to(x.device), lampow[:-1, :]], dim = 0).T
        lampow = torch.index_select(lampow, 1, time_diffs.view(-1).to(x.device)).view(hidden_size, T)
        
        outputs = torch.einsum('hs,bsh->bh', lampow, Bx)
        outputs += h0_contrib
        
        return outputs.real
    
    def forward(self, x):
    
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        hidden_size = self.B_real.shape[0]

        h0_real = self.h0_real.squeeze(0).repeat(batch_size, 1)
        h0_img = self.h0_img.squeeze(0).repeat(batch_size, 1)
        h0 = h0_real + 1j*h0_img

        lam = torch.exp(-torch.exp(self.nu) + 1j * self.theta)
    
        lam = lam.unsqueeze(0)
        h0_contrib = torch.cumprod(lam.unsqueeze(1).repeat(1, seq_len, 1), dim=1).to(x.device) * h0.unsqueeze(1)
        
        oringinal_seq_length = x.shape[1]
        
        # 1. Sequence padding
        L_log2 = int(math.ceil(math.log2(x.shape[1])))
        x = F.pad(x, (0, 0, 2**L_log2 - x.shape[1], 0, 0, 0))
        
        
        B_real=self.B_real*(self.B_mask.to(x.device))
        B_img=self.B_img*(self.B_mask.to(x.device))
            
        # 2. Recursive split
        h_real = torch.matmul(x, B_real)*self.delta
        h_img = torch.matmul(x, B_img)*self.delta
        h = h_real + 1j*h_img
        
        N, L, H = h.shape 
        
        for i in range(1, L_log2 + 1):
            l = 2 ** i  # length of subsequences
            h = h.reshape(N * L // l, l, H)
    
            # 3. Parallel forward pass
            h1, h2 = h[:, :l // 2], h[:, l // 2:]
            if i > 1:  # compute [lam, lam^2, ..., lam^l]
                lam = torch.cat((lam, lam * lam[-1]), 0)
            h2 = h2 + lam * h1[:, -1:]  # linear recurrence
            h = torch.cat([h1, h2], 1)
        
        h = h.reshape(N, L, H)[:,-oringinal_seq_length:,:] + h0_contrib
        return h.real
    
class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    
    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)
    
class Triu(nn.Module):
    def __init__(self):
        super(Triu, self).__init__()
    
    def forward(self, x):
        return torch.tril(x)

class TransformerEncoder(nn.Module):
    def __init__(self,input_dim=-1, out_dim=-1, seq_len=-1, d_model=512, nhead=8, num_encoder_layers=6):
        super(TransformerEncoder, self).__init__()
        self.pos_encoder = nn.Embedding(seq_len, d_model)
        self.ff = nn.Linear(input_dim, d_model) 
        self.transformer_encoder_blocks = nn.ModuleList()
        for n in range(num_encoder_layers):
            self.transformer_encoder_blocks.append(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead))
        self.fc_out = nn.Linear(d_model, out_dim)
        
    def forward(self, x):
        x = self.ff(x) + self.pos_encoder(torch.arange(0, x.size(1), device=x.device)).unsqueeze(0).repeat(x.size(0), 1, 1)
        for transformer in self.transformer_encoder_blocks:
            x = transformer(x)
        x = self.fc_out(x)
        return x
        

class FST(nn.Module):
    def __init__(self, hid_seq_size, hid_feature_size, seq_out=None, fea_out=None, causal=False, dropout_rate=0.1):
        super(FST, self).__init__()

        if seq_out==None: seq_out = hid_seq_size
        if fea_out==None: fea_out = hid_feature_size
        
        self.seqence_ff = nn.Sequential(MaskLinearWeights(nn.Linear(hid_seq_size, hid_seq_size), causal=causal),
                                nn.ReLU(),
                                nn.Dropout(dropout_rate),
                                MaskLinearWeights(nn.Linear(hid_seq_size, seq_out), causal=causal))
        
        #self.seqence_skip = nn.Sequential(MaskLinearWeights(nn.Linear(hid_seq_size, hid_seq_size), causal=causal))

        self.alpha_seq_skip_init = torch.tensor(0.)
        self.alpha_seq_skip = nn.Parameter(self.alpha_seq_skip_init)
        
        self.seqence_rnn = LRU(hid_seq_size, hid_seq_size, mask_inputs=causal)
        
        self.feature_ff = nn.Sequential(nn.Linear(hid_feature_size, hid_feature_size),
                          nn.ReLU(),
                          nn.Dropout(dropout_rate),
                          nn.Linear(hid_feature_size, fea_out))
        
        #self.feature_skip = nn.Sequential(nn.Linear(hid_feature_size, fea_out))

        self.alpha_fea_skip_init = torch.tensor(0.)
        self.alpha_fea_skip = nn.Parameter(self.alpha_fea_skip_init)
        
        self.feature_rnn = LRU(hid_feature_size, hid_feature_size)  
        
        self.layer_norm_s = nn.LayerNorm(hid_seq_size, eps=1e-4)
        self.layer_norm_f = nn.LayerNorm(hid_feature_size, eps=1e-4)
        
    def forward_recurrent(self, x):
        
        afs = torch.sigmoid(self.alpha_fea_skip)
        ass = torch.sigmoid(self.alpha_seq_skip)

        x = x.transpose(1, 2)
        y = x #self.seqence_skip(x)
        x = self.seqence_rnn.forward_recurrent(x)
        x = self.seqence_ff(x)
        x = ass*x + (1-ass)*y
        x = self.layer_norm_s(x)
        x = x.transpose(1, 2)
        y = x #self.feature_skip(x)
        x = self.feature_rnn.forward_recurrent(x)
        x = self.feature_ff(x)
        x = afs*x + (1-afs)*y
        x = self.layer_norm_f(x)
        return x

    def forward(self, x):
        
        afs = torch.sigmoid(self.alpha_fea_skip)
        ass = torch.sigmoid(self.alpha_seq_skip)

        x = x.transpose(1, 2)
        y = x #self.seqence_skip(x)
        x = self.seqence_rnn(x)
        x = self.seqence_ff(x)
        x = ass*x + (1-ass)*y
        x = self.layer_norm_s(x)
        x = x.transpose(1, 2)
        y = x #self.feature_skip(x)
        x = self.feature_rnn(x)
        x = self.feature_ff(x)
        x = afs*x + (1-afs)*y
        x = self.layer_norm_f(x)
        return x
    
class Block_FST(nn.Module):
    def __init__(self, inp_size, seq_len, out_size, hid_seq_size, hid_feature_size, n_blocks, causal=False, dropout_rate=0.1):
        super(Block_FST, self).__init__()

        if type(hid_feature_size) == int:
            hid_feature_size = [hid_feature_size]*(n_blocks+1)
        if type(hid_seq_size) == int:
            hid_seq_size = [hid_seq_size]*(n_blocks+1)
        
        self.feature_in = nn.Sequential(nn.Linear(inp_size, hid_feature_size[0]),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(hid_feature_size[0], hid_feature_size[0]))
        
        
        
        self.FST_encoder_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.FST_encoder_blocks.append(FST(hid_seq_size[i], hid_feature_size[i], seq_out=hid_seq_size[i+1], fea_out=hid_feature_size[i+1], causal=causal))
        
        self.feature_out = nn.Sequential(nn.Linear(hid_feature_size[-1], out_size))
        
    def forward(self, x):
        
        x = self.feature_in(x)
        
        for block in self.FST_encoder_blocks:
            x = block(x)
          
        x = self.feature_out(x)
        
        return x
    
    
class FSU(nn.Module):
    def __init__(self, seq_len, hid_feature_size, fea_out=None, dropout_rate=0.1):
        super(FSU, self).__init__()

        if fea_out==None: fea_out = hid_feature_size
        
        self.feature_ff = nn.Sequential(nn.Linear(hid_feature_size, hid_feature_size),
                          nn.ReLU(),
                          nn.Dropout(dropout_rate),
                          nn.Linear(hid_feature_size, fea_out))
        
        self.feature_skip = nn.Sequential(nn.Linear(hid_feature_size, fea_out))
        
        self.feature_rnn = LRU(hid_feature_size, hid_feature_size, mask_inputs=False)
        
        self.alpha_fea_skip_init = torch.tensor(0.)
        self.alpha_fea_skip = nn.Parameter(self.alpha_fea_skip_init)
        
        self.layer_norm_f = nn.LayerNorm(hid_feature_size, eps=1e-6)


    def forward(self, x):
        
        afs = torch.sigmoid(self.alpha_fea_skip)
        
        y = self.feature_skip(x)
        x = self.feature_rnn(x)
        x = self.feature_ff(x)
        x = afs*x + (1-afs)*y
        x = self.layer_norm_f(x)
        return x
    
class Block_FSU(nn.Module):
    def __init__(self, inp_size, seq_len, out_size, hid_feature_size, n_blocks, dropout_rate = 0.1):
        super(Block_FSU, self).__init__()

        if type(hid_feature_size) == int:
            hid_feature_size = [hid_feature_size]*(n_blocks+1)
                    
        self.feature_in = nn.Sequential(nn.Linear(inp_size, hid_feature_size[0]),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(hid_feature_size[0], hid_feature_size[0]))
        
        self.FSU_encoder_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.FSU_encoder_blocks.append(FSU(seq_len, hid_feature_size[i], fea_out= hid_feature_size[i+1]))
        
        self.feature_out = nn.Sequential(nn.Linear(hid_feature_size[-1], out_size))
        
    def forward(self, x):
        
        x = self.feature_in(x)
        
        for block in self.FSU_encoder_blocks:
            x = block(x)
            
        x = self.feature_out(x)
        return x
