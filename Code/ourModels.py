import warnings
warnings.filterwarnings("ignore")
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import dill
import numpy as np
import argparse
import sklearn.metrics as metrics
from sklearn.metrics import matthews_corrcoef,confusion_matrix,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, roc_curve
import random
from collections import Counter
import math
from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.combine import SMOTEENN,SMOTETomek
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from CMG import CMG

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def sclar_coder(rundf,num_v,cate_v):
    scaler = MinMaxScaler()
    rundf[num_v] = scaler.fit_transform(rundf[num_v])
    
    label_encoder = LabelEncoder()
    for col in cate_v:
        rundf[col] = label_encoder.fit_transform(rundf[col])

    return rundf

def code2text(df,d_ids,dictionary,coli,colt,name):
    column_mapping = {}

    for col in d_ids:
        icd_code = col.split('_')[1]
        texts = dictionary.loc[dictionary[coli] == icd_code, colt].values
        
        if len(texts) > 0:
            column_mapping[col] = tokenizer(texts[0], return_tensors='pt').input_ids
        else:
            column_mapping[col] = col

    df[name] = df[d_ids].rename(columns=column_mapping).apply(lambda row: [col for col in row.index if row[col] == 1], axis=1)
    return df

def atc2smile(df,d_ids,ATC_SMIL,name):
    column_mapping = {}
    for col in d_ids:
        icd_code = col.split('_')[1]
        
        if icd_code in ATC_SMIL:
            d = '\t'.join(ATC_SMIL[icd_code])  
            column_mapping[col] = d
        else:
            column_mapping[col] = col 

    df[name] = df[d_ids].rename(columns=column_mapping).apply(lambda row: [col for col in row.index if row[col] == 1], axis=1)
    return df

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        qkv = self.to_qkv(x)  
        q, k, v = qkv.chunk(3, dim=-1)  
        q = q.view(q.size(0), -1, self.heads, q.size(-1) // self.heads).transpose(1, 2)  
        k = k.view(k.size(0), -1, self.heads, k.size(-1) // self.heads).transpose(1, 2)  
        v = v.view(v.size(0), -1, self.heads, v.size(-1) // self.heads).transpose(1, 2)  

        attention_scores = torch.matmul(q, k.transpose(-2, -1))  
        scaled_attention_scores = attention_scores * self.scale

        attn_weights = F.softmax(scaled_attention_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        weighted_values = torch.matmul(attn_weights, v) 
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(q.size(0), -1, self.heads * (v.size(-1)))  
        
        out = self.to_out(weighted_values)

        return out, attn_weights

class TransformerLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.attn = nn.ModuleList([
            nn.LayerNorm(dim),
            SelfAttention(dim, heads, dim_head, dropout=attn_dropout)
        ])
        self.ffn = nn.ModuleList([
            nn.LayerNorm(dim),
            FeedForward(dim,mult=heads, dropout=ff_dropout)
        ])

    def forward(self, x):
        residual = x  
        x, attn = self.attn[1](self.attn[0](x))
        x = residual + x  
        x = x + self.ffn[1](self.ffn[0](x))
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(dim, heads, dim_head, attn_dropout, ff_dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x 
    
class Tab_Transformer(nn.Module):
    def __init__(self, k_ci, emb_dim, d_dim, p_dim, m_dim, o_dim, device):
        super(Tab_Transformer, self).__init__()
        
        self.k_ci = k_ci
        self.emb_dim = emb_dim
        self.d_dim = d_dim
        self.p_dim = p_dim
        self.m_dim = m_dim
        self.o_dim = o_dim
        self.device = device
        
        self.transformer_model = Transformer(
            emb_dim*4, depth=6, heads=8, dim_head=8, attn_dropout=0.75, ff_dropout=0.75
        )

        self.d = nn.Sequential(nn.Linear(self.d_dim, emb_dim), nn.ReLU(),nn.Linear(self.emb_dim, emb_dim))
        self.p = nn.Sequential(nn.Linear(self.p_dim+self.d_dim, emb_dim), nn.ReLU(),nn.Linear(self.emb_dim, emb_dim))
        self.m = nn.Sequential(nn.Linear(self.m_dim+self.p_dim+self.d_dim, emb_dim), nn.ReLU(),nn.Linear(self.emb_dim, emb_dim))
        self.o = nn.Sequential(nn.Linear(self.o_dim, emb_dim), nn.ReLU(),nn.Linear(self.emb_dim, emb_dim))

        self.fc = nn.Sequential(
            nn.Linear(4*emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(emb_dim, 1)
        )

    def forward(self, x):
        samples = x.clone()  
        samples1 = samples.clone() 
        
        for i in range(samples1.shape[0]):
            item = samples1[i, :self.d_dim]  
            existing_ones = (item == 1).nonzero(as_tuple=True)[0].tolist()
            all_indices = set(range(self.d_dim))
            available_indices = list(all_indices - set(existing_ones))
            
            num_to_select = min(self.k_ci, len(available_indices))
            if num_to_select > 0:
                selected_indices = torch.tensor(
                    torch.randperm(len(available_indices))[:num_to_select].tolist(),
                    device=self.device
                )
                item[selected_indices] = 1

        d_seq = self.d(samples[:, :self.d_dim])
        p_seq = self.p(samples[:, :self.d_dim + self.p_dim])
        m_seq = self.m(samples[:, :self.d_dim + self.p_dim + self.m_dim])
        o_seq = self.o(samples[:, self.d_dim + self.p_dim + self.m_dim:])
        
        d_seq1 = self.d(samples1[:, :self.d_dim])
        p_seq1 = self.p(samples1[:, :self.d_dim + self.p_dim])
        m_seq1 = self.m(samples1[:, :self.d_dim + self.p_dim + self.m_dim])
        o_seq1 = self.o(samples1[:, self.d_dim + self.p_dim + self.m_dim:])

        cate_seq = self.transformer_model(torch.cat([d_seq, p_seq, m_seq, o_seq], dim=-1))
        cate_seq1 = self.transformer_model(torch.cat([d_seq1, p_seq1, m_seq1, o_seq1], dim=-1))

        out = self.fc(cate_seq)
        result = torch.sigmoid(out.mean(dim=1))

        out1 = self.fc(cate_seq1)
        result1 = torch.sigmoid(out1.mean(dim=1))

        return result.squeeze(), result1.squeeze()   
