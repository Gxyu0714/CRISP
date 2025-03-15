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
from ourModels import *

# Running settings
parser = argparse.ArgumentParser()

parser.add_argument('--CI', type=bool, default=True, help='causal inference loss')
parser.add_argument('--Dataset', type=str, default='III-IV', help='III|III-IV')
parser.add_argument('--Label', type=str, default='DIEINHOSPITAL', help='DIEINHOSPITAL')
parser.add_argument('--Balance', type=str, default='None', help='CMG |SMOTE | ADASYN | SMOTEENN | SMOTETomek | None')

parser.add_argument('--dim', type=int, default=64, help='dimension')
parser.add_argument('--epoch', type=int, default=200, help='training epoches')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--decay', type=float, default=0, help='weight decay')
parser.add_argument('--warmup', type=bool, default=False, help='lr warm up')

parser.add_argument('--w_ci', type=float, default=0.2, help='weight of causal inference loss')
parser.add_argument('--k_ci', type=int, default=10, help='k diagnoses of causal inference loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
args = parser.parse_args()
print(vars(args))

basic = ['AGE','GENDER','ADMISSION_TYPE']
diag_f = diagnoses
proce_f = procedures
med_f = medications
ts_f = indicators
all_f = diag_f + proce_f + med_f + ts_f + basic
print('all_f:',len(all_f),'diag:',len(diag_f),'proce:',len(proce_f),'med:',len(med_f),'ts',len(ts_f))

# load data
num_v = ts_f + ['AGE'] 
cate_v = ['GENDER','ADMISSION_TYPE','DIEINHOSPITAL'] + diag_f + proce_f + med_f

print('Loaded data!')
if args.Dataset == 'III-IV':
    III = pd.read_csv('output/Final_III.csv')
    III['MIMIC_Version'] = 'III'
    IV = pd.read_csv('output/Final_IV.csv')
    IV['MIMIC_Version'] = 'IV'
    df = pd.concat([III[all_f+['ICUSTAY_ID',args.Label,'MIMIC_Version']],IV[all_f+['ICUSTAY_ID',args.Label,'MIMIC_Version']]])
    df['ICUSTAY_ID'] = df['ICUSTAY_ID'].astype(int)
    df['GENDER'] = df['GENDER'].replace({2.0:'M',1.0:'F'})
    df['ADMISSION_TYPE'] = df['ADMISSION_TYPE'].replace({'EW EMER.':'EMERGENCY','DIRECT EMER.':'EMERGENCY'})
    df = df.reset_index(drop=True)
    df = code2text(df,diag_f,D_ICD_DIAGNOSES,'ICD_CODE','ICD_TEXT','Diag_TEXTs')
    df = code2text(df,proce_f,d_icd_procedures,'ICD_CODE','ICD_TEXT','Pro_TEXTs')
    df = atc2smile(df,med_f,ATC_SMIL,'Drug_TEXTs')
    
    print('Base samples:',df.shape,Counter(df[args.Label]))
    df = sclar_coder(df,num_v,cate_v)
    Train, valid = train_test_split(df[df.MIMIC_Version == 'III'], test_size=0.3, random_state=42)
    Test = df[df.MIMIC_Version == 'IV']
    
else:
    df = pd.read_csv('output/Final_III.csv')
    df['MIMIC_Version'] = 'III'
    print('Base samples:',df.shape,Counter(df[args.Label]))
    Train, Test = train_test_split(sclar_coder(df.copy(),num_v,cate_v), test_size=0.3, random_state=42)
    Train, valid = train_test_split(Train, test_size=0.2, random_state=42)

if args.Balance == 'CMG':
    print('CMG')
    Train = CMG(Train,all_f,diag_f, proce_f, med_f, ts_f, basic,args.Label)
    print(args.Label,Counter(Train[args.Label]))
elif args.Balance == 'SMOTE':
    bal = SMOTE(random_state=42)
    bal_x, bal_Y = bal.fit_resample(Train[all_f], Train[[args.Label]])
    Train = pd.concat([bal_x, bal_Y],axis=1)
    print(args.Label,Counter(Train[args.Label]))
elif args.Balance == 'ADASYN':
    bal = ADASYN(random_state=42)
    bal_x, bal_Y = bal.fit_resample(Train[all_f], Train[[args.Label]])
    Train = pd.concat([bal_x, bal_Y],axis=1)
    print(args.Label,Counter(Train[args.Label]))
elif args.Balance == 'SMOTEENN':
    bal = SMOTEENN(random_state=42)
    bal_x, bal_Y = bal.fit_resample(Train[all_f], Train[[args.Label]])
    Train = pd.concat([bal_x, bal_Y],axis=1)
    print(args.Label,Counter(Train[args.Label]))
elif args.Balance == 'SMOTETomek':
    bal = SMOTETomek(random_state=42)
    bal_x, bal_Y = bal.fit_resample(Train[all_f], Train[[args.Label]])
    Train = pd.concat([bal_x, bal_Y],axis=1)
    print(args.Label,Counter(Train[args.Label]))
else:
    print('without balance')
    print(args.Label,Counter(Train[args.Label]))


train_dataset = TensorDataset(torch.tensor(Train[all_f].values, dtype=torch.float), torch.tensor(Train[args.Label].values, dtype=torch.float))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_dataset = TensorDataset(torch.tensor(valid[all_f].values, dtype=torch.float), torch.tensor(valid[args.Label].values, dtype=torch.float))
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
test_dataset = TensorDataset(torch.tensor(Test[all_f].values, dtype=torch.float), torch.tensor(Test[args.Label].values, dtype=torch.float))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

def eval(model, data_eval,device):
    print('eval_adms:')
    print(vars(args))
    
    y_val_true = []
    y_val_pred = []
    
    for inputs, targets in tqdm(data_eval):
        inputs, targets = inputs.to(device), targets.to(device)
        result, result1 = model(inputs)

        y_val_true.extend(targets.cpu().numpy())
        y_val_pred.extend(np.atleast_1d(result.cpu().numpy()))
        
    best_f1 = 0
    best_threshold = 0

    for threshold in np.arange(0.0, 1.01, 0.01):
        predicted_labels = (y_val_pred > threshold).astype(int)
        f1 = f1_score(y_val_true, predicted_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Best F1 Score: {best_f1} at threshold: {best_threshold}")
    y_scale_pred = (y_val_pred > best_threshold).astype(int)
    
    
    val_auc = roc_auc_score(y_val_true, y_val_pred)
    prec_curve, recall_curve, thresholds_curve = metrics.precision_recall_curve(y_val_true, y_val_pred)
    val_prc = metrics.auc(recall_curve, prec_curve)
    
    fpr, tpr, _ = roc_curve(y_val_true, y_val_pred)

    acc = accuracy_score(y_val_true, y_scale_pred)
    pr = precision_score(y_val_true, y_scale_pred)
    re = recall_score(y_val_true, y_scale_pred)
    f1 = f1_score(y_val_true, y_scale_pred)
    mcc = matthews_corrcoef(y_val_true, y_scale_pred)  # MCC
    conf_matrix = confusion_matrix(y_val_true, y_scale_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    brier = brier_score_loss(y_val_true, y_val_pred)
        
    print(f'Eval: Val AUROC: {val_auc:.4f},val_prc:{val_prc:0.4f},acc:{acc:0.4f},pr:{pr:0.4f},re:{re:0.4f},f1:{f1:0.4f},mcc:{mcc:0.4f},brier:{brier:0.4f},specificity:{specificity:0.4f}')

    return val_auc, val_prc, acc, pr, re, f1, mcc, specificity, brier

def main():
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(True)
    
    # diag_f + proce_f + med_f + ts_f + basic
    model = Tab_Transformer(k_ci=args.k_ci, emb_dim=64, d_dim=len(diag_f), p_dim=len(proce_f), m_dim=len(med_f), o_dim=len(ts_f + basic), device=device)
    model = model.cuda()
    
    best_val_roc = float('-inf')
    patience = 10
    patience_counter = 0

    # start iterations
    history = defaultdict(list)

    EPOCH = args.epoch
    optimizer = Adam(list(model.parameters()), lr=args.lr, weight_decay=args.decay)
    if args.warmup:
        warmup_epoch = int(EPOCH * 0.3)
        iter_per_epoch = len(train_loader)
        warm_up_with_cosine_lr = lambda epoch: epoch / (warmup_epoch * iter_per_epoch) if epoch <= (
                warmup_epoch * iter_per_epoch) else 0.5 * (
                math.cos((epoch - warmup_epoch * iter_per_epoch) / (
                        (EPOCH - warmup_epoch) * iter_per_epoch) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    learning_rates = []
     
    print('EPOCH:', EPOCH)
    previous_model_path = ""
    for epoch in range(EPOCH):
        model.train()
        print('\n epoch {} -----------------------------------------------------'.format(epoch))
        for inputs, targets in tqdm(train_loader):
            optimizer.zero_grad()  
            inputs, targets = inputs.to(device), targets.to(device)
            result, result1 = model(inputs)

            # Forward pass
            loss_ce = nn.BCELoss()(result, targets)  
            loss_ci = -torch.log(F.sigmoid(torch.sign(targets - 0.5) * (result1 - result))).mean()  

            if args.CI:
                loss = loss_ce + args.w_ci * loss_ci  
            else:
                loss = loss_ce 

            # Backward pass
            loss.backward()

            optimizer.step()

            if args.warmup:
                scheduler.step()
                learning_rates.append(scheduler.get_last_lr()[0])

        # Eval state  
        model.eval()
        with torch.no_grad():
            auroc, auprc, acc, pr, re, f1, mcc, specificity, brier = eval(model,valid_loader,device)

            history['auroc'].append(auroc)
            history['auprc'].append(auprc)
            history['acc'].append(acc)
            history['pr'].append(pr)
            history['re'].append(re)
            history['f1'].append(f1)
            history['mcc'].append(mcc)
            history['specificity'].append(specificity)
            history['brier'].append(brier)
        
        if auprc > best_val_roc:
            if os.path.exists(previous_model_path):
                os.remove(previous_model_path)
                
            best_val_roc = auprc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break
        
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
