import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from psmpy import PsmPy
from psmpy.functions import cohenD
from imblearn.over_sampling import RandomOverSampler
from psmpy.plotting import *

import warnings
warnings.filterwarnings("ignore")

def get_1nn_CF(majority,minority,Fset,thislabel,metric):
    
    MatchID = pd.DataFrame(columns = ['majority_id','minority_id'])
    
    knn = KNeighborsClassifier(n_neighbors=1, algorithm='auto',metric = metric) 
    knn.fit(minority[Fset], minority[thislabel]) 

    for i in majority.index.values:
        indxs = knn.kneighbors(majority[majority.index == i][Fset], return_distance=False)
        pdict = {'majority_id':[i],'minority_id':[indxs[0][0]]}
        pdict = pd.DataFrame(pdict)
        MatchID = pd.concat([MatchID,pdict])
    
    MatchID = MatchID.reset_index(drop = True)
    print('MatchID:',MatchID.shape)
    return MatchID

def CMG(df,all_f,diag_f, proce_f, med_f, ts_f, basic,thislabel):
    
    df = df[all_f + ['ICUSTAY_ID',thislabel]]
    majority = df[df[thislabel] == 0].reset_index(drop = True)
    minority = df[df[thislabel] == 1].reset_index(drop = True)
    print(df.shape,majority.shape,minority.shape)
    
    psm = PsmPy(df[basic+diag_f+ts_f+['ICUSTAY_ID',thislabel]], treatment=thislabel, indx='ICUSTAY_ID')
    psm.logistic_ps(balance = True)
    psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=True)
    
    psmdata = psm.df_matched
    minority_paired = psmdata.iloc[:psmdata.shape[0]//2]
    paired = majority[majority['ICUSTAY_ID'].isin(minority_paired['matched_ID'])][all_f + ['ICUSTAY_ID',thislabel]]
    unpaird = majority[~majority['ICUSTAY_ID'].isin(paired['ICUSTAY_ID'])][all_f + ['ICUSTAY_ID',thislabel]]
    print(paired.shape,unpaird.shape)
    
    CFA_set = pd.concat([unpaird[minority.columns],minority])
    RO =  RandomOverSampler(random_state=42) 
    D_x, D_Y = RO.fit_resample(CFA_set[basic+diag_f+ts_f], CFA_set[[thislabel]])
    print(D_Y[thislabel].value_counts())
    
    D_RO = pd.concat([D_x, D_Y],axis=1)
    synthetic_D = D_RO.iloc[len(CFA_set):, :]
    synthetic_D = synthetic_D.reset_index(drop = True)
    
    DPM_original = CFA_set[basic+diag_f+ts_f+proce_f+med_f+[thislabel]]
    DPM_original = DPM_original.reset_index(drop = True)
    print(DPM_original.shape)
    
    MatchID_D = get_1nn_CF(synthetic_D[basic+diag_f+ts_f+[thislabel]],DPM_original[basic+diag_f+ts_f+[thislabel]],basic+diag_f+ts_f,thislabel,metric ='euclidean')
    paird_DPM = DPM_original.iloc[MatchID_D['minority_id'].values.tolist()].reset_index(drop = True)
    New_DPM = pd.concat([synthetic_D,paird_DPM[proce_f+med_f]],axis=1)
    
    New_ALL_generate = pd.concat([df[New_DPM.columns],New_DPM])
    
    return New_ALL_generate