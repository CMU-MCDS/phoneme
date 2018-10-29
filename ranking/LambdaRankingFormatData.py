
# coding: utf-8

# In[303]:


import csv
import pandas as pd
import os
import csv
import numpy as np
import random
from scipy.stats import rankdata
from sklearn.preprocessing import normalize
root = "/Users/yuyanzhang/Desktop/Capstone/"


# In[169]:


### Load baseline experiment result: Get ranking ###
baseline = pd.read_csv(root+"baseline.csv")
baseline = baseline.set_index("v extra v \ trg >>")
lang_set = baseline.columns.values
baseline.head()
table = []

header = ["Target","Training","Baseline Score"]
# header.extend(category)
table.append(header)
rank = ["Rank"]
for lan_target in lang_set:
    score = []
    for lan_train in lang_set: 
        if(lan_target == lan_train or baseline.loc[lan_train, lan_target] == 'X'):
            continue
        row = [lan_target, lan_train, baseline.loc[lan_train, lan_target]]
        table.append(row)
        score.append(baseline.loc[lan_train, lan_target])
    
    rank.extend(rankdata(score))
        
print(len(table), len(rank))   
table = np.column_stack((np.array(table), np.array(rank)))


# In[170]:


print(table)
#print(pd.DataFrame(table).sort_values([0,2]))


# In[171]:


extracted_root = "/Users/yuyanzhang/Desktop/Capstone/extracted/"
ttr_table = pd.read_csv(extracted_root+"TTR.csv")
ttr_table = ttr_table.set_index("Lang")
overlap_word_table = pd.read_csv(extracted_root+"Overlap_wordlevel.csv")
overlap_word_table = overlap_word_table.set_index("lang")
overlap_subword_table = pd.read_csv(extracted_root+"Overlap_subwordlevel.csv")
overlap_subword_table = overlap_subword_table.set_index("lang")


# In[172]:


### Reformat features: TTR & overlap ###
extracted_type = [["TTR", "Overlap_wordlevel","Overlap_subwordlevel"]]
for lan_target in lang_set:   
    for lan_train in lang_set: 
        if(lan_target == lan_train or baseline.loc[lan_train, lan_target] == 'X'):
            continue
        ttr = ttr_table.loc[lan_train].values[0]
        overlap_word = overlap_word_table.loc[lan_train, lan_target]
        overlap_subword = overlap_subword_table.loc[lan_train, lan_target]
        row = [ttr, overlap_word, overlap_subword]
        extracted_type.append(row)
        
   
        
print(len(table), len(extracted_type))   
table = np.column_stack((np.array(table), np.array(extracted_type)))


# In[173]:


### Reformat features: distances ###
distance_root = "/Users/yuyanzhang/Desktop/Capstone/uriel_v0_2/distances/"
distance_category = [a.split(".")[0] for a in os.listdir(distance_root) if a.split(".")[1]=="csv"]

for item in distance_category:
    print(item)
    distance = [item]
    geo_dist = pd.read_csv(distance_root+item+".csv")
    geo_dist = geo_dist.set_index('G_CODE')
    geo_dist_sub = geo_dist[geo_dist.columns.intersection(lang_set)]
    geo_dist_sub = geo_dist_sub[geo_dist_sub.index.isin(lang_set)]
    count = 1
    for lan_target in lang_set:
        for lan_train in lang_set:
            if(lan_target == lan_train or baseline.loc[lan_train, lan_target] == 'X'):
                continue
            distance.append(geo_dist_sub.loc[lan_target, lan_train])
            count += 1
    table = np.column_stack((np.array(table), np.array(distance)))
    


# In[251]:


with open('all_2.csv','w') as f:
    writer = csv.writer(f)
    for row in table:
        writer.writerow(row)
print(table.shape)

pd.DataFrame(table).head()


# In[243]:


table = np.genfromtxt('all.csv',delimiter=',', dtype=None)
print(table.shape)


# In[249]:


### Reformat features: other ###
other_root = "/Users/yuyanzhang/Desktop/Capstone/uriel_v0_2/features/"
other_category = [a.split(".")[0] for a in os.listdir(other_root) if a.split(".")[1]=="csv" and a.split(".")[0] not in ['avg','predicted','all_sources']]
print(other_category)


for item in other_category:
    print(item)
    geo_dist = pd.read_csv(other_root+item+".csv")
    geo_dist = geo_dist.set_index('G_CODE')
    other = [list(geo_dist.columns.values)]
    
    count = 1
    for lan_target in lang_set:
        for lan_train in lang_set:
            if(lan_target == lan_train or baseline.loc[lan_train, lan_target] == 'X'):
                continue
           
            other.append(geo_dist.loc[lan_train].values)
           
            
            count += 1
    other = np.array(other).reshape(len(other), len(geo_dist.columns.values))
    print(table.shape, np.array(other).shape)
    table = np.column_stack((np.array(table), np.array(other)))
    


# In[250]:


pd.DataFrame(table).head()


# In[277]:


## Create file for ranking algorithm ##
data = np.genfromtxt('all_2.csv', delimiter = ",", dtype = None)


# In[343]:


data_sub = data[1:len(data_sub):, 0:14]
print(data[0,0:14])
print(data_sub.shape)

#Normalize
data_sub[:,5:len(data_sub[0])] = data_sub[:,5:len(data_sub[0])].astype(float)
normalized = normalize(data_sub[:,5:len(data_sub[0])], axis=0)
full = np.concatenate(( data_sub[:,[0,1,2,3,4]], normalized), axis=1)
print(full.shape)
np.random.shuffle(full)
split = 0.7

with open('rank.train','w') as f:
    pass
with open('rank.test','w') as f:
    pass
with open('rank.train.query','w') as f:
    pass
with open('rank.test.query','w') as f:
    pass

#Training
idx = 0
for row in full:    
    group = row[0]
    target = row[1]
    training = row[2]
    rank= str(int(float(row[4])))
    feature = row[5:len(row)]
    feature_dict = {k: v for k, v in enumerate(feature)}
    out = [rank]
    out.extend([str(k)+":"+str(v) for k,v in feature_dict.iteritems()])
    if idx < len(data_sub)*split:
        with open('rank.train','a') as f:
            f.write(" ".join(out)+"\n")
        with open('rank.train.query','a') as f:
            f.write(str(1)+"\n")
    else:
        with open('rank.test','a') as f:
            f.write(" ".join(out)+"\n")
        with open('rank.test.query','a') as f:
            f.write(str(1)+"\n")
    idx += 1


# In[344]:


#!/usr/bin/env python
import math
import os
import unittest

import lightgbm as lgb
import numpy as np
from sklearn import __version__ as sk_version
from sklearn.base import clone
from sklearn.datasets import (load_boston, load_breast_cancer, load_digits,
                              load_iris, load_svmlight_file)
from sklearn.exceptions import SkipTestWarning
from sklearn.externals import joblib
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils.estimator_checks import (_yield_all_checks, SkipTest,check_parameters_default_constructible)



X_train, y_train = load_svmlight_file('rank.train')
X_test, y_test = load_svmlight_file('rank.test')
q_train = np.loadtxt('rank.train.query')
q_test = np.loadtxt('rank.test.query')
gbm = lgb.LGBMRanker()
gbm.fit(X_train, y_train, group=q_train, eval_set=[(X_test, y_test)],eval_group=[q_test], eval_at=[1, 3], early_stopping_rounds=5, verbose=False,callbacks=[lgb.reset_parameter(learning_rate=lambda x: 0.95 ** x * 0.1)])


