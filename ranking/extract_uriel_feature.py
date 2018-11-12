# Extract uriel distances between each pair of training and target language for the experiment
import csv
import pandas as pd
import os
import csv
import numpy as np
from scipy.stats import rankdata


root = "/home/yuhsianl/public/phoneme_common_data/data/"
#Set of languages for the experiment
lang_set = np.genfromtxt(root+"mt/language_set.txt", dtype = str)

### Load baseline experiment result: Get ranking ###
baseline = pd.read_csv(root+"mt/mt.csv")
baseline = baseline.set_index("v extra v \ trg >>")

table = []
header = ["Target","Training"]

table.append(header)
for lan_target in lang_set:
    for lan_train in lang_set: 
        if(lan_target == lan_train or baseline.loc[lan_train, lan_target] == 'X'):
            continue
        row = [lan_target, lan_train]
        table.append(row)
    

     
### Extract features: distances ###
distance_root = root + "uriel_v0_2/distances/"
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


with open(root + 'mt/uriel_distance.csv','w') as f:
    writer = csv.writer(f)
    for row in table:
        writer.writerow(row)
    