import pandas as pd
import subprocess

lang_names = []
wordcounts = []
stats = pd.read_csv('stats.csv')
for idx, row in stats.iterrows():
    lname = row['language name']
    dir = row['language dir']
    file_name = dir + '/' + lname + '_' + dir.split('-')[-1].lower() + '-ud-train.txt'
    if int(row['num of training']) == 0:
        print('continue', file_name)
        continue
    lang_names.append(lname)
    print(file_name)
    res = subprocess.check_output(('wc -w ' + file_name).split())
    print(res, int(res.split()[0]))
    wordcounts.append(int(res.split()[0]))

print(len(lang_names), len(wordcounts))
df = pd.DataFrame({'Language':lang_names, 'Number pf words in training set':wordcounts})
df.to_csv('POS_word_datasize.csv')