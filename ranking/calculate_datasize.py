import pandas as pd
import subprocess
import os
from nltk.tokenize import sent_tokenize
from conllu import parse_incr

lang_names = []
wordcounts = []
sentence_counts = []
stats = pd.read_csv('stats.csv')
for idx, row in stats.iterrows():
    lname = row['language name']
    dir = row['language dir']
    train_file_name = dir + '/' + lname + '_' + dir.split('-')[-1].lower() + '-ud-train.conllu'
    dev_file_name = dir + '/' + lname + '_' + dir.split('-')[-1].lower() + '-ud-dev.conllu'
    test_file_name = dir + '/' + lname + '_' + dir.split('-')[-1].lower() + '-ud-test.conllu'
    sentence_count = 0
    word_count = 0
    # if int(row['num of training']) == 0:
    #     print('continue', file_name)
    #     continue
    print(idx, '=========================')
    lang_names.append(lname)
    if os.path.exists(train_file_name):
        print('train file exist:', train_file_name)
        train_words = subprocess.check_output(('wc -w ' + train_file_name).split())
        # train_sentence = subprocess.check_output(('wc -l ' + train_file_name).split())
        fin = open(train_file_name, 'r', encoding='utf-8')
        sentences = list(parse_incr(fin))
        if len(sentences) != int(row['num of training']):
            print('----------------')
            print('differ!', len(sentences), int(row['num of training']))
            print('----------------')
        sentence_count += len(sentences)
        word_count += int(train_words.split()[0])
        print('word count res:', train_words, int(train_words.split()[0]), 'sentence count res:', len(sentences))

    if os.path.exists(dev_file_name):
        print('dev file exist:', dev_file_name)
        dev_words = subprocess.check_output(('wc -w ' + dev_file_name).split())
        # dev_sentence = subprocess.check_output(('wc -l ' + dev_file_name).split())
        fin = open(dev_file_name, 'r', encoding='utf-8')
        sentences = list(parse_incr(fin))
        sentence_count += len(sentences)
        word_count += int(dev_words.split()[0])
        # sentence_count += int(dev_sentence.split()[0])
        print('word count res:', dev_words, int(dev_words.split()[0]), 'sentence count res:', len(sentences))

    if os.path.exists(test_file_name):
        print('test file exist:', test_file_name)
        test_words = subprocess.check_output(('wc -w ' + test_file_name).split())
        # test_sentence = subprocess.check_output(('wc -l ' + test_file_name).split())
        fin = open(test_file_name, 'r', encoding='utf-8')
        sentences = list(parse_incr(fin))
        sentence_count += len(sentences)
        word_count += int(test_words.split()[0])
        # sentence_count += int(test_sentence.split()[0])
        print('word count res:', test_words, int(test_words.split()[0]), 'sentence count res:', len(sentences))
    print('================================\n')
    wordcounts.append(word_count)
    sentence_counts.append(sentence_count)

print(len(lang_names), len(wordcounts))
df = pd.DataFrame({'Language': lang_names, 'Number pf words in training set': wordcounts})
df1 = pd.DataFrame({'Language': lang_names, 'Number pf sentences in training set': sentence_counts})
df.to_csv('POS_word_datasize.csv')
df1.to_csv('POS_sentence_datasize.csv')
