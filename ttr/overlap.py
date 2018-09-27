from __future__ import division
from __future__ import print_function

import re
import string
import sys
import os

def main():

    file_dir = "/home/gneubig/exp/transfer-exp/data/"

    dir_list = os.listdir(file_dir)
    lang_list = []
    f = open("overlap.txt", "a")

    for d in dir_list:
        lang = d.split("_")[0]
        if lang !="eng":
            lang_list.append(lang)

    for lang1 in lang_list:
        wordset1 = get_wordset(lang1)
        for lang2 in lang_list:
            wordset2 = get_wordset(lang2)
            overlap = len(wordset1.intersection(wordset2))
            #print (wordset1.intersection(wordset2))
            print (lang1+" "+lang2+" "+str(overlap))
            f.write(lang1+" "+lang2+" "+str(overlap)+"\n")
    f.close()



def get_wordset(lang):

    filename = "/home/gneubig/exp/transfer-exp/data/"+lang+"_eng/ted-train.orig."+lang
    word_set = set()
    with open(filename, 'r') as file:
        for line in file:
            #line = strip_punctuation.sub('', line.lower())
            words = line.split(" ")
            for w in words:
                if w != '':
                    word_set.add(w)

    print ('filename = ',filename)
    return word_set 

if __name__ == '__main__':
    main()
