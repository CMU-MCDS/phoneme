import csv
from collections import defaultdict
import os
def main():
    file_dir = "/home/gneubig/exp/transfer-exp/data/"

    dir_list = os.listdir(file_dir)
    lang_list = []

    for d in dir_list:
        lang = d.split("_")[0]
        if lang !="eng":
            lang_list.append(lang)

    dict_lang = defaultdict(dict)
    with open('overlap.txt',mode='r') as file:
        for line in file.readlines():
            line = line.strip().split(' ')
            lang1 = line[0]
            lang2 = line[1]
            count = line[2]
            dict_lang[lang1][lang2] = count
            dict_lang[lang1]['lang'] = lang1
    with open('overlap.csv', mode='w') as csv_file:
        fieldnames = ['lang']+sorted(lang_list)
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for lang in sorted(dict_lang.keys()):
            #print (dict_lang[lang])
            writer.writerow(dict_lang[lang])

if __name__ == '__main__':
    main()
