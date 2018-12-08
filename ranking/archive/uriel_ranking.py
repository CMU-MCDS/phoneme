import csv
from collections import defaultdict
import os
import sys
def main(*args):
    f = sys.argv[1]
    print(f)
    distance_dict = {}
    lang_set = set()
    with open(f, mode='r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        headers = next(reader, None)
        print ('headers',headers)
        
        #for feature in headers[2:]:
        for row in reader:
            lang_set.add(row[0])
            if row[0] not in distance_dict:
                distance_dict[row[0]] = {row[1]:row[2:]}
            else:
                distance_dict[row[0]][row[1]] = row[2:]
        
        for index,feature in enumerate(headers[2:]):
            with open(feature+"_rank.csv", mode='w') as output_file:
                writer = csv.writer(output_file)
                new_head = ['Task lang','rank1 lang','rank2 lang','rank3 lang']
                writer.writerow(new_head)
                for lang in sorted(lang_set):
                    task_dict = distance_dict[lang]
                    temp_list = []
                    for key,val in task_dict.items():
                        if lang == key:
                            continue
                        temp_list.append((float(val[index]),key))
                    temp_list = sorted(temp_list,key=lambda x:(-x[0],x[1]))[:3]
                    temp_row = [lang]
                    for i in range(len(temp_list)):
                        temp_row.append(temp_list[i][1])
                    writer.writerow(temp_row)
if __name__ == '__main__':
    main()
