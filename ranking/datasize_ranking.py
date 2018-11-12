import csv
from collections import defaultdict
import os
import sys
def main(*args):
    f = sys.argv[1]
    print(f)
    datasize_dict = {}
    lang_set = set()
    with open(f, mode='r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        headers = next(reader, None)
        print ('headers',headers)
        
        #for feature in headers[2:]:
        for row in reader:
            lang_set.add(row[0])
            if row[0] not in datasize_dict:
                datasize_dict[row[0]] = int(row[1])
        print (datasize_dict.items())
        rank_list = sorted(datasize_dict.items(),key=lambda x:(-x[1],x[0]))
        print (rank_list)
        with open("datasize_rank.csv", mode='w') as output_file:
            writer = csv.writer(output_file)
            new_head = ['Task lang','rank1 lang','rank2 lang','rank3 lang']
            writer.writerow(new_head)
            for lang in sorted(lang_set):
                temp_row = []
                temp_row.append(lang)
                for i in range(3):
                    temp_row.append(rank_list[i][0])
                writer.writerow(temp_row)
if __name__ == '__main__':
    main()
