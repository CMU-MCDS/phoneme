import csv
from collections import defaultdict
import os
import sys
def main(*args):
    f = sys.argv[1]
    print(f)


    latex_pre = """    \\begin{table}[t!]
    \\begin{center}
    \\begin{tabular}{|l|r|r|}
    \\hline
    Task Lang & Selected Lang & Baseline Lang \\\\
    \\hline\n"""

    latex_suf = """    \\end{tabular}
    \\end{center}
    \\caption{\\label{tabel:CompareSelectedBaselineLang} Selected Languages compared to actual baseline languages}
    \\end{table}"""


    with open(f, mode='r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        temp = []
        line  = "    \\hline\n"
        count = 0
        for row in reader:
            count+=1
            if count == 1:
                continue
            target = row[0]
            select = ', '.join(row[1:4])
            baseline = ', '.join(row[4:])
            temp.append("    "+row[0]+" & "+select+" & "+baseline+" \\\\\n" + line)
        result = latex_pre+''.join(temp)+latex_suf
        print (result)
if __name__ == '__main__':
    main()
