#from collections import defaultdict
import os
import json

def main():
    # Working directory on clio
    # root = "/home/yuhsianl/public/phoneme_common_data/data/mt"

    # Working directory on your local machine
    root = "/Users/yuhsianglin/Dropbox/cmu/11634A_11632A_capstone/20181029 Jupyter notebook"

    output_dir = os.path.join(root, "output_mt")

    # Read NDCG output json file
    with open(os.path.join(output_dir, "NDCG.json"), "r") as f:
        NDCG_output_dict = json.load(f)

    # The head and tail of the latex table
    latex_head = "\\begin{table}[t!]\n\\begin{center}\n\\begin{tabular}{|l|r|r|}\n\\hline\n\\bf Model & \\bf Avg & \\bf Std \\\\\n\\hline\n"
    latex_tail = "\\hline\n\\end{tabular}\n\\end{center}\n\\caption{\\label{tabel:SingleFeatureNDCGinMT} The average and standard deviation of NDCG@3 over 54 task languages in machine translation, using the LambdaRank mode, each single statistical feature, and each single URIEL distance.}\n\\end{table}\n"

    latex_filling = ""
    for model, ndcg_dict in NDCG_output_dict.items():
        latex_filling += "%s & %.3f & %.3f \\\\\n" % (model, NDCG_output_dict[model]["avg"], NDCG_output_dict[model]["std"])

    table_latex = latex_head + latex_filling + latex_tail
    with open(os.path.join(output_dir, "latex_table_single_feature_NDCG.txt"), "w") as f:
        print(table_latex, file=f)

if __name__ == '__main__':
    main()
