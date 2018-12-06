import os
import json

def main():
    # Working directory on clio
    root = "/home/yuhsianl/public/phoneme_common_data/data/mt"

    # Working directory on your local machine
    # root = "/Users/yuhsianglin/Dropbox/cmu/11634A_11632A_capstone/20181029 Jupyter notebook"

    output_dir = os.path.join(root, "output_mt")

    symbol_dict = {"LambdaRank": "LambdaRank",
                   "Overlap word-level": "$o_w$",
                   "Overlap subword-level": "$o_{sw}$",
                   "Transfer lang dataset size": "$s_{tf}$",
                   "Target lang dataset size": "$s_{tg}$",
                   "Transfer over target size ratio": "$s_{tf} / s_{tg}$",
                   "Transfer lang TTR": "$t_{tf}$",
                   "Target lang TTR": "$t_{tg}$",
                   "Transfer target TTR distance": "$d_{ttr}$",
                   "GENETIC": "$d_{gen}$",
                   "SYNTACTIC": "$d_{syn}$",
                   "FEATURAL": "$d_{fea}$",
                   "PHONOLOGICAL": "$d_{pho}$",
                   "INVENTORY": "$d_{inv}$",
                   "GEOGRAPHIC": "$d_{geo}$"}

    # Read NDCG output json file
    with open(os.path.join(output_dir, "NDCG.json"), "r") as f:
        NDCG_output_dict = json.load(f)

    # The head and tail of the latex table
    latex_head = "\\begin{table}[t!]\n\\begin{center}\n\\begin{tabular}{l|c}\n\\midrule\n\\bf Model & \\bf avg $\\pm$ std \\\\\n\\midrule\n"
    latex_tail = "\\midrule\n\\end{tabular}\n\\end{center}\n\\caption{\\label{tabel:SingleFeatureNDCGinMT} The average and standard deviation of NDCG@3 over 54 task languages in machine translation, using the LambdaRank mode, each single statistical feature, and each single URIEL distance.}\n\\end{table}\n"

    latex_filling = ""
    for model, ndcg_dict in NDCG_output_dict.items():
        latex_filling += "%s & %.1f $\\pm$ %.1f \\\\\n" % (symbol_dict[model], NDCG_output_dict[model]["avg"] * 100, NDCG_output_dict[model]["std"] * 100)

    table_latex = latex_head + latex_filling + latex_tail
    with open(os.path.join(output_dir, "latex_table_single_feature_NDCG.txt"), "w") as f:
        print(table_latex, file=f)

if __name__ == '__main__':
    main()
