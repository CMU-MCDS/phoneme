import os
import json
import math

def main():
    # Working directory on clio
    # root = "/home/yuhsianl/public/phoneme_common_data/data/mt"

    # Working directory on your local machine
    root = "/Users/yuhsianglin/Dropbox/cmu/11634A_11632A_capstone/20181029 Jupyter notebook"

    output_dir = os.path.join(root, "output_mt")

    # The head and tail of the latex table
    latex_head = "\\begin{table}[t!]\n\\begin{center}\n\\begin{tabular}{|c|c|c|}\n\\hline\n\\bf Task Lang & \\bf Suggested Lang & \\bf True Best Lang \\\\\n\\hline\n"
    latex_tail = "\\end{tabular}\n\\end{center}\n\\caption{\label{tabel:CompareSelectedBaselineLang} Suggested auxiliary languages compared to true best auxiliary languages. The numbers in the parentheses are the true ranking according to the BLEU scores.}\n\\end{table}\n"

    MAX_ROW_NUM = 15
    latex_filling_list = ["" for _ in range(int(math.ceil(54 / MAX_ROW_NUM)))]
    cnt = 0

    for f in os.listdir(output_dir):
        if os.path.isfile(os.path.join(output_dir, f)):
            file_name, file_ext = os.path.splitext(f)
            if file_name.split("_")[0] == "topK" and file_ext == ".json":
                with open(os.path.join(output_dir, f), "r") as fp:
                    topK_output_dict = json.load(fp)
                latex_filling_list[cnt // MAX_ROW_NUM] += "\\multirow{3}{*}{%s} & %s (%d) & %s (%d) \\\\\n & %s (%d) & %s (%d) \\\\\n & %s (%d) & %s (%d) \\\\\n\\hline\n" % \
                    (topK_output_dict["task_lang"],
                     topK_output_dict["LambdaRank"][0][0],
                     topK_output_dict["LambdaRank"][1][0],
                     topK_output_dict["Truth"][0][0],
                     topK_output_dict["Truth"][1][0],
                     topK_output_dict["LambdaRank"][0][1],
                     topK_output_dict["LambdaRank"][1][1],
                     topK_output_dict["Truth"][0][1],
                     topK_output_dict["Truth"][1][1],
                     topK_output_dict["LambdaRank"][0][2],
                     topK_output_dict["LambdaRank"][1][2],
                     topK_output_dict["Truth"][0][2],
                     topK_output_dict["Truth"][1][2])
                cnt += 1

    for i in range(len(latex_filling_list)):
        table_latex = latex_head + latex_filling_list[i] + latex_tail
        with open(os.path.join(output_dir, "latex_table_topK_lang_" + str(i) + ".txt"), "w") as f:
            print(table_latex, file=f)

if __name__ == '__main__':
    main()
