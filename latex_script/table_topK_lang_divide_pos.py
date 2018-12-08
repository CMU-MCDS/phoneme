import os
import json
import math

def main():
    # Working directory on clio
    root = "/home/yuhsianl/public/phoneme_common_data/data/pos"

    output_dir = os.path.join(root, "output_pos")

    # The head and tail of the latex table
    latex_head = "\\begin{table}[t!]\n\\begin{center}\n\\begin{tabular}{ccc}\n\\midrule\n\\bf Task Lang & \\bf Suggested Lang & \\bf True Best Lang \\\\\n\\midrule\n"

    MAX_ROW_NUM = 14
    TOTAL_ROW_NUM = 26
    latex_filling_list = ["" for _ in range(int(math.ceil(TOTAL_ROW_NUM / MAX_ROW_NUM)))]
    cnt = 0

    file_list = []
    for f in os.listdir(output_dir):
        if os.path.isfile(os.path.join(output_dir, f)):
            file_name, file_ext = os.path.splitext(f)
            if file_name.split("_")[0] == "topK" and file_ext == ".json":
                file_list.append(os.path.join(output_dir, f))

    file_list = sorted(file_list)
    for file_name in file_list:
        with open(file_name, "r") as fp:
            topK_output_dict = json.load(fp)
        latex_filling_list[cnt // MAX_ROW_NUM] += "\\multirow{3}{*}{%s} & %s (%d) & %s (%d) \\\\\n & %s (%d) & %s (%d) \\\\\n & %s (%d) & %s (%d) \\\\\n\\midrule\n" % \
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
        latex_tail = "\\end{tabular}\n\\end{center}\n\\caption{\\label{tabel:SuggestedLangPOS%d} Suggested transfer languages for POS tagging compared to true best transfer languages. The numbers in the parentheses are the true ranking according to the BLEU scores. (Part %d)}\n\\end{table}" % (i+1, i+1)
        table_latex = latex_head + latex_filling_list[i] + latex_tail
        with open(os.path.join(output_dir, "suggested_lang_pos_%d.tex" % (i+1)), "w") as f:
            print(table_latex, file=f)

if __name__ == '__main__':
    main()
