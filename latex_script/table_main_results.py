import os
import json


def main():
    # Working directory on clio
    root = "/home/yuhsianl/public/phoneme_common_data/data"

    symbol_dict = {"LambdaRank": "LambdaRank",
                   "Overlap word-level": "$o_w$",
                   "Overlap subword-level": "$o_{sw}$",
                   "Transfer lang dataset size": "$s_{tf}$",
                   "Target lang dataset size": "$s_{tg}$",
                   "Transfer over target size ratio": "$s_{tf} / s_{tg}$",
                   "Transfer lang TTR": "$t_{tf}$",
                   "Target lang TTR": "$t_{tg}$",
                   "Transfer target TTR distance": "$d_{ttr}$",
                   #"Entity overlap": "$o_e$",
                   "Entity overlap": "$o_w$",
                   "GENETIC": "$d_{gen}$",
                   "SYNTACTIC": "$d_{syn}$",
                   "FEATURAL": "$d_{fea}$",
                   "PHONOLOGICAL": "$d_{pho}$",
                   "INVENTORY": "$d_{inv}$",
                   "GEOGRAPHIC": "$d_{geo}$"}


    output_dir = os.path.join(root, "mt/output_mt")
    with open(os.path.join(output_dir, "NDCG.json"), "r") as f:
        NDCG_output_dict_mt = json.load(f)

    output_dir = os.path.join(root, "el/output_el")
    with open(os.path.join(output_dir, "NDCG.json"), "r") as f:
        NDCG_output_dict_el = json.load(f)

    output_dir = os.path.join(root, "pos/output_pos")
    with open(os.path.join(output_dir, "NDCG.json"), "r") as f:
        NDCG_output_dict_pos = json.load(f)


    output_dir = os.path.join(root, "mt/output_mt_dataset")
    with open(os.path.join(output_dir, "NDCG.json"), "r") as f:
        NDCG_output_dict_mt_dataset = json.load(f)

    output_dir = os.path.join(root, "el/output_el_dataset")
    with open(os.path.join(output_dir, "NDCG.json"), "r") as f:
        NDCG_output_dict_el_dataset = json.load(f)

    output_dir = os.path.join(root, "pos/output_pos_dataset")
    with open(os.path.join(output_dir, "NDCG.json"), "r") as f:
        NDCG_output_dict_pos_dataset = json.load(f)


    output_dir = os.path.join(root, "mt/output_mt_uriel")
    with open(os.path.join(output_dir, "NDCG.json"), "r") as f:
        NDCG_output_dict_mt_uriel = json.load(f)

    output_dir = os.path.join(root, "el/output_el_uriel")
    with open(os.path.join(output_dir, "NDCG.json"), "r") as f:
        NDCG_output_dict_el_uriel = json.load(f)

    output_dir = os.path.join(root, "pos/output_pos_uriel")
    with open(os.path.join(output_dir, "NDCG.json"), "r") as f:
        NDCG_output_dict_pos_uriel = json.load(f)


    latex_head = "\\begin{table*}[t!]\n\\begin{center}\n\\begin{tabular}{c|ccc}\n\\midrule\n\\bf Model & \\bf MT & \\bf EL & \\bf POS \\\\\n\\midrule\n"

    latex_filling_dataset = ""

    feature_name = "Overlap word-level"
    latex_filling_dataset += "%s & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f \\\\\n" % \
        (symbol_dict[feature_name],
         NDCG_output_dict_mt[feature_name]["avg"] * 100,
         NDCG_output_dict_mt[feature_name]["std"] * 100,
         NDCG_output_dict_el["Entity overlap"]["avg"] * 100,
         NDCG_output_dict_el["Entity overlap"]["std"] * 100,
         NDCG_output_dict_pos[feature_name]["avg"] * 100,
         NDCG_output_dict_pos[feature_name]["std"] * 100)

    feature_name = "Overlap subword-level"
    latex_filling_dataset += "%s & %.1f $\\pm$ %.1f & N/A & N/A \\\\\n" % \
        (symbol_dict[feature_name],
         NDCG_output_dict_mt[feature_name]["avg"] * 100,
         NDCG_output_dict_mt[feature_name]["std"] * 100)

    feature_name = "Transfer over target size ratio"
    latex_filling_dataset += "%s & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f \\\\\n" % \
        (symbol_dict[feature_name],
         NDCG_output_dict_mt[feature_name]["avg"] * 100,
         NDCG_output_dict_mt[feature_name]["std"] * 100,
         NDCG_output_dict_el[feature_name]["avg"] * 100,
         NDCG_output_dict_el[feature_name]["std"] * 100,
         NDCG_output_dict_pos[feature_name]["avg"] * 100,
         NDCG_output_dict_pos[feature_name]["std"] * 100)

    feature_name = "Transfer target TTR distance"
    latex_filling_dataset += "%s & %.1f $\\pm$ %.1f & N/A & %.1f $\\pm$ %.1f \\\\\n" % \
        (symbol_dict[feature_name],
         NDCG_output_dict_mt[feature_name]["avg"] * 100,
         NDCG_output_dict_mt[feature_name]["std"] * 100,
         NDCG_output_dict_pos[feature_name]["avg"] * 100,
         NDCG_output_dict_pos[feature_name]["std"] * 100)


    latex_filling_uriel = "\\midrule\n"

    feature_name = "GENETIC"
    latex_filling_uriel += "%s & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f \\\\\n" % \
        (symbol_dict[feature_name],
         NDCG_output_dict_mt[feature_name]["avg"] * 100,
         NDCG_output_dict_mt[feature_name]["std"] * 100,
         NDCG_output_dict_el[feature_name]["avg"] * 100,
         NDCG_output_dict_el[feature_name]["std"] * 100,
         NDCG_output_dict_pos[feature_name]["avg"] * 100,
         NDCG_output_dict_pos[feature_name]["std"] * 100)

    feature_name = "SYNTACTIC"
    latex_filling_uriel += "%s & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f \\\\\n" % \
        (symbol_dict[feature_name],
         NDCG_output_dict_mt[feature_name]["avg"] * 100,
         NDCG_output_dict_mt[feature_name]["std"] * 100,
         NDCG_output_dict_el[feature_name]["avg"] * 100,
         NDCG_output_dict_el[feature_name]["std"] * 100,
         NDCG_output_dict_pos[feature_name]["avg"] * 100,
         NDCG_output_dict_pos[feature_name]["std"] * 100)

    feature_name = "FEATURAL"
    latex_filling_uriel += "%s & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f \\\\\n" % \
        (symbol_dict[feature_name],
         NDCG_output_dict_mt[feature_name]["avg"] * 100,
         NDCG_output_dict_mt[feature_name]["std"] * 100,
         NDCG_output_dict_el[feature_name]["avg"] * 100,
         NDCG_output_dict_el[feature_name]["std"] * 100,
         NDCG_output_dict_pos[feature_name]["avg"] * 100,
         NDCG_output_dict_pos[feature_name]["std"] * 100)

    feature_name = "PHONOLOGICAL"
    latex_filling_uriel += "%s & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f \\\\\n" % \
        (symbol_dict[feature_name],
         NDCG_output_dict_mt[feature_name]["avg"] * 100,
         NDCG_output_dict_mt[feature_name]["std"] * 100,
         NDCG_output_dict_el[feature_name]["avg"] * 100,
         NDCG_output_dict_el[feature_name]["std"] * 100,
         NDCG_output_dict_pos[feature_name]["avg"] * 100,
         NDCG_output_dict_pos[feature_name]["std"] * 100)

    feature_name = "INVENTORY"
    latex_filling_uriel += "%s & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f \\\\\n" % \
        (symbol_dict[feature_name],
         NDCG_output_dict_mt[feature_name]["avg"] * 100,
         NDCG_output_dict_mt[feature_name]["std"] * 100,
         NDCG_output_dict_el[feature_name]["avg"] * 100,
         NDCG_output_dict_el[feature_name]["std"] * 100,
         NDCG_output_dict_pos[feature_name]["avg"] * 100,
         NDCG_output_dict_pos[feature_name]["std"] * 100)

    feature_name = "GEOGRAPHIC"
    latex_filling_uriel += "%s & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f \\\\\n" % \
        (symbol_dict[feature_name],
         NDCG_output_dict_mt[feature_name]["avg"] * 100,
         NDCG_output_dict_mt[feature_name]["std"] * 100,
         NDCG_output_dict_el[feature_name]["avg"] * 100,
         NDCG_output_dict_el[feature_name]["std"] * 100,
         NDCG_output_dict_pos[feature_name]["avg"] * 100,
         NDCG_output_dict_pos[feature_name]["std"] * 100)


    latex_filling_ranking = "\\midrule\n"

    feature_name = "LambdaRank"
    latex_filling_ranking += "%s & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f \\\\\n" % \
        ("LambdaRank-all",
         NDCG_output_dict_mt[feature_name]["avg"] * 100,
         NDCG_output_dict_mt[feature_name]["std"] * 100,
         NDCG_output_dict_el[feature_name]["avg"] * 100,
         NDCG_output_dict_el[feature_name]["std"] * 100,
         NDCG_output_dict_pos[feature_name]["avg"] * 100,
         NDCG_output_dict_pos[feature_name]["std"] * 100)

    feature_name = "LambdaRank"
    latex_filling_ranking += "%s & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f \\\\\n" % \
        ("LambdaRank-dataset",
         NDCG_output_dict_mt_dataset[feature_name]["avg"] * 100,
         NDCG_output_dict_mt_dataset[feature_name]["std"] * 100,
         NDCG_output_dict_el_dataset[feature_name]["avg"] * 100,
         NDCG_output_dict_el_dataset[feature_name]["std"] * 100,
         NDCG_output_dict_pos_dataset[feature_name]["avg"] * 100,
         NDCG_output_dict_pos_dataset[feature_name]["std"] * 100)

    feature_name = "LambdaRank"
    latex_filling_ranking += "%s & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f \\\\\n" % \
        ("LambdaRank-URIEL",
         NDCG_output_dict_mt_uriel[feature_name]["avg"] * 100,
         NDCG_output_dict_mt_uriel[feature_name]["std"] * 100,
         NDCG_output_dict_el_uriel[feature_name]["avg"] * 100,
         NDCG_output_dict_el_uriel[feature_name]["std"] * 100,
         NDCG_output_dict_pos_uriel[feature_name]["avg"] * 100,
         NDCG_output_dict_pos_uriel[feature_name]["std"] * 100)



    latex_tail = "\\midrule\n\\end{tabular}\n\\end{center}\n\\caption{\\label{tabel:MainResults} Our LambdaRank model leads to higher average NDCG@3 over the baselines on all three tasks: machine translation (MT), entity linking (EL), and part-of-speech tagging (POS). The reported baseline scores use the best-performing features per task.}\n\\end{table*}"

    table_latex = latex_head + latex_filling_dataset + latex_filling_uriel + latex_filling_ranking + latex_tail

    output_dir = os.path.join(root, "all")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "main_results.tex"), "w") as f:
        print(table_latex, file=f)

if __name__ == '__main__':
    main()
