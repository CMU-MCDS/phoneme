#python ../ttr.py /home/gneubig/exp/transfer-exp/data/ara_eng/ted-dev.mtok.eng " " >> /home/jeanl1/results/ttr_results.txt
for path in /home/gneubig/exp/transfer-exp/data/*
do
    [ -d "${path}" ] || continue # if not a directory, skip
    base="$(basename "${path}")"
    lang="$(cut -d'_' -f1 <<<"${base}")"
    dirname="${path}/ted-train.orig.${lang}"
    python ../ttr.py ${dirname} " ">> /home/jeanl1/results/ttr_results.txt
    echo $dirname

done
