
home="/home/chianyuc"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # current file
# tokenize
$home/mosesdecoder/scripts/tokenizer/tokenizer.perl < $1 > $home/corpus/phoneme.tok
$home/mosesdecoder/scripts/tokenizer/tokenizer.perl < $2 > $home/corpus/translate.tok

#truecasing
$home/mosesdecoder/scripts/recaser/train-truecaser.perl --model $home/corpus/truecase-model.phoneme --corpus $home/corpus/phoneme.tok

$home/mosesdecoder/scripts/recaser/train-truecaser.perl --model $home/corpus/truecase-model.translate --corpus $home/corpus/translate.tok

$home/mosesdecoder/scripts/recaser/truecase.perl --model $home/corpus/truecase-model.phoneme < $home/corpus/phoneme.tok > $home/corpus/true.phoneme

$home/mosesdecoder/scripts/recaser/truecase.perl --model $home/corpus/truecase-model.translate < $home/corpus/translate.tok  > $home/corpus/true.translate

#clean 
$home/mosesdecoder/scripts/training/clean-corpus-n.perl $home/corpus/true translate phoneme $home/corpus/clean 1 30

#mkdir $home/lm
cd $home/lm
$home/mosesdecoder/bin/lmplz -o 3 < $home/corpus/true.translate > arpa.translate


$home/mosesdecoder/bin/build_binary arpa.translate blm.translate

#mkdir $home/working
cd $home/working


nohup nice $home/mosesdecoder/scripts/training/train-model.perl -root-dir train \
 -corpus $home/corpus/clean                             \
 -f phoneme -e translate -alignment grow-diag-final-and -reordering msd-bidirectional-fe \
 -lm 0:3:$home/lm/blm.translate:8                          \
 -external-bin-dir $home/mosesdecoder/tools > training.out 2>&1 

gunzip $home/working/train/model/phrase-table.gz
python /home/jeanl1/phoneme/glossing.py $home/working/train/model/phrase-table glossing_dict_train.txt
echo $DIR
cp glossing_dict_train.txt $DIR/_glossing_dict/
