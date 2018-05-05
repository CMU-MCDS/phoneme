
home="/home/chianyuc"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # current file
# tokenize
user_path=$DIR/_glossing_dict/$3/corpus/
echo $1 
echo $2
echo $3
echo $user_path
mkdir $user_path
$home/mosesdecoder/scripts/tokenizer/tokenizer.perl < $1 > $user_path/phoneme.tok
$home/mosesdecoder/scripts/tokenizer/tokenizer.perl < $2 > $user_path/translate.tok

#truecasing
$home/mosesdecoder/scripts/recaser/train-truecaser.perl --model $user_path/truecase-model.phoneme --corpus $user_path/phoneme.tok

$home/mosesdecoder/scripts/recaser/train-truecaser.perl --model $user_path/truecase-model.translate --corpus $user_path/translate.tok

$home/mosesdecoder/scripts/recaser/truecase.perl --model $user_path/truecase-model.phoneme < $user_path/phoneme.tok > $user_path/true.phoneme

$home/mosesdecoder/scripts/recaser/truecase.perl --model $user_path/truecase-model.translate < $user_path/translate.tok  > $user_path/true.translate

#clean 
$home/mosesdecoder/scripts/training/clean-corpus-n.perl $user_path/true translate phoneme $user_path/clean 1 30

#mkdir $home/lm
user_lm=$DIR/_glossing_dict/$3/lm
mkdir $user_lm
cd $user_lm
$home/mosesdecoder/bin/lmplz -o 3 < $user_path/true.translate > arpa.translate


$home/mosesdecoder/bin/build_binary arpa.translate blm.translate

#mkdir $home/working

user_working=$DIR/_glossing_dict/$3/working
mkdir $user_working 
cd $user_working

nohup nice $home/mosesdecoder/scripts/training/train-model.perl -root-dir train \
 -corpus $user_path/clean                             \
 -f phoneme -e translate -alignment grow-diag-final-and -reordering msd-bidirectional-fe \
 -lm 0:3:$user_lm/blm.translate:8                          \
 -external-bin-dir $home/mosesdecoder/tools > training.out 2>&1 

gunzip $user_working/train/model/phrase-table.gz
python $DIR/glossing.py $user_working/train/model/phrase-table glossing_dict_train.txt
echo $DIR
# cp glossing_dict_train.txt $DIR/_glossing_dict/$3/
