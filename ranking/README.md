# Setup LambdaRank with LightGBM

On clio server, first prepare the pip environment:

```
pip install --upgrade pip
pip install wheel
```

Then install lightgbm from pip:

```
pip install lightgbm
```

# Run scripts on clio

The raw data (NLP transfer experiments results) is stored on directories under ```/home/yuhsianl/public/phoneme_common_data/data```.

[Download](http://www.cs.cmu.edu/~dmortens/uriel.html?fbclid=IwAR16ivxYi6kuaCFMapB-pkTGLk2B7x3MTITGi7sENiCzMsh5WDpGErGQYKo) URIEL distance data; it has been downloaded to clio in ```/home/yuhsianl/public/phoneme_common_data/data/uriel_v0_2/distances```.

Currently running the experiments takes two steps:

1. Aggregate various raw data files into a single big table (each row is about a training instance with some (target language, transfer language), and multiple rows constitute a query group (instances of all transfer languages for a particular target language)): Run

```
python GenerateDataFile_RankMT.py
python GenerateDataFile_RankEL.py
python GenerateDataFile_RankPOS.py
python GenerateDataFile_RankParsing.py
```

for each task.

2. Run the experiments (with all features; the "full LangRank model") by

```
python LambdaRankMT.py
python LambdaRankEL.py
python LambdaRankPOS.py
python LambdaRankParsing.py
```

The experiment results are written into output directories under the corresponding raw data directory.

To train LangRank-dataset or LangRank-URIEL models, run

```
python LambdaRankMT_dataset.py
python LambdaRankMT_uriel.py
```

and so on.

# Generate LaTeX tables

Use scripts in ```phoneme/latex_script``` of this repo. Many hardcoded things (for table format) there so modify as needed.
