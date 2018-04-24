from modules.persephone.persephone import corpus
from modules.persephone.persephone import run

corp = corpus.ReadyCorpus("./data/na_example")

run.train_ready(corp)
