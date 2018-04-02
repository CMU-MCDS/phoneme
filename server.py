from flask import Flask, render_template
from flask import request, redirect, url_for, flash
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os
import sys
import inspect
import time
import zipfile

from modules.persephone.persephone import corpus, corpus_reader, rnn_ctc


UPLOAD_DIR = "./_uploads"
if not os.path.exists(UPLOAD_DIR):
  os.makedirs(UPLOAD_DIR)

TRAIN_DIR = "./_train"
if not os.path.exists(TRAIN_DIR):
  os.makedirs(TRAIN_DIR)

EXP_DIR = "./_exp"
if not os.path.exists(EXP_DIR):
  os.makedirs(EXP_DIR)


app = Flask(__name__)
app.config["UPLOAD_DIR"] = UPLOAD_DIR
app.config["TRAIN_DIR"] = TRAIN_DIR
app.config["EXP_DIR"] = EXP_DIR
app.secret_key = "secret_key"


@app.route("/", methods = ["GET", "POST"])
def index():
  if request.method == 'GET':
    return render_template("index.html")

  # Check if the POST request has the file part
  if "file" not in request.files:
    flash("No file part")
    return redirect(request.url)

  file = request.files["file"]

  # If user does not select file, browser would also submit a empty part
  # without filename
  if file.filename == "":
    flash("No selected file")
    return redirect(request.url)

  if not file:
    return redirect(request.url)

  file.save(os.path.join(UPLOAD_DIR, secure_filename(file.filename)))

  prefix = None
  for f in os.listdir(UPLOAD_DIR):
    if f.endswith(".zip"):
      zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_DIR, f), "r")
      zip_ref.extractall(TRAIN_DIR)
      prefix = f.split(".")[0]
      # Here we only select the first zip file we found
      # In the future, users should be able to select a particular training batch
      break

  if not prefix:
    flash("Zip file not found")
    return redirect(request.url)

  # Check if wav and label folders are uploaded
  if not os.path.exists(os.path.join(TRAIN_DIR, prefix, "wav")):
    flash("You do not have a wav folder")
    return redirect(request.url)

  if not os.path.exists(os.path.join(TRAIN_DIR, prefix, "label")):
    flash("You do not have a label folder")
    return redirect(request.url)

  # TODO: check how many training files were uploaded and if filenames of wav and labels match each other

  # TODO: Have users input num_train and batch_size
  num_train = len(os.listdir(os.path.join(TRAIN_DIR, prefix, "wav")))
  flash("Successfully uploaded " + str(num_train) + " training files")
  batch_size = 14   # Hard code for my debugging dataset for now
  return redirect(url_for("train", num_train=num_train, batch_size=batch_size))

  """
  wav_path = os.path.join(UPLOAD_DIR, "wav")
  if not os.path.exists(wav_path):
    os.makedirs(wav_path)

  file.save(os.path.join(wav_path, filename))

  # Empty label file
  label_path = os.path.join(UPLOAD_DIR, "label")
  if not os.path.exists(label_path):
    os.makedirs(label_path)
  empty_label_file = open(os.path.join(UPLOAD_DIR, "label", filename[:-4] + ".phonemes"), "w+")
  empty_label_file.close()

  prefix_file = open(os.path.join(UPLOAD_DIR, "untranscribed_prefixes.txt"), "w+")
  # Drop the extension ".wav" to get prefix
  prefix_file.write(filename[:-4] + "\n")
  prefix_file.close()

  return redirect(url_for("transcription", filename=filename))
  """


@app.route("/train/<num_train>/<batch_size>")
def train(num_train, batch_size):
  # Input is casted into string, so restore their original type
  num_train = int(num_train)
  batch_size = int(batch_size)

  for folder in os.listdir(TRAIN_DIR):
    na_corpus = corpus.ReadyCorpus(os.path.join(TRAIN_DIR, folder))
    na_reader = corpus_reader.CorpusReader(na_corpus, num_train=num_train, batch_size=batch_size)
    model = rnn_ctc.Model(EXP_DIR, na_reader, num_layers=2, hidden_size=250)
    model.train()
    # Expect only one training data folder...
    break
  return render_template("train.html")


"""
@app.route("/transcribe/<filename>")
def transcribe(filename):
  na_corpus = corpus.ReadyCorpus(UPLOAD_DIR)
  na_reader = corpus_reader.CorpusReader(na_corpus, num_train=1, batch_size=1)

  output_dir = UPLOAD_DIR
  model = rnn_ctc.Model(output_dir, na_reader, num_layers=2, hidden_size=250)

  model.transcribe(restore_model_path = "../model_na_example/model_best.ckpt")

  return send_from_directory(os.path.join(UPLOAD_DIR, "transcriptions"), "hyps.txt")
"""


if __name__ == "__main__":
  app.run(debug=True)
