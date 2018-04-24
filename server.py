from flask import Flask, render_template
from flask import request, redirect, url_for, flash
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os
import sys
import inspect
import time
import zipfile
import shutil

from modules.persephone.persephone import corpus, corpus_reader, rnn_ctc


TRANSCRIBED_DATA_DIR = "./_transcribed_data"
if not os.path.exists(TRANSCRIBED_DATA_DIR):
  os.makedirs(TRANSCRIBED_DATA_DIR)

MODELS_DIR = "./_models"
if not os.path.exists(MODELS_DIR):
  os.makedirs(MODELS_DIR)

UNTRANSCRIBED_DATA_DIR = "./_untranscribed_data"
if not os.path.exists(UNTRANSCRIBED_DATA_DIR):
  os.makedirs(UNTRANSCRIBED_DATA_DIR)

TRANSCRIBE_RESULTS_DIR = "./_transcribe_results"
if not os.path.exists(TRANSCRIBE_RESULTS_DIR):
  os.makedirs(TRANSCRIBE_RESULTS_DIR)

PRETRAINED_MODELS_DIR = "./pretrained_models"


app = Flask(__name__)
#app.config["TRANSCRIBED_DATA_DIR"] = TRANSCRIBED_DATA_DIR
#app.config["MODELS_DIR"] = MODELS_DIR
#app.config["UNTRANSCRIBED_DATA_DIR"] = UNTRANSCRIBED_DATA_DIR
#app.config["TRANSCRIBE_RESULTS_DIR"] = TRANSCRIBE_RESULTS_DIR
app.secret_key = "secret_key"

#app.config["TRAIN_PREFIX"] = None
#app.config["TRANSCRIBE_PREFIX"] = None


@app.route("/")
def index():
  return render_template('index.html')


@app.route("/upload_train/", methods = ["GET", "POST"])
def upload_train():
  if request.method == "GET":
    return render_template("upload_train.html")

  print("request.files =", request.files)
  print("request.url =", request.url)

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

  #file.save(os.path.join(UPLOAD_DIR, secure_filename(file.filename)))
  file.filename = secure_filename(file.filename)
  #print("file.filename =", file.filename)

  #for f in os.listdir(UPLOAD_DIR):
  #if f.endswith(".zip"):
  if file.filename.endswith(".zip"):
    #zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_DIR, f), "r")
    zip_ref = zipfile.ZipFile(file, "r")
    zip_ref.extractall(TRANSCRIBED_DATA_DIR)
    train_prefix = file.filename[:-4]
    # Here we only select the first zip file we found
    # In the future, users should be able to select a particular training batch
    #break

  if not train_prefix:
    flash("Zip file not found")
    return redirect(request.url)

  # Methods for check the format of the training file. User needs to upload a 
  # zipped folder that contains two folder "wav" and "label". The name and number of the wav
  # audio and labels should match. If the format does not meet the requirement, the interface
  # should return some error messages. 
  # TODO: implement functions to check the format of the uploaded data. 
  # TODO: Consider what kind of files are supported and what if users uploaded multiple files
  # TODO: add some functions to organize the uploaded data and manage different experiments. (
  # if user uploaded multiple batches of data, the interface needs to allow user select a 
  # particular training batch)

  # Check if wav and label folders are uploaded
  if not os.path.exists(os.path.join(TRANSCRIBED_DATA_DIR, train_prefix, "wav")):
    print(os.path.join(TRANSCRIBED_DATA_DIR, train_prefix, "wav"))
    flash("You do not have a wav folder")
    return redirect(request.url)

  if not os.path.exists(os.path.join(TRANSCRIBED_DATA_DIR, train_prefix, "label")):
    flash("You do not have a label folder")
    return redirect(request.url)

  # TODO: check how many training files were uploaded and if filenames of wav and labels match each other

  num_train = len(os.listdir(os.path.join(TRANSCRIBED_DATA_DIR, train_prefix, "wav")))
  flash("Successfully added " + str(num_train) + " training files")
  print("Successfully added " + str(num_train) + " training files")


  dataset_list = []
  for dataset_dir in os.listdir(TRANSCRIBED_DATA_DIR):
    if dataset_dir == "__MACOSX":
      continue
    else:
      dataset_list.append(" ".join((dataset_dir, str(num_train))))

  return render_template("train.html", dataset_list=dataset_list)


@app.route("/train/", methods = ["GET", "POST"])
def train():
  if request.method == "GET":
    return render_template("train.html")

  dataset_and_num_train = request.form["dataset_and_num_train"].split(" ")
  train_prefix = dataset_and_num_train[0]
  num_train = int(dataset_and_num_train[1])
  batch_size = int(request.form["batch_size"])
  min_epochs = int(request.form["min_epochs"])
  max_epochs = int(request.form["max_epochs"])

  model_dir = os.path.join(MODELS_DIR, train_prefix)

  na_corpus = corpus.ReadyCorpus(os.path.join(TRANSCRIBED_DATA_DIR, train_prefix))
  na_reader = corpus_reader.CorpusReader(na_corpus, num_train=num_train, batch_size=batch_size)
  model = rnn_ctc.Model(model_dir, na_reader, num_layers=2, hidden_size=250)
  model.train(min_epochs=min_epochs, max_epochs=max_epochs)

  shutil.copyfile(
    os.path.join(TRANSCRIBED_DATA_DIR, train_prefix, "phoneme_set.txt"),
    os.path.join(MODELS_DIR, train_prefix, "phoneme_set.txt"))

  print("\nTraining completed")
  flash("Completed training on " + train_prefix + " dataset")
  return render_template("train_complete.html")


@app.route("/upload_transcribe/", methods = ["GET", "POST"])
def upload_transcribe():
  if request.method == "GET":
    return render_template("upload_transcribe.html")

  print("request.files =", request.files)
  print("request.url =", request.url)

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

  #file.save(os.path.join(TRANSCRIBE_UPLOAD_DIR, secure_filename(file.filename)))
  file.filename = secure_filename(file.filename)

  if file.filename.endswith(".zip"):
    #zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_DIR, f), "r")
    zip_ref = zipfile.ZipFile(file, "r")
    zip_ref.extractall(UNTRANSCRIBED_DATA_DIR)
    transcribe_prefix = file.filename[:-4]

  if not transcribe_prefix:
    flash("Zip file not found")
    return redirect(request.url)

  # Check if wav and label folders are uploaded
  if not os.path.exists(os.path.join(UNTRANSCRIBED_DATA_DIR, transcribe_prefix, "wav")):
    flash("You do not have a wav folder")
    return redirect(request.url)

  num_transcribe = len(os.listdir(os.path.join(UNTRANSCRIBED_DATA_DIR, transcribe_prefix, "wav")))
  flash("Successfully added " + str(num_transcribe) + " untranscribed files")
  print("Successfully added " + str(num_transcribe) + " untranscribed files")


  dataset_list = []
  for dataset_dir in os.listdir(UNTRANSCRIBED_DATA_DIR):
    if dataset_dir == "__MACOSX":
      continue
    else:
      dataset_list.append(" ".join((dataset_dir, str(num_transcribe))))

  model_list = []
  for model in os.listdir(MODELS_DIR):
    if model == ".DS_Store":
      continue
    else:
      model_list.append(os.path.join(MODELS_DIR, model))

  for model in os.listdir(PRETRAINED_MODELS_DIR):
    if model == ".DS_Store":
      continue
    else:
      model_list.append(os.path.join(PRETRAINED_MODELS_DIR, model))

  return render_template("transcribe.html", dataset_list=dataset_list, model_list=model_list)


@app.route("/transcribe/", methods = ["GET", "POST"])
def transcribe():
  if request.method == "GET":
    return render_template("transcribe.html")

  dataset_and_num_transcribe = request.form["dataset_and_num_transcribe"].split(" ")
  transcribe_prefix = dataset_and_num_transcribe[0]
  num_transcribe = int(dataset_and_num_transcribe[1])

  batch_size = int(request.form["batch_size"])

  restore_model_name = request.form["model"]
  restore_model_path = os.path.join(restore_model_name, "model/model_best.ckpt")
  label_file_path = os.path.join(restore_model_name, "phoneme_set.txt")

  safe_restore_model_name = "-".join(restore_model_name.split("/")[1:])
  print("safe_restore_model_name=",safe_restore_model_name)

  result_path = os.path.join(TRANSCRIBE_RESULTS_DIR, "__".join((transcribe_prefix, safe_restore_model_name)))

  na_corpus = corpus.ReadyCorpus(os.path.join(UNTRANSCRIBED_DATA_DIR, transcribe_prefix), label_file_path=label_file_path, transcribe_new=True)
  na_reader = corpus_reader.CorpusReader(na_corpus, batch_size=batch_size, transcribe_new=True)
  model = rnn_ctc.Model(result_path, na_reader, num_layers=2, hidden_size=250)
  model.transcribe(restore_model_path=restore_model_path)

  return send_from_directory(os.path.join(result_path, "transcriptions"), "hyps.txt")


if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=4000)
