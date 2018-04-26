from flask import Flask, render_template
from flask import request, redirect, url_for, flash
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os
import sys
import inspect
import time
import zipfile
import json

from modules.persephone.persephone import corpus, corpus_reader, rnn_ctc


UPLOAD_DIR = "./_uploads"
if not os.path.exists(UPLOAD_DIR):
  os.makedirs(UPLOAD_DIR)

UPLOAD_DIR_GLOSSING = "./_uploads_glossing"
if not os.path.exists(UPLOAD_DIR_GLOSSING):
  os.makedirs(UPLOAD_DIR_GLOSSING)

TRAIN_DIR = "./_train"
if not os.path.exists(TRAIN_DIR):
  os.makedirs(TRAIN_DIR)

TRAIN_DIR_GLOSSING = "./_train_glossing"
if not os.path.exists(TRAIN_DIR_GLOSSING):
  os.makedirs(TRAIN_DIR_GLOSSING)

EXP_DIR = "./_exp"
if not os.path.exists(EXP_DIR):
  os.makedirs(EXP_DIR)

TRANSCRIBE_UPLOAD_DIR = "./_transcribe_uploads"
if not os.path.exists(TRANSCRIBE_UPLOAD_DIR):
  os.makedirs(TRANSCRIBE_UPLOAD_DIR)

TRANSCRIBE_DIR = "./_transcribe"
if not os.path.exists(TRANSCRIBE_DIR):
  os.makedirs(TRANSCRIBE_DIR)

TRANSCRIBE_EXP_DIR = "./_transcribe_exp"
if not os.path.exists(TRANSCRIBE_EXP_DIR):
  os.makedirs(TRANSCRIBE_EXP_DIR)

GLOSS_DICT_DIR = "./_glossing_dict/"
if not os.path.exists(GLOSS_DICT_DIR):
  os.makedirs(GLOSS_DICT_DIR)

app = Flask(__name__)
#Directories for glossing suggestion algorithm
app.config["UPLOAD_DIR_GLOSSING"] = UPLOAD_DIR_GLOSSING
app.config["TRAIN_DIR_GLOSSING"] = TRAIN_DIR_GLOSSING
app.config["GLOSS_DICT_DIR"] = GLOSS_DICT_DIR


app.config["UPLOAD_DIR"] = UPLOAD_DIR
app.config["TRAIN_DIR"] = TRAIN_DIR
app.config["EXP_DIR"] = EXP_DIR
app.config["TRANSCRIBE_UPLOAD_DIR"] = TRANSCRIBE_UPLOAD_DIR
app.config["TRANSCRIBE_DIR"] = TRANSCRIBE_DIR
app.config["TRANSCRIBE_EXP_DIR"] = TRANSCRIBE_EXP_DIR
app.secret_key = "secret_key"

app.config["TRAIN_PREFIX"] = None
app.config["TRANSCRIBE_PREFIX"] = None


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

  file.save(os.path.join(UPLOAD_DIR, secure_filename(file.filename)))

  for f in os.listdir(UPLOAD_DIR):
    if f.endswith(".zip"):
      zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_DIR, f), "r")
      zip_ref.extractall(TRAIN_DIR)
      app.config["TRAIN_PREFIX"] = f[:-4]
      # Here we only select the first zip file we found
      # In the future, users should be able to select a particular training batch
      break

  if not app.config["TRAIN_PREFIX"]:
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
  if not os.path.exists(os.path.join(TRAIN_DIR, app.config["TRAIN_PREFIX"], "wav")):
    print(os.path.join(TRAIN_DIR, app.config["TRAIN_PREFIX"], "wav"))
    flash("You do not have a wav folder")
    return redirect(request.url)

  if not os.path.exists(os.path.join(TRAIN_DIR, app.config["TRAIN_PREFIX"], "label")):
    flash("You do not have a label folder")
    return redirect(request.url)

  # TODO: check how many training files were uploaded and if filenames of wav and labels match each other

  num_train = len(os.listdir(os.path.join(TRAIN_DIR, app.config["TRAIN_PREFIX"], "wav")))
  flash("Successfully added " + str(num_train) + " training files")
  print("Successfully added " + str(num_train) + " training files")

  return redirect(url_for("train"))


@app.route("/train/", methods = ["GET", "POST"])
def train():
  if request.method == "GET":
    return render_template("train.html")

  # TODO: Have users input num_train and batch_size
  num_train = len(os.listdir(os.path.join(TRAIN_DIR, app.config["TRAIN_PREFIX"], "wav")))
  batch_size = 7   # Hard code for my debugging dataset for now
  min_epochs = 1
  max_epochs = 10

  na_corpus = corpus.ReadyCorpus(os.path.join(TRAIN_DIR, app.config["TRAIN_PREFIX"]))
  na_reader = corpus_reader.CorpusReader(na_corpus, num_train=num_train, batch_size=batch_size)
  model = rnn_ctc.Model(EXP_DIR, na_reader, num_layers=2, hidden_size=250)
  model.train(min_epochs=min_epochs, max_epochs=max_epochs)

  print("\nTraining completed")
  flash("Completed training on " + app.config["TRAIN_PREFIX"] + " dataset")
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

  file.save(os.path.join(TRANSCRIBE_UPLOAD_DIR, secure_filename(file.filename)))

  for f in os.listdir(TRANSCRIBE_UPLOAD_DIR):
    if f.endswith(".zip"):
      zip_ref = zipfile.ZipFile(os.path.join(TRANSCRIBE_UPLOAD_DIR, f), "r")
      zip_ref.extractall(TRANSCRIBE_DIR)
      app.config["TRANSCRIBE_PREFIX"] = f[:-4]
      # Here we only select the first zip file we found
      # In the future, users should be able to select a particular training batch
      break

  if not app.config["TRANSCRIBE_PREFIX"]:
    flash("Zip file not found")
    return redirect(request.url)

  # Check if wav and label folders are uploaded
  if not os.path.exists(os.path.join(TRANSCRIBE_DIR, app.config["TRANSCRIBE_PREFIX"], "wav")):
    flash("You do not have a wav folder")
    return redirect(request.url)

  num_transcribe = len(os.listdir(os.path.join(TRANSCRIBE_DIR, app.config["TRANSCRIBE_PREFIX"], "wav")))
  flash("Successfully added " + str(num_transcribe) + " untranscribed files")
  print("Successfully added " + str(num_transcribe) + " untranscribed files")

  return redirect(url_for("transcribe"))


@app.route("/transcribe/", methods = ["GET", "POST"])
def transcribe():
  if request.method == "GET":
    return render_template("transcribe.html")

  # TODO: Have users input num_train and batch_size
  batch_size = 64   # Hard code for my debugging dataset for now

  label_file_path = "./_train/na_train_tiny/phoneme_set.txt"

  na_corpus = corpus.ReadyCorpus(os.path.join(TRANSCRIBE_DIR, app.config["TRANSCRIBE_PREFIX"]), label_file_path=label_file_path, transcribe_new=True)
  na_reader = corpus_reader.CorpusReader(na_corpus, batch_size=batch_size, transcribe_new=True)
  model = rnn_ctc.Model(TRANSCRIBE_EXP_DIR, na_reader, num_layers=2, hidden_size=250)
  model.transcribe(restore_model_path="./_exp/model/model_best.ckpt")

  return send_from_directory(os.path.join(TRANSCRIBE_EXP_DIR, "transcriptions"), "hyps.txt")


@app.route("/upload_glossing_train/", methods = ['GET','POST'])
def upload_glossing_train():
  if request.method == "GET":
    return render_template("upload_glossing_train.html")

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

  file.save(os.path.join(UPLOAD_DIR_GLOSSING, secure_filename(file.filename)))

  for f in os.listdir(UPLOAD_DIR_GLOSSING):
    if f.endswith(".zip"):
      zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_DIR_GLOSSING, f), "r")
      zip_ref.extractall(TRAIN_DIR_GLOSSING)

      break

  #Check if the uploaded zip contains the correct files
  print(os.path.join(TRAIN_DIR_GLOSSING, 'phoneme.txt'))
  if not os.path.exists(os.path.join(TRAIN_DIR_GLOSSING, 'phoneme.txt')):
    flash("You do not have a phoneme.txt")
    return redirect(request.url)
  if not os.path.exists(os.path.join(TRAIN_DIR_GLOSSING, 'translation.txt')):
    flash("You do not have a translation.txt")
    return redirect(request.url)

  flash("Training data uploaded Successful")
  return redirect(url_for("train_glossing"))


@app.route("/train_glossing/", methods = ["GET", "POST"])
def train_glossing():
  if request.method == "GET":
    return render_template("train_glossing.html")
  
  cmd = 'bash /home/chianyuc/run_moses.sh phoneme.txt translate.txt'
  os.system(cmd)

  print("\nTraining completed")
  flash("Completed extracting dictionary")

  return render_template("train_complete.html")


@app.route("/upload_glossing/", methods = ['GET','POST'])
def upload_glossing():
  if request.method == "GET":
    return render_template("upload_glossing.html")
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

  file.save(os.path.join(UPLOAD_DIR_GLOSSING, secure_filename(file.filename)))

  output = []
  #Load the trained dictionary to generate glossing: create a dropdown list for user to choose a trained dictionary
  gloss_dictionary_list = []
  for dict_name in  os.listdir(GLOSS_DICT_DIR): 
    gloss_dictionary_list.append(os.path.join(GLOSS_DICT_DIR, dict_name))

  return render_template("upload_glossing.html", gloss_dict = gloss_dictionary_list)



@app.route("/suggest_glossing/",methods = ['GET','POST'])
def suggest_glossing():
  # if request.method == "GET":
  #   return render_template("suggest_glossing.html")


  # print("request.files =", request.files)
  # print("request.url =", request.url)

  # # Check if the POST request has the file part
  # if "file" not in request.files:
  #   flash("No file part")
  #   return redirect(request.url)

  # file = request.files["file"]

  # # If user does not select file, browser would also submit a empty part
  # # without filename
  # if file.filename == "":
  #   flash("No selected file")
  #   return redirect(request.url)

  # if not file:
  #   return redirect(request.url)

  # file.save(os.path.join(UPLOAD_DIR_GLOSSING, secure_filename(file.filename)))
  

  output = []
  # #Load the trained dictionary to generate glossing: create a dropdown list for user to choose a trained dictionary
  # gloss_dictionary_list = []
  # for dict_name in  os.listdir(GLOSS_DICT_DIR): 
  #   gloss_dictionary_list.append(os.path.join(GLOSS_DICT_DIR, dict_name))

  #Get user's selected dict
  selected_dict = request.form['gloss_dict']
  #print(selected_dict)
  gloss_dictionary = json.load(open(selected_dict))
  #print(gloss_dictionary)

  with open(os.path.join(UPLOAD_DIR_GLOSSING, 'new_phoneme.txt')) as f:
    for original_sent in f:
      original_sent = original_sent.strip().split(" ")
      output_gloss = []
      for phoneme_group in original_sent:
        try:
          gloss = gloss_dictionary[phoneme_group]
        except:
          gloss = "NA"
        output_gloss.append(gloss)
      
      output.append([original_sent, output_gloss])

  return render_template("suggest_glossing.html", gloss_generated = output)





if __name__ == "__main__":
  app.run(debug=True,host= '0.0.0.0',port=4000)
