from __future__ import print_function
from flask import Flask, render_template
from flask import request, redirect, url_for, flash
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os
import sys
import time
import zipfile
from persephone import corpus
from persephone import run
from persephone import corpus_reader
from persephone import rnn_ctc

#Helper Functions: print to console for debugging
def print_console(str_to_print):
  print(str_to_print, file=sys.stderr)


UPLOAD_FOLDER = './uploads'
TRAIN_FOLDER = './train_file'
exp_dir ='./exp_dir'
persephone_venv_path = './persephone-venv'

if not os.path.exists(UPLOAD_FOLDER):
  os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(TRAIN_FOLDER):
  os.makedirs(TRAIN_FOLDER)
if not os.path.exists(exp_dir):
  os.makedirs(exp_dir)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TRAIN_FOLDER'] = TRAIN_FOLDER
app.config['exp_dir'] = exp_dir
app.config['persephone_venv_path'] = persephone_venv_path
app.secret_key = 'some_secret'


@app.route('/')
def index():
  return render_template('index.html')


@app.route("/to-upload/", methods=['GET', 'POST'])
def upload_file():
  if request.method == 'GET':
    return render_template('upload.html')

  # check if the post request has the file part
  print("request.files =", request.files)
  print("request.url =", request.url)
  if 'file' not in request.files:
    flash('No file part')
    return redirect(request.url)
  file = request.files['file']
  # if user does not select file, browser also
  # submit a empty part without filename
  if file.filename == '':
    flash('No selected file')
    return redirect(request.url)
  if file:
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return redirect('training.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
  return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Render the training.html: this is the page for training new models. User
# is able to upload training data and labels and train a new models.
# TODO: output the model performance back to the interface
# TODO: allow user to specify batch size, num_train, num_layers, and hidden size
@app.route('/training.html')
def training():
  return render_template('training.html')

# Methods for check the format of the training file. User needs to upload a 
# zipped folder that contains two folder "wav" and "label". The name and number of the wav
# audio and labels should match. If the format does not meet the requirement, the interface
# should return some error messages. 
# TODO: implement functions to check the format of the uploaded data. 
# TODO: Consider what kind of files are supported and what if users uploaded multiple files
# TODO: add some functions to organize the uploaded data and manage different experiements. (
# if user uploaded multiple batches of data, the interface needs to allow user select a 
# particular training batch)
@app.route('/check_train_file/',methods=['POST'])
def check_train_file():
  path_to_zip_file = app.config['UPLOAD_FOLDER']
  for filename in os.listdir(path_to_zip_file):
    if filename.endswith(".zip"):
      zip_ref = zipfile.ZipFile(os.path.join(path_to_zip_file,filename), 'r')
      zip_ref.extractall(app.config['TRAIN_FOLDER'])
      #In the future, users should be able to select a particular training batch
      break

  #Check if wav and label folders are uploaded
  base_addr = os.path.join(app.config['TRAIN_FOLDER'],filename.split(".")[0])
  if not os.path.exists(os.path.join(base_addr,'wav')):
    flash("You do not have a wav folder")
    
  if not os.path.exists(os.path.join(base_addr,'label')):
    flash("You do not have a label folder")
    
  else:
    train_num = len(os.listdir(os.path.join(base_addr,'label')))
    flash("Successuly uploaded "+str(train_num)+" training files")
    return redirect(url_for('training'))
  #TODO: check how many training files were uploaded and if filenames of wav and labels match each other
  return render_template('training.html')

# Call the persephone library to train a new model
# TODO: render the training output and progress back to the interface
@app.route('/train_new/',methods=['POST'])
def train_new_model():
  for folder in os.listdir(app.config['TRAIN_FOLDER']):
    na_corpus = corpus.ReadyCorpus(os.path.join(app.config['TRAIN_FOLDER'],folder))
    #Todo: Have users input num_train and batch_size
    na_reader = corpus_reader.CorpusReader(na_corpus, num_train=9, batch_size=9)
    model = rnn_ctc.Model(app.config['exp_dir'], na_reader, num_layers=2, hidden_size=250)
    model.train()
    break
  return render_template('training.html')


if __name__ == '__main__':
  app.run(debug=True)
