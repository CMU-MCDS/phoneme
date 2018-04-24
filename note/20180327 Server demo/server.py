from flask import Flask, render_template
from flask import request, redirect, url_for, flash
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os
import time


UPLOAD_FOLDER = './_uploads'
if not os.path.exists(UPLOAD_FOLDER):
  os.makedirs(UPLOAD_FOLDER)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
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
    return redirect(url_for('uploaded_file', filename=filename))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
  return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
  app.run(debug=True)
