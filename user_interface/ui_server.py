from flask import Flask, render_template
from flask import request, redirect, url_for, flash
from flask import send_from_directory
from werkzeug.utils import secure_filename
import requests

import os
import sys
import inspect
import time
import zipfile
import shutil
import io


TRANSCRIBED_DATA_DIR = "./_transcribed_data"
if not os.path.exists(TRANSCRIBED_DATA_DIR):
  os.makedirs(TRANSCRIBED_DATA_DIR)


app = Flask(__name__)
app.secret_key = "secret_key"


@app.route("/")
def index():
  return render_template("index.html")


@app.route("/upload/", methods=["GET", "POST"])
def upload():
  if request.method == "GET":
    return render_template("index.html")

  print("request.files =", request.files)
  print("request.url =", request.url)

  file = request.files["file"]
  print("file =", file)
  print("Original file name =", file.filename)
  file_name = secure_filename(file.filename)
  print("Secure file name =", file_name)

  file_path = os.path.join(TRANSCRIBED_DATA_DIR, file_name)
  file.save(file_path)

  # Open as Python Binary I/O
  file_binary = open(file_path, "rb")


  """
  jsondata = {
      "audioFile": (file_binary, file_name),
      "fileURL": file_path,
      "id": 0,
      "name": file_name
    }

  r = requests.post("http://0.0.0.0:8080/v0.1/audio", json=jsondata)

  print("response from server =", r)
  dictFromServer = r.json()
  print("json from server =", dictFromServer)
  """

  data = {
      "audioFile": (file_binary, file_name)
      #"fileURL": file_path,
      #"id": 0,
      #"name": file_name
    }

  r = requests.post("http://0.0.0.0:8080/v0.1/audio", data=data)

  print("response from server =", r)
  dictFromServer = r.json()
  print("json from server =", dictFromServer)


  return "done"


if __name__ == "__main__":
  app.run(debug=True, port=4000)
