from flask import Flask, render_template, flash
from flask import request, redirect, url_for
from flask import send_from_directory
from werkzeug.utils import secure_filename

import os
import sys
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from modules.persephone.persephone import corpus, corpus_reader, rnn_ctc


UPLOAD_FOLDER = "./_uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
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

    if file:
        filename = werkzeug.utils.secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], "wav", filename))

        prefix_file = open("UPLOAD_FOLDER" + "/untranscribed_prefixes.txt", "w+")
        # Drop the extension ".wav" to get prefix
        prefix_file.write(filename[:-4] + "\n")
        prefix_file.close()

        return redirect(url_for("transcription", filename = filename))
    else:
        return redirect(request.url)

@app.route("/transcription/<filename>")
def transcription(filename):
    na_corpus = corpus.ReadyCorpus(UPLOAD_FOLDER)
    na_reader = corpus_reader.CorpusReader(na_corpus, num_train = 1, batch_size = 1)

    output_dir = UPLOAD_FOLDER
    model = rnn_ctc.Model(output_dir, na_reader, num_layers = 2, hidden_size = 250)

    model.transcribe(restore_model_path = "../model_na_example/model_best.ckpt")

    return send_from_directory(app.config["UPLOAD_FOLDER"] + "/transcriptions", "hyps.txt")

if __name__ == "__main__":
    app.run(debug = True)
