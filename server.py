from flask import Flask, render_template, abort
from flask import request, redirect, url_for, flash
from flask import send_from_directory
from werkzeug.utils import secure_filename
from flask_login import login_required, LoginManager, UserMixin, logout_user, login_user, current_user
from flask_sqlalchemy import SQLAlchemy
from forms import *
import os
import sys
from sqlalchemy.orm.exc import NoResultFound
import inspect
from datetime import datetime
import time
import zipfile
import json
from flask_dance.contrib.google import make_google_blueprint, google
from modules.persephone.persephone import corpus, corpus_reader, rnn_ctc
from flask_dance.consumer.backend.sqla import OAuthConsumerMixin, SQLAlchemyBackend
from flask_dance.consumer import oauth_authorized, oauth_error
import ast

"""Todo 
1. run train in back end and click a button to check[]
"""

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
# manage authentication
login_manage = LoginManager()

blueprint = make_google_blueprint(
    client_id="1027474582240-peg8bf4qfu8f4fmbn7jboro4epmovm0g.apps.googleusercontent.com",
    client_secret="Jkn0TENqtiFx-N4ZnD1OEmOm",
    scope=["profile", "email"]
)

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

USER_DOWNLOAD_DIR="./_user_download"
if not os.path.exists(USER_DOWNLOAD_DIR):
    os.makedirs(USER_DOWNLOAD_DIR)

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
# Directories for glossing suggestion algorithm
app.config["UPLOAD_DIR_GLOSSING"] = UPLOAD_DIR_GLOSSING
app.config["TRAIN_DIR_GLOSSING"] = TRAIN_DIR_GLOSSING
app.config["GLOSS_DICT_DIR"] = GLOSS_DICT_DIR

app.config["UPLOAD_DIR"] = UPLOAD_DIR
app.config["TRAIN_DIR"] = TRAIN_DIR
app.config["EXP_DIR"] = EXP_DIR
app.config["TRANSCRIBE_UPLOAD_DIR"] = TRANSCRIBE_UPLOAD_DIR
app.config["TRANSCRIBE_DIR"] = TRANSCRIBE_DIR
app.config["TRANSCRIBE_EXP_DIR"] = TRANSCRIBE_EXP_DIR
app.config['USER_DOWNLOAD_DIR']= USER_DOWNLOAD_DIR

app.secret_key = "secret_key"

app.config["TRAIN_PREFIX"] = []
app.config["TRANSCRIBE_PREFIX"] = []

# config for database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///phoneme.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# declare sqlalchemy
db = SQLAlchemy(app)
app.register_blueprint(blueprint, url_prefix="/login")
login_manage.init_app(app)
login_manage.login_view = "login"


# User class:
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80),  nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    is_traindata_pub = db.Column(db.Boolean, nullable=False, default=False)
    is_untranscript_pub = db.Column(db.Boolean, nullable=False, default=False)
    is_glosstrain_pub = db.Column(db.Boolean, nullable=False, default=False)
    is_model_pub = db.Column(db.Boolean, nullable=False, default=False)

    def __init__(self, username, password, email):
        self.username = username
        self.password = password
        self.email = email

    def __repr__(self):
        return '<User %r>' % self.username


class OAuth(OAuthConsumerMixin, db.Model):
    provider_user_id = db.Column(db.String(256), unique=True)
    user_id = db.Column(db.Integer, db.ForeignKey(User.id))
    user = db.relationship(User)



@oauth_authorized.connect_via(blueprint)
def google_logged_in(blueprint, token):
    if not google.authorized:
        return redirect(url_for("google.login"))
    if not token:
        flash("Failed to log in with Google.")
        return False
    resp = google.get("/oauth2/v2/userinfo")
    if not resp.ok:
        msg = "Failed to fetch user info from Google."
        flash(msg)
        return False
    google_info = resp.json()
    print('google info',google_info)
    google_user_id = str(google_info["id"])

    query = OAuth.query.filter_by(
        provider=blueprint.name,
        provider_user_id=google_user_id,
    )
    try:
        oauth = query.one()
    except NoResultFound:
        oauth = OAuth(
            provider=blueprint.name,
            provider_user_id=google_user_id,
            token=token
        )
    if oauth.user:
        login_user(oauth.user)
        flash("Existed user. Successfully signed in with Google.")

    else:
        # Create a new local user account for this user
        user = User(
            # Remember that `email` can be None, if the user declines
            # to publish their email address on GitHub!
            email=google_info["email"],
            username=google_info["name"],
            password=google_info["email"],
        )
        # Associate the new local user account with the OAuth token
        oauth.user = user
        # Save and commit our database models
        db.session.add_all([user, oauth])
        db.session.commit()
        # Log in the new local user account
        login_user(user)
        flash("New user. Successfully signed in with Google.")
    # Disable Flask-Dance's default behavior for saving the OAuth token
    return False

# notify on OAuth provider error
@oauth_error.connect_via(blueprint)
def google_error(blueprint, error, error_description=None, error_uri=None):
    msg = (
        "OAuth error from {name}! "
        "error={error} description={description} uri={uri}"
    ).format(
        name=blueprint.name,
        error=error,
        description=error_description,
        uri=error_uri,
    )
    flash(msg)

# setup SQLAlchemy backend
blueprint.backend = SQLAlchemyBackend(OAuth, db.session, user=current_user)


# models
# class uploaded_traindata(db.Model):
#   pass
#
# class uploaded_untranscriped_data(db.Model):
#   pass
#
# class trained_model(db.Model):
#   pass
#
# class uploaded_glossing(db.Model):
#   pass
#
# class uploaded_glosstest(db.Model):
#   pass

# For safe redirect, currently for simplicity maybe we don't need that?
def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and \
           ref_url.netloc == test_url.netloc


# To be used
# def get_redirect_target():
#     for target in request.values.get('next'), request.referrer:
#         if not target:
#             continue
#         if is_safe_url(target):
#             return target
#
# def redirect_back(endpoint, **values):
#     target = request.form['next']
#     if not target or not is_safe_url(target):
#         target = url_for(endpoint, **values)
#     return redirect(target)

@login_manage.user_loader
def load_user(user_id):
    try:
        user = User.query.get(int(user_id))
    except:
        user = None
    return user


@app.route("/login/", methods=['GET', 'POST'])
def login():
    form = LoginForm(request.form)
    if request.method == 'POST' and form.validate():
        user = User.query.filter_by(username=form.username.data).first()
        print('fetch user', user)
        if user == None:
            flash("Invalid username or password!")
            return render_template('login.html', form=form)
        print('Successfully login!', user)
        print('remember me:', form.remember_me.data)
        if form.remember_me.data == None or int(form.remember_me.data) == 0:
            remember_me = False
        else:
            remember_me = True

        login_user(user, remember=remember_me)

        next = request.args.get('next')
        if not is_safe_url(next):
            return abort(400)
        return redirect(next or url_for('index'))

    else:
        return render_template('login.html', form=form)
    pass


@app.route("/register/", methods=['GET', 'POST'])
def register():
    form = RegistrationForm(request.form)
    if request.method == 'POST' and form.validate():
        user = User(form.username.data, form.password.data, form.email.data)
        print('get new user!', user)
        db.session.add(user)
        db.session.commit()
        flash('Thanks for registering!')
        return redirect(url_for('index'))
    return render_template('register.html', form=form)
    pass


@app.route("/logout/", methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))
    pass


@app.route("/")
@login_required
def index():
    return render_template('index.html')


@app.route("/upload_train/", methods=["GET", "POST"])
@login_required
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

    user_dir = current_user.username
    print('current user:', user_dir)

    user_upload_dir = os.path.join(UPLOAD_DIR, user_dir)
    if not os.path.exists(user_upload_dir):
        os.makedirs(user_upload_dir)

    user_train_dir = os.path.join(TRAIN_DIR, user_dir)
    if not os.path.exists(user_train_dir):
        os.makedirs(user_train_dir)

    # If user does not select file, browser would also submit a empty part
    # without filename
    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)

    if not file:
        return redirect(request.url)
    print('selected files:',file)
    print('multiple file?',request.files.getlist('file'))
    for f in request.files.getlist('file'):
        if f.filename.endswith(".zip"):
            f.save(os.path.join(user_upload_dir, secure_filename(f.filename)))
            app.config["TRAIN_PREFIX"].append(f.filename[:-4])

    for f in app.config["TRAIN_PREFIX"]:
        zip_ref = zipfile.ZipFile(os.path.join(user_upload_dir, f+'.zip'), "r")
        zip_ref.extractall(user_train_dir)
        print("path",user_train_dir,f)

        # Here we only select the first zip file we found
        # In the future, users should be able to select a particular training batch
        #break
    print('app.config TRAIN_PREFIX:', app.config["TRAIN_PREFIX"])

    if len(app.config["TRAIN_PREFIX"])==0:
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

    if not os.path.exists(os.path.join(user_train_dir, app.config["TRAIN_PREFIX"][-1], "wav")):
        print(os.path.join(user_train_dir, app.config["TRAIN_PREFIX"][-1], "wav"))
        flash("You do not have a wav folder")
        return redirect(request.url)

    if not os.path.exists(os.path.join(user_train_dir, app.config["TRAIN_PREFIX"][-1], "label")):
        flash("You do not have a label folder")
        return redirect(request.url)

    # TODO: check how many training files were uploaded and if filenames of wav and labels match each other

    num_train = len(os.listdir(os.path.join(user_train_dir, app.config["TRAIN_PREFIX"][-1], "wav")))
    flash("Successfully added " + str(num_train) + " training files")
    print("Successfully added " + str(num_train) + " training files")

    return redirect(url_for("train"))


@app.route("/train/", methods=["GET", "POST"])
@login_required
def train():
    if request.method == "GET":
        return render_template("train.html")
    t1 = time.time()
    user_dir = current_user.username
    user_train_dir = os.path.join(TRAIN_DIR, user_dir)
    if not os.path.exists(user_train_dir):
        os.makedirs(user_train_dir)
    # TODO: Have users input num_train and batch_size
    num_train = len(os.listdir(os.path.join(user_train_dir, app.config["TRAIN_PREFIX"][-1], "wav")))
    batch_size = 7  # Hard code for my debugging dataset for now
    min_epochs = 1
    max_epochs = 10

    user_exp_dir = os.path.join(app.config['EXP_DIR'], user_dir,app.config["TRAIN_PREFIX"][-1])
    print('train dir',user_exp_dir)
    na_corpus = corpus.ReadyCorpus(os.path.join(user_train_dir, app.config["TRAIN_PREFIX"][-1]))
    na_reader = corpus_reader.CorpusReader(na_corpus, num_train=num_train, batch_size=batch_size)
    model = rnn_ctc.Model(user_exp_dir, na_reader, num_layers=2, hidden_size=250)
    model.train(min_epochs=min_epochs, max_epochs=max_epochs)
    t2 = time.time()
    print("\nTraining completed")
    flash("Completed training on " + app.config["TRAIN_PREFIX"][-1] + " dataset")
    t_time = t2 - t1
    return render_template("train_complete.html", time=t_time)


@app.route("/upload_transcribe/", methods=["GET", "POST"])
@login_required
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
    user_dir = current_user.username
    user_tran_dir = os.path.join(TRANSCRIBE_UPLOAD_DIR, user_dir)
    if not os.path.exists(user_tran_dir):
        os.makedirs(user_tran_dir)

    transcrib_dir = os.path.join(TRANSCRIBE_DIR, user_dir)
    if not os.path.exists(transcrib_dir):
        os.makedirs(transcrib_dir)

    for f in request.files.getlist('file'):
        if f.filename.endswith(".zip"):
            f.save(os.path.join(user_tran_dir, secure_filename(f.filename)))
            app.config["TRANSCRIBE_PREFIX"].append(f.filename[:-4])

    for f in app.config["TRANSCRIBE_PREFIX"]:

        zip_ref = zipfile.ZipFile(os.path.join(user_tran_dir, f+'.zip'), "r")
        zip_ref.extractall(transcrib_dir)
        # Here we only select the first zip file we found
        # In the future, users should be able to select a particular training batch
        #break

    if len(app.config["TRANSCRIBE_PREFIX"])==0:
        flash("Zip file not found")
        return redirect(request.url)
    print('app config transcrip',app.config['TRANSCRIBE_PREFIX'])
    # Check if wav and label folders are uploaded

    if not os.path.exists(os.path.join(transcrib_dir, app.config["TRANSCRIBE_PREFIX"][-1], "wav")):
        flash("You do not have a wav folder")
        return redirect(request.url)

    num_transcribe = len(os.listdir(os.path.join(transcrib_dir, app.config["TRANSCRIBE_PREFIX"][-1], "wav")))
    flash("Successfully added " + str(num_transcribe) + " untranscribed files")
    print("Successfully added " + str(num_transcribe) + " untranscribed files")

    # fetch trained model
    user_model_dir = os.path.join(EXP_DIR, user_dir)
    models = []
    if os.path.exists(user_model_dir):
        for f in os.listdir(user_model_dir):
            if f.startswith('.'):
                continue
            models.append(str(f))
        print('trained mode;s:',models)
    else:
        models=None
    return render_template('transcribe.html',models=models)


@app.route("/transcribe/", methods=["GET", "POST"])
@login_required
def transcribe():
    if request.method == "GET":
        return render_template("transcribe.html")

    user_dir = current_user.username
    user_train_dir = os.path.join(TRAIN_DIR, user_dir)
    if not os.path.exists(user_train_dir):
        os.makedirs(user_train_dir)


    # TODO: Have users input num_train and batch_size
    batch_size = 64  # Hard code for my debugging dataset for now

    selected_model=request.form['model_dict']
    print('selected model:',selected_model)

    # label_file_path = "./_train/na_train_tiny/phoneme_set.txt"
    label_file_path = os.path.join(user_train_dir, selected_model, 'phoneme_set.txt')

    transcrib_dir = os.path.join(TRANSCRIBE_DIR, user_dir)
    if not os.path.exists(transcrib_dir):
        os.makedirs(transcrib_dir)

    trans_exp_dir = os.path.join(TRANSCRIBE_EXP_DIR, user_dir)

    na_corpus = corpus.ReadyCorpus(os.path.join(transcrib_dir, app.config["TRANSCRIBE_PREFIX"][-1]),
                                   label_file_path=label_file_path, transcribe_new=True)
    na_reader = corpus_reader.CorpusReader(na_corpus, batch_size=batch_size, transcribe_new=True)
    model = rnn_ctc.Model(trans_exp_dir, na_reader, num_layers=2, hidden_size=250)

    model_restore_path = os.path.join(app.config['EXP_DIR'], user_dir, selected_model,'model/model_best.ckpt')
    model.transcribe(restore_model_path=model_restore_path)
    results = []
    with open(os.path.join(trans_exp_dir, "transcriptions", "hyps.txt"), 'r') as f:
        i = 1
        for line in f.readlines():
            line = line.strip()
            if i%3==1:
                results.append(('Transcrib file ' + str((i//3)+1) + " path: " + line,False))
            elif i%3==2:
                results.append((line,True))
            i += 1

    return render_template('transcript_complete.html', result=results)


@app.route("/upload_glossing_train/", methods=['GET', 'POST'])
@login_required
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
    user_dir = current_user.username
    user_upload_gloss = os.path.join(UPLOAD_DIR_GLOSSING, user_dir)
    if not os.path.exists(user_upload_gloss):
        os.makedirs(user_upload_gloss)

    train_gloss_dir = os.path.join(TRAIN_DIR_GLOSSING, user_dir)
    if not os.path.exists(train_gloss_dir):
        os.makedirs(train_gloss_dir)

    file.save(os.path.join(user_upload_gloss, secure_filename(file.filename)))

    for f in os.listdir(user_upload_gloss):
        if f.endswith(".zip"):
            zip_ref = zipfile.ZipFile(os.path.join(user_upload_gloss, f), "r")
            zip_ref.extractall(train_gloss_dir)

            break

    # Check if the uploaded zip contains the correct files
    #print(os.path.join(train_gloss_dir, 'phoneme.txt'))
    if not os.path.exists(os.path.join(train_gloss_dir, 'phoneme.txt')):
        flash("You do not have a phoneme.txt")
        return redirect(request.url)
    if not os.path.exists(os.path.join(train_gloss_dir, 'translation.txt')):
        flash("You do not have a translation.txt")
        return redirect(request.url)

    flash("Training data uploaded Successful")
    return redirect(url_for("train_glossing"))


@app.route("/train_glossing/", methods=["GET", "POST"])
@login_required
def train_glossing():
    if request.method == "GET":
        return render_template("train_glossing.html")
    user_dir = current_user.username
    train_gloss_dir = os.path.join(TRAIN_DIR_GLOSSING, user_dir)
    phoneme_path=os.path.join(train_gloss_dir,'phoneme.txt')
    translate_path=os.path.join(train_gloss_dir,'translation.txt')
    cmd = 'bash run_moses.sh {} {} {}'.format(phoneme_path,translate_path, user_dir)
    os.system(cmd)

    print("\nTraining completed")
    flash("Completed extracting dictionary")

    return render_template("glossing_train_complete.html")


@app.route("/upload_glossing/", methods=['GET', 'POST'])
@login_required
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
    user_dir = current_user.username
    user_upload_gloss = os.path.join(UPLOAD_DIR_GLOSSING, user_dir)
    if not os.path.exists(user_upload_gloss):
        os.makedirs(user_upload_gloss)

    file.save(os.path.join(user_upload_gloss, secure_filename(file.filename)))

    gloss_dir = os.path.join(GLOSS_DICT_DIR, user_dir,'working')
    if not os.path.exists(gloss_dir):
        os.makedirs(gloss_dir)

    output = []
    # Load the trained dictionary to generate glossing: create a dropdown list for user to choose a trained dictionary
    gloss_dictionary_list = []
    for dict_name in os.listdir(gloss_dir):
        if dict_name.endswith(".txt"):
            gloss_dictionary_list.append(os.path.join(gloss_dir, dict_name))

    return render_template("upload_glossing.html", gloss_dict=gloss_dictionary_list)


@app.route("/suggest_glossing/", methods=['GET', 'POST'])
@login_required
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

    # Get user's selected dict
    selected_dict = request.form['gloss_dict']
    # print(selected_dict)
    gloss_dictionary = json.load(open(selected_dict))
    # print(gloss_dictionary)

    user_dir = current_user.username
    user_upload_gloss = os.path.join(UPLOAD_DIR_GLOSSING, user_dir)
    if not os.path.exists(user_upload_gloss):
        os.makedirs(user_upload_gloss)

    with open(os.path.join(user_upload_gloss, 'new_phoneme.txt')) as f:
        for original_sent in f:
            original_sent = original_sent.strip().split(" ")
            output_gloss = []
            for phoneme_group in original_sent:
                try:
                    gloss = gloss_dictionary[phoneme_group]
                except:
                    gloss = [["NA","NA"]]
                output_gloss.append((phoneme_group, gloss))
               

            output.append(output_gloss)

    #print(type(output))
    return render_template("suggest_glossing.html", gloss_generated=output)

@app.route("/trained_model/", methods=['GET', 'POST'])
@login_required
def view_trained_model():
    user_dir=current_user.username
    user_model_dir=os.path.join(EXP_DIR,user_dir)
    if os.path.exists(user_model_dir):
        model_dict={}
        for f in os.listdir(user_model_dir):
            if f.startswith('.'):
                continue
            key=str(f)
            model_path=os.path.join(user_model_dir,key)
            with open(os.path.join(model_path,'best_scores.txt')) as f1:
                value=f1.readlines()[0].split('.',1)[1].strip()
                value="Model performance: "+value
            model_dict[key]=value
        model_dict=model_dict.items()
    else:
        model_dict=None
    return render_template('trained_model.html',models=model_dict)
    pass

@app.route("/download/", methods=['GET', 'POST'])
def download():
    if request.method=='GET':
        file_content=request.args.get('origin')
        tmp=ast.literal_eval(request.args.get('raw'))
        print('get method:',file_content)
    else:
        file_content = request.values.get('trans')
        tmp=ast.literal_eval(request.values.get('raw'))
        print('post method',file_content)
    user_dir = current_user.username
    user_download_dir = os.path.join(USER_DOWNLOAD_DIR, user_dir)
    if not os.path.exists(user_download_dir):
        os.makedirs(user_download_dir)
    new_filename=str(datetime.now())
    idx=new_filename.rfind('.')
    if idx!=-1:
        new_filename=new_filename[:idx]
    new_filename+='.txt'
    with open(os.path.join(user_download_dir,new_filename),'w')as f:
        f.write(file_content)
    flash("Transcript download to {}".format(os.path.join(user_download_dir,new_filename)))
    return render_template('transcript_complete.html',result=tmp)
    pass

if __name__ == "__main__":
    if "--setup" in sys.argv:
        with app.app_context():
            db.create_all()
            db.session.commit()
            print("Database tables created!")
    app.run(debug=True, host='0.0.0.0', port=3000)
