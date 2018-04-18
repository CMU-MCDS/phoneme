from django.shortcuts import render
from phonemeapps.settings import *
from django.core.files.base import ContentFile
import zipfile
from modules.persephone.persephone import corpus, corpus_reader, rnn_ctc
# Create your views here.

def home(request):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    if not os.path.exists(EXP_DIR):
        os.makedirs(EXP_DIR)
    if not os.path.exists(TRANSCRIBE_UPLOAD_DIR):
        os.makedirs(TRANSCRIBE_UPLOAD_DIR)
    if not os.path.exists(TRANSCRIBE_DIR):
        os.makedirs(TRANSCRIBE_DIR)
    if not os.path.exists(TRANSCRIBE_EXP_DIR):
        os.makedirs(TRANSCRIBE_EXP_DIR)
    # configs["UPLOAD_DIR"] = UPLOAD_DIR
    # configs["TRAIN_DIR"] = TRAIN_DIR
    # configs["EXP_DIR"] = EXP_DIR
    # configs["TRANSCRIBE_UPLOAD_DIR"] = TRANSCRIBE_UPLOAD_DIR
    # configs["TRANSCRIBE_DIR"] = TRANSCRIBE_DIR
    # configs["TRANSCRIBE_EXP_DIR"] = TRANSCRIBE_EXP_DIR
    return render(request, "phoneme/index.html", {})
        
def upload_train(request):
    context={}
    if request.method == "GET":
        return render(request, "phoneme/upload_train.html", context)

    print("request.files =", request.FILES)
    print("request.url =", request.build_absolute_uri)

    # Check if the POST request has the file part
    if "file" not in request.FILES:
        context['message']="No file part"
        return render(request, "phoneme/upload_train.html", context)

    file = request.FILES["file"]

    # If user does not select file, browser would also submit a empty part
    # without filename
    if not file or file.name == "":
        context['message']="No selected file"
        return render(request, 'phoneme/upload_train.html', context)

    # save file
    full_path=os.path.join(UPLOAD_DIR,request.FILES['file'].name)
    file_content=ContentFile(request.FILES['file'].read())
    with open(full_path,'wb+') as f:
        for chunk in file_content.chunks():
            f.write(chunk)
    

    for f in os.listdir(UPLOAD_DIR):
        if f.endswith(".zip"):
            zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_DIR, f), "r")
            zip_ref.extractall(TRAIN_DIR)
            configs["TRAIN_PREFIX"] = f[:-4]
            # Here we only select the first zip file we found
            # In the future, users should be able to select a particular training batch
            break

    if not configs["TRAIN_PREFIX"]:
        context['message'] ="Zip file not found"
        return render(request, 'phoneme/upload_train.html', context)
    
    # Methods for check the format of the training file. User needs to upload a 
    # zipped folder that contains two folder "wav" and "label". The name and number of the wav
    # audio and labels should match. If the format does not meet the requirement, the interface
    # should return some message messages. 
    # TODO: implement functions to check the format of the uploaded data. 
    # TODO: Consider what kind of files are supported and what if users uploaded multiple files
    # TODO: add some functions to organize the uploaded data and manage different experiments. (
    # if user uploaded multiple batches of data, the interface needs to allow user select a 
    # particular training batch)

        # Check if wav and label folders are uploaded
    if not os.path.exists(os.path.join(TRAIN_DIR, configs["TRAIN_PREFIX"], "wav")):
        print(os.path.join(TRAIN_DIR, configs["TRAIN_PREFIX"], "wav"))
        context['message'] ="You do not have a wav folder"
        return render(request, 'phoneme/upload_train.html', context)

    if not os.path.exists(os.path.join(TRAIN_DIR, configs["TRAIN_PREFIX"], "label")):
        context['message'] ="You do not have a label folder"
        return render(request, 'phoneme/upload_train.html', context)

        # TODO: check how many training files were uploaded and if filenames of wav and labels match each other

    num_train = len(os.listdir(os.path.join(TRAIN_DIR, configs["TRAIN_PREFIX"], "wav")))
    context['message'] ="Successfully added " + str(num_train) + " training files"
    print("Successfully added " + str(num_train) + " training files")
    #configs['TRAIN_PREFIX']=None
    return render(request, "phoneme/train.html", context)

def train(request):
    context={}
    if request.method == "GET":
        return render(request,"phoneme/train.html",context)

    # TODO: Have users input num_train and batch_size
    num_train = len(os.listdir(os.path.join(TRAIN_DIR, configs["TRAIN_PREFIX"], "wav")))
    batch_size = 7  # Hard code for my debugging dataset for now
    min_epochs = 1
    max_epochs = 10

    na_corpus = corpus.ReadyCorpus(os.path.join(TRAIN_DIR, configs["TRAIN_PREFIX"]))
    na_reader = corpus_reader.CorpusReader(na_corpus, num_train=num_train, batch_size=batch_size)
    model = rnn_ctc.Model(EXP_DIR, na_reader, num_layers=2, hidden_size=250)
    model.train(min_epochs=min_epochs, max_epochs=max_epochs)

    print("\nTraining completed")
    context['message']="Completed training on " + configs["TRAIN_PREFIX"] + " dataset"
    return render(request,"phoneme/train_complete.html",context)

def upload_transcribe(request):
    context={}
    if request.method == "GET":
        return render(request,"phoneme/upload_transcribe.html",context)

    print("request.files =", request.FILES)
    print("request.url =", request.build_absolute_uri)

    # Check if the POST request has the file part
    if "file" not in request.FILES:
        context['message']="No file part"
        return render(request,'phoneme/upload_transcribe.html',context)

    file = request.FILES["file"]

    # If user does not select file, browser would also submit a empty part
    # without filename
    if not file or file.name == "":
        context['message']="No selected file"
        return render(request,'phoneme/upload_transcribe.html',context)

    # save file
    full_path = os.path.join(TRANSCRIBE_UPLOAD_DIR, request.FILES['file'].name)
    file_content = ContentFile(request.FILES['file'].read())
    with open(full_path, 'wb+') as f:
        for chunk in file_content.chunks():
            f.write(chunk)

    for f in os.listdir(TRANSCRIBE_UPLOAD_DIR):
        if f.endswith(".zip"):
            zip_ref = zipfile.ZipFile(os.path.join(TRANSCRIBE_UPLOAD_DIR, f), "r")
            zip_ref.extractall(TRANSCRIBE_DIR)
            configs["TRANSCRIBE_PREFIX"] = f[:-4]
            # Here we only select the first zip file we found
            # In the future, users should be able to select a particular training batch
            break

    if not configs["TRANSCRIBE_PREFIX"]:
        context['message']="Zip file not found"
        return render(request, 'phoneme/upload_transcribe.html', context)

    # Check if wav and label folders are uploaded
    if not os.path.exists(os.path.join(TRANSCRIBE_DIR, configs["TRANSCRIBE_PREFIX"], "wav")):
        context['message']="You do not have a wav folder"
        return render(request, 'phoneme/upload_transcribe.html', context)

    num_transcribe = len(os.listdir(os.path.join(TRANSCRIBE_DIR, configs["TRANSCRIBE_PREFIX"], "wav")))
    context['message'] ="Successfully added " + str(num_transcribe) + " untranscribed files"
    print("Successfully added " + str(num_transcribe) + " untranscribed files")
    return render(request, 'phoneme/transcribe.html', context)

    pass

def transcribe(request):
    context={}
    if request.method == "GET":
        return render(request,"phoneme/transcribe.html",context)

    # TODO: Have users input num_train and batch_size
    batch_size = 64  # Hard code for my debugging dataset for now

    label_file_path = "_train/na_train_tiny/phoneme_set.txt"

    na_corpus = corpus.ReadyCorpus(os.path.join(TRANSCRIBE_DIR, configs["TRANSCRIBE_PREFIX"]),
                                   label_file_path=label_file_path, transcribe_new=True)
    na_reader = corpus_reader.CorpusReader(na_corpus, batch_size=batch_size, transcribe_new=True)
    model = rnn_ctc.Model(TRANSCRIBE_EXP_DIR, na_reader, num_layers=2, hidden_size=250)
    model.transcribe(restore_model_path="_exp/model/model_best.ckpt")

    file_p=os.path.join(TRANSCRIBE_EXP_DIR, "transcriptions",'hyps.txt')
    with open(file_p,'r') as f:
        file_content=f.readlines()
        context['file_content']=file_content

    return render(request,'phoneme/transcribe_complete.html',context)
    pass

