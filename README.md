# Prototype usage

1. Download the git repo with submodules.
```
git clone --recurse-submodules git@github.com:CMU-MCDS/phoneme.git [target directory]
```
The `[target directory]` can be any name you want to put this git repo in. In this README, we assume it to be `phoneme`.

2. Set up the python virtual environment (ideally outside the directory `phoneme`).
```
virtualenv phoneme_env
source phoneme_env/bin/activate
pip install -r phoneme/requirements.txt
```

3. Set up the git repo for development.
```
cd phoneme/modules/persephone
git checkout -b mcds
git merge origin/mcds

cd ../..
python server.py
```

4. In browser, go to
```
http://127.0.0.1:5000
```
The training files must be prepared as a `.zip` file, which include a `wav` directory containing the wav files and a `label` directory containing the corresponding label files.
The untranscribed files must be prepared as a `.zip` file, which include a `wav` directory containing the wav files.

5. When done, exit virtual environment with
```
deactivate
```

# Potential next steps
1) Integrate the interface with Moses
2) Back-end API for transcribing new wav audio
3) Improve the design/implementation API: 
	- TODO: output the model performance and progress back to the interface
	- TODO: allow user to specify batch size, num_train, num_layers, and hidden size 
	- TODO: implement functions to check the format of the uploaded data. 
	- TODO: Consider what kind of files are supported and what if users uploaded multiple files
	- TODO: add some functions to organize the uploaded data and manage different experiments. (if user uploaded multiple batches of data, the interface needs to allow user select a particular training batch)