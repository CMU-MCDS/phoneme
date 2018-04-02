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

5. When done, exit virtual environment with
```
deactivate
```
