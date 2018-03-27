```
virtualenv server_demo_venv
source server_demo_venv/bin/activate
pip install flask==0.12.2

git clone --recurse-submodules git@github.com:CMU-MCDS/phoneme.git

cd phoneme/modules/persephone
git checkout -b mcds
git merge origin/mcds

cd ../..
git checkout -b yhl
git merge origin/yhl

cd note/20180327\ Server\ demo/
python server.py
```
