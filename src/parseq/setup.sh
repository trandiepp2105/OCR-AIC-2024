apt-get update
apt-get install -y libgl1-mesa-glx

pip install -r ./requirements.txt

pip install pip==23.2.1
pip install pytorch-lightning==1.6.5