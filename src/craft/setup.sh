echo "install necessary packages!"
apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libxrender1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*


echo "upgrade pip!"
pip install --upgrade pip 

echp "install requirements!"
pip install --no-cache-dir -r requirements.txt

