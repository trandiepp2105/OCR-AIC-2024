#!/bin/bash

apt update
apt install unzip
pip install kaggle
# Thiết lập biến môi trường cho Kaggle API
export KAGGLE_USERNAME="vniptrn"
export KAGGLE_KEY="cd706d0fd387738b8787d699646b4caf"

# Tên dataset trên Kaggle
DATASET="hkhnhduy/ocr-kf"

# Thư mục lưu dataset
DEST_DIR="/keyframes"

# Tạo thư mục nếu chưa có
mkdir -p $DEST_DIR

# Tải dataset từ Kaggle
kaggle datasets download -d $DATASET -p $DEST_DIR

# Giải nén các file zip trong thư mục đích
unzip $DEST_DIR/*.zip -d $DEST_DIR

# Xóa các file zip sau khi giải nén
rm $DEST_DIR/*.zip

echo "Dataset đã được tải về và giải nén thành công!"
