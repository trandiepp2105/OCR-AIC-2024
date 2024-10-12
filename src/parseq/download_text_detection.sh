#!/bin/bash

apt update
apt install unzip
pip install kaggle
# Thiết lập biến môi trường cho Kaggle API
export KAGGLE_USERNAME="vniptrn"
export KAGGLE_KEY="a7b6bf5b5f7afa3532898b66c4c484b8"

# Tên dataset trên Kaggle
DATASET="vniptrn/detect-result-b3"
# kaggle datasets download -d vniptrn/detect-result-b3
# Thư mục lưu dataset
DEST_DIR="/text-detection"

# Tạo thư mục nếu chưa có
mkdir -p $DEST_DIR

# Tải dataset từ Kaggle
kaggle datasets download -d $DATASET -p $DEST_DIR

# # Giải nén các file zip trong thư mục đích
unzip $DEST_DIR/*.zip -d $DEST_DIR

# # Xóa các file zip sau khi giải nén
rm $DEST_DIR/*.zip

echo "Dataset đã được tải về và giải nén thành công!"
