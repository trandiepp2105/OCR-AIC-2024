#!/bin/bash

apt update
apt install unzip
pip install kaggle
# Thiết lập biến môi trường cho Kaggle API
export KAGGLE_USERNAME="vniptrn"
export KAGGLE_KEY="e0681eed12844367a890c54b11898ac9"

# Tên dataset trên Kaggle
DATASET="vniptrn/text-detection-batch-1"

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
