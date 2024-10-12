#!/bin/bash
apt update
apt install unzip
pip install kaggle
# Thiết lập biến môi trường cho Kaggle API
export KAGGLE_USERNAME="vniptrn"
export KAGGLE_KEY="a314214296e4baba214ded52c8bd2ae9"

# Tên dataset trên Kaggle
DATASET="mduy3107/kf-full-l3"
# kaggle datasets download -d mduy3107/kf-full-l3
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
