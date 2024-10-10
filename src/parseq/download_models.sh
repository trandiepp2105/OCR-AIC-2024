#!/bin/bash

# Cập nhật gdown và cài đặt unzip nếu chưa có
pip install gdown --upgrade

# Kiểm tra và cài đặt unzip nếu cần
if ! command -v unzip &> /dev/null
then
    echo "Unzip chưa được cài đặt. Đang tiến hành cài đặt unzip..."
    apt update
    apt install -y unzip
fi

# Tải mô hình từ Google Drive bằng gdown
echo "Đang tải mô hình..."
gdown --id 1Az9psFV6C1qiqFCL2s6DIBYRKr3Bkjd7

# Giải nén tệp weights.zip
echo "Đang giải nén tệp weights.zip..."
unzip weights.zip

# Xóa tệp weights.zip sau khi giải nén
echo "Đang xóa tệp weights.zip..."
rm -rf weights.zip

echo "Hoàn thành."
