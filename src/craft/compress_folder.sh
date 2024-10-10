#!/bin/bash
sudo apt update
sudo apt install zip -y

# Kiểm tra tham số đầu vào
if [ "$#" -ne 1 ]; then
    echo "Sử dụng: $0 <tên_thư_mục>"
    exit 1
fi

# Thư mục cần nén
FOLDER_TO_ZIP=$1

# Tên file zip sẽ tạo
ZIP_FILE="${FOLDER_TO_ZIP}.zip"

# Kiểm tra xem thư mục có tồn tại không
if [ ! -d "$FOLDER_TO_ZIP" ]; then
    echo "Thư mục không tồn tại: $FOLDER_TO_ZIP"
    exit 1
fi

# Nén thư mục thành file zip
echo "Đang nén thư mục $FOLDER_TO_ZIP thành file $ZIP_FILE..."
zip -r $ZIP_FILE $FOLDER_TO_ZIP

# Kiểm tra kết quả
if [ $? -eq 0 ]; then
    echo "Nén thành công: $ZIP_FILE"
else
    echo "Nén thất bại"
    exit 1
fi
