#!/bin/bash

# 1. Kiểm tra tham số đầu vào: Tên file zip
if [ "$#" -ne 1 ]; then
    echo "Sử dụng: $0 <tên_file_zip>"
    exit 1
fi

# 2. Tên file zip cần upload
ZIP_FILE=$1

# 3. Kiểm tra xem file có tồn tại không
if [ ! -f "$ZIP_FILE" ]; then
    echo "File không tồn tại: $ZIP_FILE"
    exit 1
fi

# 4. Kiểm tra nếu lệnh 'gdrive' đã được cài đặt
if ! [ -x "$(command -v gdrive)" ]; then
    echo "Công cụ gdrive chưa được cài đặt. Đang cài đặt gdrive..."

    # Tải và cài đặt gdrive cho hệ điều hành Linux 64-bit
    wget -O gdrive https://github.com/prasmussen/gdrive/releases/download/2.1.0/gdrive-linux-x64
    chmod +x gdrive
    sudo mv gdrive /usr/local/bin/gdrive

    # Kiểm tra cài đặt thành công
    if ! [ -x "$(command -v gdrive)" ]; then
        echo "Cài đặt gdrive thất bại"
        exit 1
    fi
fi

# 5. Tải file lên Google Drive
echo "Đang tải file $ZIP_FILE lên Google Drive..."
gdrive upload $ZIP_FILE

# 6. Kiểm tra kết quả tải lên
if [ $? -eq 0 ]; then
    echo "Tải lên thành công: $ZIP_FILE"
else
    echo "Tải lên thất bại"
    exit 1
fi
