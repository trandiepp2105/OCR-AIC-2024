#!/bin/bash

# Đảm bảo rằng bạn đã xuất các biến môi trường cho Kaggle
export KAGGLE_USERNAME="vniptrn"
export KAGGLE_KEY="cd706d0fd387738b8787d699646b4caf"

# Kiểm tra tham số đầu vào
if [ $# -ne 3 ]; then
    echo "Usage: $0 <folder_to_compress> <kaggle_dataset_slug> <dataset_title>"
    exit 1
fi

# Đầu vào từ dòng lệnh
FOLDER_TO_COMPRESS=$1
KAGGLE_DATASET_SLUG=$2
DATASET_TITLE=$3

# Chạy script Python để nén và tải lên Kaggle
python3 << EOF
import os
import zipfile
import kaggle
import shutil

# Nén thư mục
folder_to_compress = "$FOLDER_TO_COMPRESS"
zip_file = f"{folder_to_compress}.zip"

# Nén thư mục thành file zip
shutil.make_archive(folder_to_compress, 'zip', folder_to_compress)
print(f"Folder {folder_to_compress} compressed successfully into {zip_file}")

# Tạo tệp dataset-metadata.json
metadata = {
    "title": "$DATASET_TITLE",
    "id": f"{os.getenv('KAGGLE_USERNAME')}/{os.getenv('KAGGLE_DATASET_SLUG')}",
    "licenses": [{"name": "CC0-1.0"}]
}

with open('dataset-metadata.json', 'w') as f:
    f.write(str(metadata))

# Sử dụng Kaggle API để tải dataset lên
kaggle.api.dataset_create_new(folder=folder_to_compress, convert_to_csv=False, dir_mode="zip")
print(f"Dataset {folder_to_compress} uploaded successfully to Kaggle.")
EOF
