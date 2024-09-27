import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Bước 1: Thiết lập biến môi trường để truy cập file kaggle.json
os.environ['KAGGLE_CONFIG_DIR'] = 'D:/OCR/.kaggle'  # Thay thế bằng đường dẫn thực tế

# Bước 2: Khởi tạo API
api = KaggleApi()
api.authenticate()  # Đảm bảo bạn đã xác thực

# Bước 3: Định nghĩa tên dataset
dataset = 'hkhnhduy/kf-full'  # Tên dataset

# Bước 4: Liệt kê tất cả các file trong dataset
files = api.dataset_list_files(dataset)

# In ra tên tất cả các file trong dataset
for file_info in files:
    print(file_info.name)  # Truy cập vào thuộc tính 'name'
