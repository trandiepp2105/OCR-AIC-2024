import os
import numpy as np
import time  # Thêm thư viện time
from craft_predict import predict_craft
from load_model import load_craft_model

# Load model
net, refine_net = load_craft_model()

# Folder chứa hình ảnh
image_root_path = "./test_images"
result_root_path = "./result"

# Duyệt qua tất cả các folder và file trong test_images
for video_name in os.listdir(image_root_path):
    video_path = os.path.join(image_root_path, video_name)
    
    if os.path.isdir(video_path):  # Kiểm tra nếu là folder
        start_time = time.time()  # Bắt đầu đếm thời gian xử lý video
        for frame_file in os.listdir(video_path):
            if frame_file.endswith(".jpg"):  # Kiểm tra nếu là file .jpg
                frame_path = os.path.join(video_path, frame_file)
                
                # Predict bounding boxes từ ảnh frame
                boxes_craft = predict_craft(net, refine_net, image_path=frame_path, text_threshold=0.65, cuda_state=False)
                
                # Tạo đường dẫn lưu kết quả
                result_folder_path = os.path.join(result_root_path, video_name)
                os.makedirs(result_folder_path, exist_ok=True)  # Tạo folder nếu chưa tồn tại
                
                # Tạo tên file .npy từ tên file frame_number.jpg
                result_file_name = frame_file.replace(".jpg", ".npy")
                result_file_path = os.path.join(result_folder_path, result_file_name)
                
                # Lưu kết quả bounding boxes vào file .npy
                np.save(result_file_path, boxes_craft)
                
                print(f"Saved result for {frame_file} in {result_file_path}")
        
        # Kết thúc đếm thời gian xử lý video
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished processing video {video_name}. Time taken: {elapsed_time:.2f} seconds")

print("All images processed.")
