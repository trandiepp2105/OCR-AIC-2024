import os
import numpy as np
import time
import torch

from craft_predict import predict_craft_batch  # Sử dụng hàm predict_craft_batch
from load_model import load_craft_model

# Kiểm tra nếu CUDA khả dụng
use_cuda = torch.cuda.is_available()
cuda_state = use_cuda
if use_cuda:
    print("CUDA is available!")
else:
    print("CUDA is not available!")
    
# Load model
net = load_craft_model(use_cuda=use_cuda)
print("Model loaded successfully!")

# Thư mục chứa hình ảnh
image_root_path = "/keyframes"
result_root_path = "./result"
batch_size = 32  # Batch size

# Duyệt qua tất cả các folder và file trong thư mục ảnh
for video_name in os.listdir(image_root_path):
    video_path = os.path.join(image_root_path, video_name)
    
    if os.path.isdir(video_path):  # Kiểm tra nếu là folder
        start_time = time.time()  # Bắt đầu tính thời gian xử lý video
        
        # Lấy danh sách tất cả file ảnh .jpg
        frame_files = [f for f in os.listdir(video_path) if f.endswith(".jpg")]
        num_batches = len(frame_files) // batch_size + int(len(frame_files) % batch_size > 0)
        
        for batch_idx in range(num_batches):
            # Lấy các file cho batch hiện tại
            batch_files = frame_files[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_image_paths = [os.path.join(video_path, frame_file) for frame_file in batch_files]

            # Dự đoán bounding boxes cho batch
            polys_batch = predict_craft_batch(net, batch_image_paths, text_threshold=0.65, cuda_state=use_cuda)

            # Lưu kết quả cho từng ảnh trong batch
            result_folder_path = os.path.join(result_root_path, video_name)
            os.makedirs(result_folder_path, exist_ok=True)

            for i, frame_file in enumerate(batch_files):
                result_file_name = frame_file.replace(".jpg", ".npy")
                result_file_path = os.path.join(result_folder_path, result_file_name)
                np.save(result_file_path, polys_batch[i])

                print(f"Saved result for {frame_file} in {result_file_path}")

        # Kết thúc tính thời gian xử lý video
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished processing video {video_name}. Time taken: {elapsed_time:.2f} seconds")

print("All videos processed.")
