import os
import cv2
import numpy as np

# Đường dẫn tới thư mục chứa kết quả .npy
result_root_path = "./result"
# Đường dẫn tới thư mục chứa ảnh gốc
image_root_path = "./test_images"

# Duyệt qua tất cả các thư mục trong result
for video_name in os.listdir(result_root_path):
    video_result_path = os.path.join(result_root_path, video_name)
    
    if os.path.isdir(video_result_path):  # Nếu là folder
        for frame_npy in os.listdir(video_result_path):
            if frame_npy.endswith(".npy"):  # Kiểm tra nếu là file .npy
                # Đọc bounding boxes từ file .npy
                npy_path = os.path.join(video_result_path, frame_npy)
                boxes = np.load(npy_path, allow_pickle=True)
                
                # Tìm tên file ảnh gốc tương ứng
                frame_jpg = frame_npy.replace(".npy", ".jpg")
                image_path = os.path.join(image_root_path, video_name, frame_jpg)
                
                # Đọc ảnh gốc
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"Image {image_path} not found!")
                    continue
                
                # Vẽ các bounding boxes lên ảnh
                for box in boxes:
                    box = np.array(box, dtype=np.int32)  # Chuyển bounding box sang kiểu int
                    cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=2)  # Vẽ bounding box
                
                # Hiển thị ảnh với bounding boxes
                cv2.imshow(f'Image with Bounding Boxes: {frame_jpg}', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
