import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Hàm loadImage, sử dụng OpenCV để đọc ảnh và xử lý các trường hợp đặc biệt.
def loadImage(img_file):
    img = cv2.imread(img_file)  # Đọc ảnh bằng OpenCV (trả về BGR)
    if img is None:
        raise FileNotFoundError(f"Image file {img_file} not found.")
    
    # Chuyển đổi ảnh grayscale sang RGB nếu cần
    if len(img.shape) == 2:  # Nếu là ảnh grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Loại bỏ kênh alpha nếu là ảnh RGBA
    if img.shape[2] == 4:  # Nếu là ảnh RGBA
        img = img[:, :, :3]
    
    return np.array(img)  # Chuyển về dạng numpy array

# Lớp ImageDataset với đa luồng để tăng tốc quá trình load ảnh.
class ImageDataset:
    def __init__(self, image_folder, num_workers=4):
        """
        image_folder: Thư mục chứa ảnh
        num_workers: Số lượng luồng để tải ảnh song song
        """
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'jpeg', 'png'))]
        self.num_workers = num_workers
        self.cache = [None] * len(self.image_files)  # Cache để lưu ảnh đã load

    def __len__(self):
        """Trả về số lượng ảnh trong dataset."""
        return len(self.image_files)

    def load_image(self, index):
        """Load ảnh ở chỉ số index và lưu vào cache nếu chưa có."""
        if self.cache[index] is None:  # Nếu ảnh chưa được load
            img_file = self.image_files[index]
            img_path = os.path.join(self.image_folder, img_file)
            image = loadImage(img_path)  # Gọi hàm loadImage để đọc ảnh
            self.cache[index] = image  # Lưu ảnh vào cache
        return self.cache[index], self.image_files[index]  # Trả về ảnh và tên file

    def __getitem__(self, index):
        """Truy xuất ảnh bằng chỉ số (index)."""
        if index < 0 or index >= len(self.image_files):
            raise IndexError("Index out of bounds")
        return self.load_image(index)

    def prefetch_images(self):
        """Load trước các ảnh bằng đa luồng để tăng tốc."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.load_image, idx) for idx in range(len(self.image_files))]
            for future in as_completed(futures):
                future.result()  # Chờ tất cả các ảnh được tải xong

# # Ví dụ sử dụng
# if __name__ == "__main__":
#     image_folder = "/path/to/images"  # Đường dẫn đến thư mục chứa ảnh
#     dataset = ImageDataset(image_folder, num_workers=8)  # Tạo dataset với 8 luồng
    
#     # Tiền tải toàn bộ ảnh
#     dataset.prefetch_images()

#     # Truy xuất ảnh theo chỉ số
#     img, filename = dataset[0]  # Lấy ảnh đầu tiên
#     print(f"Đã load ảnh: {filename}, Kích thước: {img.shape}")
