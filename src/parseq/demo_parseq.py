import time  # Thêm mô-đun để đo thời gian
from load_model import load_model_parseq
import numpy as np
import cv2
from PIL import Image
import os


def extract_frame_and_video_info(frame_path):
    video_name = os.path.basename(os.path.dirname(frame_path))  # Tên folder là video_name
    frame_number = os.path.splitext(os.path.basename(frame_path))[0]  # Tên file là số frame
    return video_name, frame_number

def load_images_and_boxes(video_folder, boxes_folder):
    # Lọc file ảnh (đuôi .jpg) và file box (đuôi .npy)
    frame_paths = sorted([os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.jpg')])
    box_paths = sorted([os.path.join(boxes_folder, f) for f in os.listdir(boxes_folder) if f.endswith('.npy')])
    
    return frame_paths, box_paths

def process_single_video_folder(parseq_model, video_folder, boxes_folder, batch_size = 8):
    image_paths, box_paths = load_images_and_boxes(video_folder, boxes_folder)
    print("load paths success!")
    all_sub_images = []
    image_info = []  

    # Duyệt qua tất cả các ảnh và boxes trong thư mục
    for img_idx, (img_path, box_path) in enumerate(zip(image_paths, box_paths)):
        # if img_idx > 1:
        #     break
        img = cv2.imread(img_path)
        boxes = np.load(box_path, allow_pickle=True)

        video_name, frame_number = extract_frame_and_video_info(img_path)

        for box in boxes:
            box = np.array(box, dtype='float32')
            sub_img = four_points_transform(img, box)
            sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
            sub_img = Image.fromarray(sub_img)
            
            all_sub_images.append(sub_img)
            image_info.append((video_name, frame_number))  # Lưu video_name và frame_number
        print(f"add boxes of images {frame_number} done!")
    # Chia các sub_images thành từng batch (8 hình một batch)
    
    results_per_video_frame = {}
    batch_size = 8  # Bắt đầu với batch size 8
    i = 0  # Biến để theo dõi thứ tự batch

    while i < len(all_sub_images):
        # Lấy batch hiện tại
        batch_sub_images = all_sub_images[i:i + batch_size]
        batch_info = image_info[i:i + batch_size]

        if not batch_sub_images:  # Dừng nếu không còn sub-image nào
            break

        # Đo thời gian bắt đầu dự đoán
        start_time = time.time()

        # Dự đoán theo batch
        preds = parseq_model.predict_batch(batch_sub_images)

        # Đo thời gian kết thúc dự đoán
        end_time = time.time()

        # In thông tin về batch và thời gian dự đoán
        print(f"Predict batch {i // batch_size + 1} (batch size: {batch_size})")
        print(f"Time bacth {batch_size}: {end_time - start_time:.2f} seconds")

        # Tăng batch size gấp đôi sau mỗi lần dự đoán
        batch_size *= 2
        i += batch_size
    # for i in range(0, len(all_sub_images), batch_size):
    #     # Lấy batch hiện tại
    #     batch_sub_images = all_sub_images[i:i + batch_size]
    #     batch_info = image_info[i:i + batch_size]
        
    #     # Dự đoán theo batch

    #     start_time = time.time()
        

    #     preds = parseq_model.predict_batch(batch_sub_images)

    #     end_time = time.time()
    #     print("predict batch: ", i)
    #     print(f"Batch {batch_size} time: {end_time - start_time:.2f} seconds")
        
        # Gom kết quả của batch theo video và frame
    #     for pred, (video_name, frame_number) in zip(preds[0], batch_info):
    #         key = (video_name, frame_number)
    #         if key not in results_per_video_frame:
    #             results_per_video_frame[key] = []
    #         results_per_video_frame[key].append(pred)

    # # In kết quả cho từng video và frame
    # output_data = {}
    # for (video_name, frame_number), lines in results_per_video_frame.items():
    #     boxes_path = os.path.join(boxes_folder, f"{frame_number}.npy")
    #     boxes = np.load(boxes_path, allow_pickle=True)
    #     rel = group_text_by_line(boxes, lines)
    #     output_data[frame_number] = ' '.join(rel)
    #     print(f"Video: {video_name}, Frame: {frame_number}: {' '.join(rel)}")
    # Lưu kết quả vào file JSON
    # with open(output_file_path, 'w', encoding='utf-8') as json_file:
    #     json.dump(output_data, json_file, ensure_ascii=False, indent=4)
# Hàm chuyển đổi 4 điểm sang hình chữ nhật
def four_points_transform(image, pts):
    tl, tr, br, bl = pts
    
    width_1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_1), int(width_2))
    
    height_1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_1), int(height_2))
    
    dst = np.array([
        [0, 0],
        [max_width, 0],
        [max_width, max_height],
        [0, max_height]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped

# Hàm gom nhóm các box và dự đoán theo dòng
def group_text_by_line(boxes, preds):
    boxes = np.array(boxes)
    preds = np.array(preds)
    
    sorted_indices = np.lexsort((boxes[:, 0, 0], boxes[:, 0, 1]))
    
    lines = []
    current_line = []
    
    for idx in sorted_indices:
        box = boxes[idx]
        pred = preds[idx]
        
        if len(current_line) == 0:
            current_line.append((box, pred))
        else:
            current_box_top_y = box[0][1]
            last_box_top_y = current_line[-1][0][0][1]
            last_box_bottom_y = current_line[-1][0][2][1]
            
            if abs(current_box_top_y - last_box_top_y) < 20 and current_box_top_y < last_box_bottom_y:
                current_line.append((box, pred))
            else:
                current_line.sort(key=lambda x: x[0][0][0])
                lines.append(" ".join([pred[0] for _, pred in current_line]))
                current_line = [(box, pred)]
    
    if len(current_line) > 0:
        current_line.sort(key=lambda x: x[0][0][0])
        lines.append(" ".join([pred[0] for _, pred in current_line]))
    
    return lines



# # Load mô hình Parseq
# parseq_model = load_model_parseq()
# print("Model loaded successfully")


# # Đo thời gian bắt đầu
# start_time = time.time()
# # Đọc hình ảnh và boxes
# img = cv2.imread('./test_images/L01_V001/9009.jpg')
# boxes = np.load("./result/L01_V001/9009.npy", allow_pickle=True)



# # # Chuyển đổi các sub-images
# sub_images = []
# for box in boxes:
#     box = np.array(box, dtype='float32')
#     sub_img = four_points_transform(img, box)
#     sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
#     sub_img = Image.fromarray(sub_img)
#     sub_images.append(sub_img)

# # Dự đoán theo batch
# preds = parseq_model.predict_batch(sub_images)
# # Gom các văn bản theo hàng
# lines = group_text_by_line(boxes, preds[0])


# print(f"rel: {' '.join(lines)}")

# # Đo thời gian kết thúc
# end_time = time.time()

# # In ra thời gian đã xử lý
# print(f"Total time taken: {end_time - start_time:.2f} seconds")

# image_paths, box_paths = load_images_and_boxes("./keyframes/L01_V001", "./result/L01_V001")

# video_name, frame_number = extract_frame_and_video_info(image_paths[0])
# print(f"video name: {video_name}, frame number: {frame_number}")

# Load mô hình Parseq
parseq_model = load_model_parseq()
print("Model loaded successfully")
start_time = time.time()
process_single_video_folder(parseq_model,"./keyframes/L01_V001",  "./result/L01_V001", 16)
end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")
