import time  # Thêm mô-đun để đo thời gian
from load_model import load_model_parseq
import numpy as np
import cv2
from PIL import Image

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

# Đo thời gian bắt đầu
start_time = time.time()

# Load mô hình Parseq
parseq_model = load_model_parseq()
print("Model loaded successfully")

# Đọc hình ảnh và boxes
img = cv2.imread('./test_images/L01_V001/9009.jpg')
boxes = np.load("./result/L01_V001/9009.npy", allow_pickle=True)



# preds = []
# for i, box in enumerate(boxes):
#     box = np.array(box, dtype='float32')
#     sub_img = four_points_transform(img, box)
#     sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
#     sub_img = Image.fromarray(sub_img)

#     pred, statis = parseq_model.predict(sub_img)
#     preds.append(pred[0])  # Lưu kết quả dự đoán

# # Gom các văn bản theo hàng
# lines = group_text_by_line(boxes, preds)

# # In các dòng văn bản
# for line in lines:
#     print(" ".join(line))


# # Chuyển đổi các sub-images
sub_images = []
for box in boxes:
    box = np.array(box, dtype='float32')
    sub_img = four_points_transform(img, box)
    sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
    sub_img = Image.fromarray(sub_img)
    sub_images.append(sub_img)

# Dự đoán theo batch
preds = parseq_model.predict_batch(sub_images)
print("predict success!")
print("preds: ", preds)
# Gom các văn bản theo hàng
lines = group_text_by_line(boxes, preds[0])

# In các dòng văn bản
# for line in lines:
#     print("line: ", line)

print(f"rel: {' '.join(lines)}")

# Đo thời gian kết thúc
end_time = time.time()

# In ra thời gian đã xử lý
print(f"Total time taken: {end_time - start_time:.2f} seconds")
