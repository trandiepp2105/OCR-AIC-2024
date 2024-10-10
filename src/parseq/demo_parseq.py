from load_model import load_model_parseq
import numpy as np
import cv2
from PIL import Image

# convert sang HCN
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
        [0, max_height]], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped

parseq_model = load_model_parseq()
print("load model done")
img = cv2.imread('/keyframes/L01_V001/9359.jpg')
boxes = np.load("/detect/result/L01_V001/9359.npy", allow_pickle=True)
# # img = cv2.resize(img, (700, 500))


# Duyệt qua các box và tách vùng ảnh tương ứng
for i, box in enumerate(boxes):
    box = np.array(box, dtype='float32')
    sub_img = four_points_transform(img, box)

    # Chuyển đổi sang định dạng PIL để xử lý hoặc lưu
    sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
    sub_img = Image.fromarray(sub_img)

    pred, statis = parseq_model.predict(sub_img)
    print("pred: ", pred)
    print("statis: ", statis)

# def merge_boxes(boxes):
#     # Sắp xếp các box theo trục y (dòng văn bản)
#     boxes = sorted(boxes, key=lambda box: box[0][1])
    
#     merged_boxes = []
#     current_line = [boxes[0]]
    
#     for box in boxes[1:]:
#         # Nếu box tiếp theo có tọa độ y gần với box trước đó, ta xem nó thuộc cùng một dòng
#         if abs(box[0][1] - current_line[-1][0][1]) < 20:  # Điều chỉnh ngưỡng nếu cần
#             current_line.append(box)
#         else:
#             # Tạo một box lớn cho dòng hiện tại
#             min_x = min([b[0][0] for b in current_line])
#             max_x = max([b[1][0] for b in current_line])
#             min_y = min([b[0][1] for b in current_line])
#             max_y = max([b[2][1] for b in current_line])
#             merged_boxes.append([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
#             current_line = [box]
    
#     # Xử lý dòng cuối cùng
#     if current_line:
#         min_x = min([b[0][0] for b in current_line])
#         max_x = max([b[1][0] for b in current_line])
#         min_y = min([b[0][1] for b in current_line])
#         max_y = max([b[2][1] for b in current_line])
#         merged_boxes.append([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])

#     return merged_boxes

# # Áp dụng cho các box đã nhận diện
# merged_boxes = merge_boxes(boxes)

# for i, box in enumerate(merged_boxes):
#     box = np.array(box, dtype='float32')
#     sub_img = four_points_transform(img, box)

#     sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
#     sub_img = Image.fromarray(sub_img)

#     pred, statis = parseq_model.predict(sub_img)
#     print("pred: ", pred)
#     print("statis: ", statis)
