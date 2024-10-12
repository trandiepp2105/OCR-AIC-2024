from load_model import load_model_parseq
import cv2
import numpy as np
from PIL import Image


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


parseq_model = load_model_parseq()
print("Model loaded successfully")




img = cv2.imread("./keyframes/L01_V001/11207.jpg")
boxes = np.load("./result/L01_V001/11207.npy", allow_pickle=True)

# # # Chuyển đổi các sub-images
sub_images = []
for box in boxes:
    box = np.array(box, dtype='float32')
    sub_img = four_points_transform(img, box)
    sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
    sub_img = Image.fromarray(sub_img)
    sub_images.append(sub_img)

# Dự đoán theo batch
preds = parseq_model.predict_batch(sub_images)

print("predict: ", preds)
# img = cv2.resize(img, (700, 500))
# sub_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# sub_img = Image.fromarray(sub_img)
# pred, statistic = parseq_model.predict(sub_img)

# print(f"text: {pred} -- score: {statistic}")