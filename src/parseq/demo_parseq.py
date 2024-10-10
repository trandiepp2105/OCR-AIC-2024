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
boxes = [[[1153.831298828125, 78.6703109741211], [1278.8968505859375, 78.6703109741211], [1278.8968505859375, 129.10000610351562], [1153.831298828125, 129.10000610351562]], [[1121.5562744140625, 86.73905944824219], [1155.848388671875, 86.73905944824219], [1155.848388671875, 114.97969055175781], [1121.5562744140625, 114.97969055175781]], [[1109.453125, 121.03125], [1214.346923828125, 121.03125], [1214.346923828125, 145.2375030517578], [1109.453125, 145.2375030517578]], [[30.837003707885742, 232.23622131347656], [130.09860229492188, 242.16238403320312], [126.18406677246094, 281.30780029296875], [26.92245864868164, 271.38165283203125]], [[127.08280944824219, 252.1484375], [183.56405639648438, 252.1484375], [183.56405639648438, 282.40625], [127.08280944824219, 282.40625]], [[411.5062561035156, 397.38592529296875], [459.91876220703125, 397.38592529296875], [459.91876220703125, 423.609375], [411.5062561035156, 423.609375]], [[0.0, 768.5484619140625], [76.65312194824219, 768.5484619140625], [76.65312194824219, 788.7203369140625], [0.0, 788.7203369140625]]]
img = cv2.imread('/keyframes/video_name/frame_number.jpg')
# img = cv2.resize(img, (700, 500))


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
