import numpy as np
from PIL import Image
import cv2
from enum import Enum


# Import các module cần thiết
from src.utils.four_points_transform import four_points_transform
from src.utils.encode_base64 import encode_base64

from src.yolov8.yolov8_predict import predict_yolov8
from src.craft.craft_predict import predict_craft
from src.parseq.parseq_predict import predict_parseq
from src.vietocr.vietocr_predict import predict_vietocr

from src.craft.load_model import load_model_craft
from src.parseq.load_model import load_model_parseq
from src.vietocr.load_model import load_model_vietocr
from src.yolov8.load_model import load_model_yolov8
from src.utils.convert_format_bbox import convert_craft_to_rectangle, convert_xyxy2xywh, convert_xywh2xyxy
from src.utils.pre_process import histogram_equalzed_rgb

# Hàm để thực hiện OCR


class DetectModels(Enum):
    Yolov8 = "Yolov8"
    Craft = "Craft"
    Best_Model = "Best Model"

class RecognizeModels(Enum):
    Vietocr = "Vietocr"
    Parseq = "Parseq"
def perform_ocr(image_path, model_detect_name="Craft", model_rec_name="Parseq"):
    # Load các model cần thiết
    detector = load_model_vietocr()
    parseq, img_transform = load_model_parseq(device='cpu')
    net, refine_net = load_model_craft()
    model_yolov8 = load_model_yolov8()

    # Đọc ảnh đầu vào
    main_image = cv2.imread(image_path)
    txt_content = ''

    # Phát hiện vùng chứa văn bản
    if model_detect_name == DetectModels.Craft.value:
        boxes = predict_craft(net, refine_net, image_path=image_path, text_threshold=0.65, cuda_state=False)
    elif model_detect_name == DetectModels.Yolov8.value:
        boxes = predict_yolov8(model_yolov8, image_path=image_path, text_threshold=0.5)
    elif model_detect_name == DetectModels.Best_Model.value:
        boxes_craft = predict_craft(net, refine_net, image_path=image_path, text_threshold=0.65, cuda_state=False)
        boxes_craft = [convert_craft_to_rectangle(box) for box in boxes_craft]
        boxes_yolov8 = predict_yolov8(model_yolov8, image_path=image_path, text_threshold=0.5)
        boxes = boxes_craft + boxes_yolov8

        # Áp dụng Non-Maximum Suppression (NMS)
        boxes = np.array([convert_xyxy2xywh(i) for i in boxes])
        score = [0.8] * len(boxes_craft) + [0.6] * len(boxes_yolov8)
        idx = cv2.dnn.NMSBoxes(boxes, score, score_threshold=0.4, nms_threshold=0.2)
        boxes = [convert_xywh2xyxy(i) for i in boxes[idx]]

    # Nhận diện văn bản từ các vùng chứa văn bản đã phát hiện
    for box in boxes:
        if model_detect_name == DetectModels.Craft.value:
            sub_img = four_points_transform(main_image, np.array(box, dtype='float32'))
            box = [list(map(int, i)) for i in box]
            txt_content += f"{box[0][0]},{box[0][1]},{box[1][0]},{box[1][1]},{box[2][0]},{box[2][1]},{box[3][0]},{box[3][1]}\t"
        elif model_detect_name == DetectModels.Yolov8.value or model_detect_name == DetectModels.Best_Model.value:
            txt_content += f"{box[0]},{box[1]},{box[2]},{box[3]}\t"
            sub_img = main_image[box[1]:box[3], box[0]:box[2]]

        # Chuyển đổi ảnh con sang định dạng RGB
        sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
        sub_img = Image.fromarray(sub_img)

        # Dự đoán văn bản bằng VietOCR
        if model_rec_name == RecognizeModels.Vietocr.value:
            pred, prob = predict_vietocr(detector, sub_img)
            txt_content += f"{pred}, {prob}\n"
        # Dự đoán văn bản bằng PARSEQ
        elif model_rec_name == RecognizeModels.Parseq.value:
            pred, prob = predict_parseq(parseq=parseq, img_transform=img_transform, image=sub_img, device='cpu')
            txt_content += f"{pred[0]}, {prob}\n"

    # Trả về kết quả OCR
    return txt_content

# Ví dụ sử dụng hàm
image_path = "upload_image.jpg"
ocr_result = perform_ocr(image_path, model_detect_name="Craft", model_rec_name="Vietocr")
print(ocr_result)
