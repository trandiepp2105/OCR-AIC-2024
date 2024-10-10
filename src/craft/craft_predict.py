
# support craft
from collections import OrderedDict
# from src.craft import craft_utils, imgproc
import craft_utils, imgproc
import cv2
import torch
from torch.autograd import Variable

canvas_size = 1280
link_threshold = 0.4 
poly = False
low_text = 0.4
mag_ratio = 1.5

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(net, image, text_threshold, cuda):
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    return polys

def predict_craft(net, image_path, text_threshold=0.65, cuda_state=False):
    image = imgproc.loadImage(image_path)
    polys = test_net(net, image, text_threshold=text_threshold, cuda=cuda_state)
    polys = [i.tolist() for i in polys]
    return polys

def test_net_batch(net, images, text_threshold, cuda):
    # Resize tất cả các ảnh trong batch và lưu tỷ lệ
    img_resized_list = []
    ratio_w_list = []
    ratio_h_list = []

    for image in images:
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
        img_resized_list.append(img_resized)
        ratio_h_list.append(1 / target_ratio)
        ratio_w_list.append(1 / target_ratio)

    # Preprocessing: chuyển đổi tất cả các ảnh trong batch thành tensor
    x_list = [torch.from_numpy(imgproc.normalizeMeanVariance(img_resized)).permute(2, 0, 1) for img_resized in img_resized_list]
    x = torch.stack([Variable(x) for x in x_list])  # Stack thành batch

    if cuda:
        x = x.cuda()

    # Forward pass cho cả batch
    with torch.no_grad():
        y, _ = net(x)  # Không cần dùng 'feature' nên bỏ qua

    # Xử lý từng ảnh trong batch
    polys_batch = []
    for i in range(len(images)):
        # Tạo score và link map cho mỗi ảnh
        score_text = y[i, :, :, 0].cpu().data.numpy()
        score_link = y[i, :, :, 1].cpu().data.numpy()

        # Post-processing để lấy bounding boxes và polys
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # Điều chỉnh tọa độ theo tỷ lệ resize
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w_list[i], ratio_h_list[i])
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w_list[i], ratio_h_list[i])

        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        polys_batch.append([poly.tolist() for poly in polys])

    return polys_batch



def predict_craft_batch(net, image_paths, text_threshold=0.65, cuda_state=False):
    # Load tất cả ảnh
    images = [imgproc.loadImage(image_path) for image_path in image_paths]

    # Gọi hàm test theo batch
    polys_list = test_net_batch(net, images, text_threshold=text_threshold, cuda=cuda_state)

    return polys_list

