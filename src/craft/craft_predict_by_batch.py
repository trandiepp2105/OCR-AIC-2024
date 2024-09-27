import os
from image_dataset import ImageDataset
from data_loader import DataLoader
from torchvision import transforms
import numpy as np
import torch
import craft_utils, imgproc
from skimage import io
import cv2
from load_model import load_craft_model
from imgproc import loadImage
canvas_size = 1280
link_threshold = 0.4 
poly = False
low_text = 0.4
mag_ratio = 1.5

# Sửa đổi test_net để xử lý nhiều ảnh
def test_net_batch(net, images, text_threshold, cuda, refine_net=None):
    batch_size = len(images)
    img_resized_list = []
    ratios = []

    # Resize tất cả hình ảnh
    for image in images:
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
        img_resized_list.append(img_resized)
        ratios.append((1 / target_ratio, 1 / target_ratio))  # Tỉ lệ cho điều chỉnh sau này

    # Chuyển đổi danh sách hình ảnh thành tensor
    x = np.stack([imgproc.normalizeMeanVariance(img) for img in img_resized_list])
    x = torch.from_numpy(x).permute(0, 3, 1, 2)  # [b, h, w, c] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # Forward pass
    with torch.no_grad():
        y, feature = net(x)

    # Tạo bản đồ điểm số và liên kết
    score_text = y[:, :, :, 0].cpu().data.numpy()
    score_link = y[:, :, :, 1].cpu().data.numpy()

    # Refinement nếu có
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[:, :, :, 0].cpu().data.numpy()

    results = []
    for i in range(batch_size):
        boxes, polys = craft_utils.getDetBoxes(score_text[i], score_link[i], text_threshold, link_threshold, low_text, poly)
        ratio_w, ratio_h = ratios[i]
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]
        results.append(polys)

    return results

# Dự đoán theo lô
def predict_craft_batch(net, refine_net, image_folder, text_threshold=0.65, cuda_state=False, batch_size=4):
    dataset = ImageDataset(image_folder)
    dataset.prefetch_images()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_polys = {}
    for images, filenames in dataloader:
        polys = test_net_batch(net, images, text_threshold=text_threshold, cuda=cuda_state, refine_net=refine_net)
        for filename, poly in zip(filenames, polys):
            all_polys[filename] = [i.tolist() for i in poly]

    return all_polys

# Sử dụng
if __name__ == '__main__':
    image_folder = './test_images'  # Thư mục chứa hình ảnh
    net, refine_net = load_craft_model()
    boxes_craft = predict_craft_batch(net, refine_net, image_folder=image_folder, text_threshold=0.65, cuda_state=False)
    print("craft")
    print(boxes_craft)


    # dataset = ImageDataset(image_folder)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    # for images, filenames in dataloader:
    #     for image in images:
    #         print("image: ", image)
    # image = loadImage("./test_images/3073.jpg")
    # print("image: ", image)