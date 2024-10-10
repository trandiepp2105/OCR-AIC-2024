import torch
import statistics


@torch.inference_mode()
def predict_parseq(parseq, img_transform, image, device='cuda:0'):

    # image = Image.open(image).convert('RGB')
    image = img_transform(image).unsqueeze(0).to(device)

    p = parseq(image).softmax(-1)
    pred, p = parseq.tokenizer.decode(p)

    return (pred, statistics.mean(p[0].tolist()))

import torch
import statistics

@torch.inference_mode()
def predict_parseq_batch(parseq, img_transform, images, device='cuda:0'):
    """
    Chạy dự đoán cho một batch hình ảnh.

    Args:
        parseq: Mô hình PARSEQ đã tải.
        img_transform: Hàm transform hình ảnh.
        images: Danh sách các hình ảnh PIL.
        device: Thiết bị ('cuda:0' hoặc 'cpu').

    Returns:
        Kết quả dự đoán cho từng hình ảnh trong batch.
    """
    # Chuyển đổi tất cả các hình ảnh trong batch
    transformed_images = [img_transform(image).unsqueeze(0) for image in images]
    
    # Gộp các hình ảnh lại thành một tensor duy nhất (batch)
    batch_images = torch.cat(transformed_images, dim=0).to(device)

    # Chạy dự đoán trên batch
    p = parseq(batch_images).softmax(-1)

    # Lưu kết quả dự đoán và xác suất
    predictions = []
    for i in range(p.size(0)):  # Duyệt qua từng ảnh trong batch
        pred, probabilities = parseq.tokenizer.decode(p[i])
        avg_prob = statistics.mean(probabilities[0].tolist())
        predictions.append((pred, avg_prob))
    
    return predictions




@torch.inference_mode()
def predict_parseq_author(parseq, img_transform, image) -> tuple:
    
    predict_image = img_transform(image).unsqueeze(0)
    logits = parseq(predict_image)
    pred, prob = parseq.tokenizer.decode(logits.softmax(-1))

    return (pred, statistics.mean(prob[0].tolist()))