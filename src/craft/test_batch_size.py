import torch
import os
from load_model import load_craft_model
from craft_predict import predict_craft, predict_craft_batch
def test_batch_size(net, video_path, cuda_state, max_batch_size=1024, start_batch_size=1, increment=2):
    """
    Kiểm tra batch size lớn nhất mà GPU có thể xử lý được.
    
    net: model CRAFT đã load
    video_path: đường dẫn tới video chứa các frame
    cuda_state: True nếu sử dụng CUDA, False nếu không
    max_batch_size: Kích thước batch lớn nhất sẽ thử (mặc định 64)
    start_batch_size: Bắt đầu từ batch size nào (mặc định là 1)
    increment: Mức tăng batch size sau mỗi lần thử (mặc định là 1)
    """
    # Lấy danh sách các frame từ video path
    image_paths = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.jpg')]
    
    best_batch_size = start_batch_size  # Batch size lớn nhất có thể xử lý
    current_batch_size = start_batch_size
    successful = True
    
    while successful and current_batch_size <= max_batch_size:
        print(f"Testing with batch size: {current_batch_size}")
        
        try:
            # Lấy batch các hình ảnh theo current_batch_size
            image_batch = image_paths[:current_batch_size]
            
            # Dự đoán theo batch
            boxes_craft = predict_craft_batch(net, image_batch, text_threshold=0.65, cuda_state=cuda_state)
            
            # In kết quả của batch
            print(f"Batch size {current_batch_size} processed successfully.")
            
            # Nếu thành công, cập nhật best_batch_size
            best_batch_size = current_batch_size
            current_batch_size *= increment  # Tăng batch size lên
            
        except RuntimeError as e:
            # Nếu gặp lỗi liên quan đến bộ nhớ CUDA, dừng thử nghiệm
            print(f"Error with batch size {current_batch_size}: {e}")
            successful = False
    
    print(f"Largest batch size successfully processed: {best_batch_size}")
    return best_batch_size

if __name__ == "__main__":
    torch.cuda.empty_cache()
    image_root_path = "/keyframes/L25_V004"
    net = load_craft_model(use_cuda=True)
    test_batch_size(net, image_root_path, True)