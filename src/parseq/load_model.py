import torch
import statistics
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint

class ParseqModel:
    def __init__(self, model_path=None, use_pretrained=False, device=None):
        """
        Khởi tạo lớp ParseqModel, tải mô hình chỉ một lần và sẵn sàng sử dụng cho các lần sau.

        Args:
            model_path (str): Đường dẫn đến mô hình đã được huấn luyện. Nếu None và use_pretrained=False thì không tải mô hình.
            use_pretrained (bool): Nếu True, tải mô hình từ repository của Parseq trên torch.hub.
            device (str): 'cuda' hoặc 'cpu'. Nếu None, tự động phát hiện thiết bị.
        """
        # Tự động chọn GPU hoặc CPU nếu không chỉ định
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = None
        self.img_transform = None
        
        # Load mô hình từ checkpoint hoặc tải mô hình pretrained nếu được yêu cầu
        if model_path:
            self.load_model_from_checkpoint(model_path)
        elif use_pretrained:
            self.load_model_pretrained()

    def load_model_from_checkpoint(self, model_path):
        """
        Tải mô hình từ checkpoint đã lưu.
        """
        self.model = load_from_checkpoint(model_path).eval().to(self.device)
        self.img_transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)
    
    def load_model_pretrained(self):
        """
        Tải mô hình pretrained từ torch.hub.
        """
        self.model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval().to(self.device)
        self.img_transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)
        print(f"Pretrained model loaded on {self.device}")

    def get_model(self):
        """
        Lấy mô hình và transform hình ảnh.
        """
        return self.model, self.img_transform

    @torch.inference_mode()
    def predict(self, image):
        """
        Sử dụng mô hình đã tải để dự đoán văn bản từ hình ảnh đầu vào.

        Args:
            image: Hình ảnh đầu vào để dự đoán.

        Returns:
            Tuple (prediction, probability)
        """
        if self.model is None or self.img_transform is None:
            raise ValueError("Model and image transform must be loaded before prediction.")
        
        # Chuyển đổi hình ảnh thành định dạng đầu vào của mô hình
        transformed_image = self.img_transform(image).unsqueeze(0).to(self.device)
        
        # Dự đoán với mô hình
        with torch.no_grad():
            output = self.model(transformed_image).softmax(-1)
            pred, prob = self.model.tokenizer.decode(output)

        return pred, statistics.mean(prob[0].tolist())

    @torch.inference_mode()
    def predict_batch(self, images):
        """
        Dự đoán cho một batch các hình ảnh đầu vào.

        Args:
            images (List[Image]): Danh sách các hình ảnh đầu vào.
        
        Returns:
            List[Tuple[str, float]]: Danh sách các dự đoán và xác suất của chúng.
        """
        if self.model is None or self.img_transform is None:
            raise ValueError("Model and image transform must be loaded before prediction.")
        
        # Chuyển đổi các hình ảnh
        transformed_images = [self.img_transform(img).unsqueeze(0) for img in images]
        batch = torch.cat(transformed_images).to(self.device)
        
        # Dự đoán trên batch
        with torch.no_grad():
            output = self.model(batch).softmax(-1)
            print("output: ", output)
        preds, probs = [], []
        for out in output:
            try:
                pred, prob = self.model.tokenizer.decode(out.unsqueeze(0))
                preds.append(pred)
                probs.append(statistics.mean(prob[0].tolist()))
            except AttributeError as e:
                print(f"Error decoding output: {e}")
        
        return preds, probs


def load_model_parseq():
    parseq_model = ParseqModel(model_path='./weights/rec/best-parseq.ckpt')
    return parseq_model



