
import torch
import torch.backends.cudnn as cudnn
from craft import CRAFT


# import handle code
from craft_predict import copyStateDict
from refinenet import RefineNet

# ==========
trained_model = './weights/detect/craft_mlt_25k.pth'
text_threshold = 0.7
low_text = 0.4
link_threshold = 0.4 
cuda = False
canvas_size = 1280
mag_ratio = 1.5
poly = False
show_time = False
test_folder = 'upload_image.jpg'
refine  = False
refiner_model = 'weights/craft_refiner_CTW1500.pth'




class CraftModelManager:
    def __init__(self, model_path, refiner_model_path=None, use_cuda=False, refine=False):
        self.model_path = model_path
        self.refiner_model_path = refiner_model_path
        self.refine = refine
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model = None
        self.refine_net = None

        # Load the models
        self.load_model()

    def load_model(self):
        # Load CRAFT model
        self.model = CRAFT()
        self.model.load_state_dict(copyStateDict(torch.load(self.model_path, map_location=self.device)))
        self.model = self.model.to(self.device)

        # Optional: Wrap model with DataParallel if using CUDA
        if self.device.type == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = False  # Disable if necessary

        self.model.eval()

        # Optionally load the RefineNet model
        if self.refine and self.refiner_model_path:
            self.refine_net = RefineNet()
            print(f'Loading refiner weights from checkpoint ({self.refiner_model_path})')
            self.refine_net.load_state_dict(copyStateDict(torch.load(self.refiner_model_path, map_location=self.device)))
            self.refine_net = self.refine_net.to(self.device)

            if self.device.type == 'cuda':
                self.refine_net = torch.nn.DataParallel(self.refine_net)

            self.refine_net.eval()

    def get_model(self):
        return self.model, self.refine_net

    

def load_craft_model():
    model_manager = CraftModelManager(
        model_path='./weights/detect/craft_mlt_25k.pth',
        # refiner_model_path='weights/craft_refiner_CTW1500.pth',
        use_cuda=False,
        refine=False
    )
    net, refine_net = model_manager.get_model()
    return net, refine_net


