from craft_predict import predict_craft
from load_model import load_craft_model

image_path = "./test_images/truck.jpg"

net, refine_net = load_craft_model()


boxes_craft = predict_craft(net, refine_net, image_path=image_path, text_threshold=0.65, cuda_state=False)
print("craft")
print(boxes_craft)
