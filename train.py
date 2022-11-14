from mobilenetv2_model import MobileNetV2

def create_model(num_classes):
    backbone1 = MobileNetV2(weights_path="./mobilenet_v2.pth").features
    backbone1.out_channels = num_classes