from mobilenetv2_model import MobileNetV2
from rpn import AnchorsGenerator
import torchvision


def create_model(num_classes):
    backbone1 = MobileNetV2(weights_path="./mobilenet_v2.pth").features
    backbone1.out_channels = num_classes
    anchor_generator = AnchorsGenerator(size=(32, 3, 128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(output_size=[7, 7], sampling_ratio=2)
def train_one_epic():
    pass