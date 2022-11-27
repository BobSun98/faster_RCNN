import math

import torch

from mobilenetv2_model import MobileNetV2
from rpn import AnchorsGenerator
import torchvision
from faster_RCNN_framework import FasterRCNN
import transforms
from dataset import VOCdataset
import torch.utils.data.dataloader as d


def create_model(num_classes):
    backbone = MobileNetV2(weights_path="./mobilenet_v2.pth").features
    backbone.out_channels = num_classes
    anchor_generator = AnchorsGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0))
    roi_pooler = torchvision.ops.RoIAlign(output_size=[7, 7], spatial_scale=1, sampling_ratio=2)
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model


def train_one_epic(model, optimizer, data_loader, device, epoch, warmup=True):
    model.train()  # 设置为训练模式

    if epoch == 0 and warmup is True:  # warmup
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader))

        def lambda_f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_f)
    mloss = torch.zeros(1).to(device)
    for i, [images, targets] in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        print("nono")
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        mloss = (mloss * i + loss_value) / (i + 1)
        assert math.isfinite(loss_value), f"loss is infinite:{loss_value}"
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        now_lr = optimizer.param_groups[0]["lr"]

    return mloss,now_lr


def evaluate():
    pass


def main():
    num_calsses = 21
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device training")
    data_transform = {
        "train": transforms.Compose((transforms.ToTensor(), transforms.RandomHorizontalFlip(prob=0.5))),
        "val": transforms.Compose((transforms.ToTensor))
    }

    train_dataset = VOCdataset(data_transform["train"], txt_name="train.txt")
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=8,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=4,
                                                    collate_fn=train_dataset.collate_fn
                                                    )
    val_dataset = VOCdataset(data_transform["val"], txt_name="val.txt")
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=4,
                                                  collate_fn=val_dataset.collate_fn
                                                  )
    model = create_model(num_calsses)
    model.to(device)

    train_loss = []
    learning_rate = []
    val_map = []

    for param in model.backbone.parameters():  # 冻结backbone权重,进行RPN和ROIHead的预训练
        param.require_grad = False

    params = [p for p in model.parameters() if p.requires_grad is True]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    init_epoch = 5
    for epoch in range(init_epoch):
        mean_loss, lr = train_one_epic(model, optimizer, train_data_loader, device, epoch, warmup=True)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)
        coco_info = evaluate(model, val_data_loader, device=device)

        with open("./result.txt", 'a') as f:
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()] + [f"{lr:6f}"]]
            txt = f"{epoch}___{result_info}"
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # 写入 pascal mAP
    torch.save(model.state_dict(), "save_weights/model.pth")

    for name, parameter in model.backbone.named_parameters():  # 继续固定backbone底层权重
        split_name = name.split(".")[0]
        if split_name in ["0", "1", "2", "3"]:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)

    num_epoch = 20

    for epoch in range(num_epoch):
        mean_loss = train_one_epic(model, optimizer, train_data_loader, device, epoch, warmup=True)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        lr_scheduler.step()

        coco_info = evaluate(model, val_data_loader, device=device)

        with open("./result.txt", "a") as f:
            result_info = [f"{i:4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:6f}"]
            txt = f"epoch:[{epoch}]____" + result_info
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # 写入 pascal mAP

        if epoch in range(epoch + init_epoch)[-5:]:  # 保存最后5个epoch的权重
            save_files = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(save_files, "./save_weights/model-{}.pth".format(epoch))
    # plot_loss
    pass


if __name__ == "__main__":
    main()
