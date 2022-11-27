import torch
from torch import nn
from rpn import RegionProposalNetwork
from roi_head import RoIHeads
from transforms import GeneralizedRCNNTransform


class FasterRCNNBase(nn.Module):
    def __init__(self, backbone, rpn, roi_heads, transform):
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images, targets=None):
        original_image_sizes = []
        if self.training:
            assert targets is not None
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))
        images, targets = self.transform(images, targets)
        feature = self.backbone(images.tensors)#现在的image已经是ImageList对象啦
        pass



class FasterRCNN(FasterRCNNBase):
    def __init__(self, backbone, num_classes=None,
                 # transform parameter
                 min_size=800, max_size=1333,  # 预处理resize时限制的最小尺寸与最大尺寸
                 image_mean=None, image_std=None,  # 预处理normalize时使用的均值和方差
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,  # rpn中在nms处理前保留的proposal数(根据score)
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  # rpn中在nms处理后保留的proposal数
                 rpn_nms_thresh=0.7,  # rpn中进行nms处理时使用的iou阈值
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,  # rpn计算损失时，采集正负样本设置的阈值
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,  # rpn计算损失时采样的样本数，以及正样本占总样本的比例
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 # 移除低目标概率      fast rcnn中进行nms处理的阈值   对预测结果根据score排序取前100个目标
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,  # fast rcnn计算误差时，采集正负样本设置的阈值
                 box_batch_size_per_image=512, box_positive_fraction=0.25,  # fast rcnn计算误差时采样的样本数，以及正样本占所有样本的比例
                 bbox_reg_weights=None):

        out_channels = backbone.out_channels

        # 默认rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
        # 默认rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # 定义整个RPN框架
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        # 将roi pooling, box_head以及box_predictor结合在一起
        roi_heads = RoIHeads(
            # box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,  # 0.5  0.5
            box_batch_size_per_image, box_positive_fraction,  # 512  0.25
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)  # 0.05  0.5  100

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        # 对数据进行标准化，缩放，打包成batch等处理部分
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)
