def box_area(box):
    return (box[3] - box[1]) * (box[4] - box[2])


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    x_min = max(boxes1[1], boxes2[1])  # 找出xmin中的较大的那一个
    y_min = max(boxes1[2], boxes2[2])  # 找出ymin中的较大的那一个
    x_max = min(boxes1[1], boxes2[1])  # 找出xmax中的较小的那一个
    y_max = min(boxes1[1], boxes2[1])  # 找出xmax中的较小的那一个

    inter = (x_max - x_min) * (y_max - y_min)
    union = area1 + area2 - inter
    IoU = inter / union
    return IoU
