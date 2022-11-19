from torch.utils.data import Dataset
import os
from lxml import etree
import json
from PIL import Image
import torch


class VOCdataset(Dataset):
    def __init__(self, transmforms=None, txt_name="train.txt"):
        self.VOC_path = "./VOCdevkit/VOC2012"
        assert os.path.exists(self.VOC_path), f"can't find DataSet at {os.getcwd()}"
        txt_path = os.path.join(self.VOC_path, "ImageSets", "Main", txt_name)
        xml_path = os.path.join(self.VOC_path, "Annotations")
        with open(txt_path) as txt:
            self.xml_path_list = [os.path.join(xml_path, name.strip() + '.xml')
                                  for name in txt.readlines() if len(name.strip()) > 0]

        json_file = './pascal_voc_classes.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file, 'r') as f:
            self.class_dict = json.load(f)
        self.transforms = transmforms

    def __len__(self):
        return len(self.xml_path_list)

    def __getitem__(self, idx):
        xml_path = self.xml_path_list[idx]
        with open(xml_path) as x:
            xml_str = x.read()
        xml_etree = etree.fromstring(xml_str)
        xml_dic = self.parse_xml_to_dict(xml_etree)
        data = xml_dic["annotation"]
        jpg_path = os.path.join(self.VOC_path, "JPEGImages", data["filename"])
        image = Image.open(jpg_path)
        assert image.format == "JPEG", f"{jpg_path} is not JPEG file"
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            ymin = float(obj["bndbox"]["ymin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(self.class_dict[obj["name"]]))
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def collate_fn(self, batch):
        return tuple(zip(*batch))
