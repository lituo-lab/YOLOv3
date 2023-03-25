import os
import cfg
import math
import numpy as np
import torchvision
from torch.utils.data import Dataset
from utils import make_image_data


class MyDataset(Dataset):

    def __init__(self, IMG_DIR, LABEL_PATH):
        self.IMG_DIR = IMG_DIR

        with open(LABEL_PATH) as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def one_hot(self, cls_num, v):
        b = np.zeros(cls_num)
        b[v] = 1.
        return b

    def __getitem__(self, index):
        labels = {}

        line = self.dataset[index]
        strs = line.split()
        _img_data = make_image_data(os.path.join(self.IMG_DIR, strs[0]))
        resize_x = cfg.IMG_WIDTH / _img_data.size[0]
        resize_y = cfg.IMG_HEIGHT / _img_data.size[1]
        
        img_data = _img_data.resize((cfg.IMG_WIDTH, cfg.IMG_HEIGHT))  # 此处要等比缩放
        img_data = torchvision.transforms.ToTensor()(img_data)

        _boxes = np.array([float(x) for x in strs[1:]])
        boxes = np.split(_boxes, len(_boxes) // 5)

        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))

            for box in boxes:
                cls, cx, cy, w, h = box
                cx, w = cx*resize_x, w*resize_x   # 跟图片一起等比缩放
                cy, h = cy*resize_y, h*resize_y   # 跟图片一起等比缩放

                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_HEIGHT)
                
                for i, anchor in enumerate(anchors):
                    anchor_area = anchor[0]*anchor[1]
                    target_area = w * h
                    iou = min(target_area, anchor_area) / max(target_area, anchor_area)
                    
                    p_w, p_h = w / (anchor[0]), h / (anchor[1])

                    if labels[feature_size][int(cy_index), int(cx_index), i][0] < iou:  # 让给iou大的目标
                        labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                            [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *self.one_hot(cfg.CLASS_NUM, int(cls))])

        return labels[13], labels[26], labels[52], img_data


if __name__ == '__main__':

    IMG_DIR = "data/train_data/images"
    LABEL_PATH = "data/train_data/data.txt"

    data = MyDataset(IMG_DIR, LABEL_PATH)

    label = data[0]

    print('13*13:', label[0].shape)
    print('26*26:', label[1].shape)
    print('52*52:', label[2].shape)
    print('image:', label[3].shape)
