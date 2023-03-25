import os
import torch
import cfg
import torchvision
import numpy as np
from utils import nms, make_image_data
from darknet53 import Darknet53
from PIL import Image, ImageDraw


class_dict = {
    0: 'person',
    1: 'horse',
    2: 'bicycle',
}


class Detector(torch.nn.Module):

    def __init__(self):
        super(Detector, self).__init__()

        self.net = Darknet53()
        self.net.load_state_dict(torch.load('darknet_params.pt'))
        self.net.eval()

    def forward(self, input, thresh, anchors, case):
        output_13, output_26, output_52 = self.net(input)

        idxs_13, vecs_13 = self._filter(output_13, thresh)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13], case)

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26], case)

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52], case)

        boxes = torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

        boxes = nms(boxes, 0.5, mode='inter')
        return boxes

    def _filter(self, output, thresh):
        output = output.permute(0, 2, 3, 1)  # N,H,W,24
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1) # N,H,W,3,8

        mask = torch.sigmoid(output[..., 0]) > thresh    # N,H,W,3
        idxs = mask.nonzero()  # 大于的阈值坐标
        vecs = output[mask]
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors, case):
        anchors = torch.Tensor(anchors)

        n = idxs[:, 0]  # 所属的图片
        a = idxs[:, 3]  # 建议框

        cy = (idxs[:, 1].float() + vecs[:, 2]) * t / case  # 原图的中心点y
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t / case  # 原图的中心点x

        w = anchors[a, 0] * torch.exp(vecs[:, 3])/case
        h = anchors[a, 1] * torch.exp(vecs[:, 4])/case

        p = vecs[:, 0]
        
        cls_p = vecs[:, 5:]
        cls_p = torch.softmax(cls_p, dim=1)
        cls_index = torch.argmax(cls_p, dim=1)

        return torch.stack([n.float(), torch.sigmoid(p), cx, cy, w, h, cls_index], dim=1)


if __name__ == '__main__':
    detector = Detector()

    IMG_DIR = "data/valid_data/images/"

    for i in os.listdir(IMG_DIR):
        img = Image.open(IMG_DIR + i)
        _img = make_image_data(IMG_DIR+i)
        w, h = _img.size[0], _img.size[1]
        case = 416 / w

        _img = _img.resize((416, 416))  # 此处要等比缩放
        _img_data = torchvision.transforms.ToTensor()(_img).unsqueeze(dim=0)

        result = detector(_img_data, 0.5, cfg.ANCHORS_GROUP, case)
        draw = ImageDraw.Draw(img)

        for rst in result:
            if len(rst) == 0:
                continue
            else:
                x1, y1, x2, y2 = rst[2]-0.5*rst[4], rst[3]-0.5*rst[5], rst[2]+0.5*rst[4], rst[3]+0.5*rst[5]
                x1, y1, x2, y2 = np.around([x1.item(), y1.item(), x2.item(), y2.item()], 1)
                print(f'{i}: 置信度：{str(rst[1].item())[:4]} 坐标点：{x1,y1,x2,y2} 类别：{class_dict[int(rst[6].item())]}')
                draw.text((x1, y1), class_dict[int(rst[6].item())]+str(rst[1].item())[:4])
                draw.rectangle((x1, y1, x2, y2), width=1, outline='red')
        img.show()
