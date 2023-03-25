
import torch
from PIL import Image

def make_image_data(path):
    img=Image.open(path)
    w,h=img.size[0],img.size[1]
    temp=max(h,w)
    mask=Image.new('RGB',(temp,temp),(0,0,0))
    mask.paste(img,(0,0))
    return mask

def iou(box, boxes, mode="inter"):
    cx, cy, w, h = box[2], box[3], box[4], box[5]
    cxs, cys, ws, hs = boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5]

    box_area = w * h # 最小面积
    boxes_area = ws * hs # 最大面积

    _x1, _x2, _y1, _y2 = cx - w/2, cx + w/2, cy - h/2, cy + h/2
    _xx1, _xx2, _yy1, _yy2 = cxs - ws / 2, cxs + ws / 2, cys - hs / 2, cys + hs / 2

    xx1 = torch.maximum(_x1, _xx1) # 左上角   最大值
    yy1 = torch.maximum(_y1, _yy1) # 左上角   最大值
    xx2 = torch.minimum(_x2, _xx2) # 右下角  最小值
    yy2 = torch.minimum(_y2, _yy2) # 右下角  最小值

    # 将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
    w = torch.clamp(xx2 - xx1, min=0) #
    h = torch.clamp(yy2 - yy1, min=0)

    inter = w * h

    if mode == 'inter':
        return inter / (box_area + boxes_area - inter) #交集除以并集
    elif mode == 'min':
        return inter / torch.min(box_area, boxes_area)


def nms(boxes, thresh, mode='inter'):
    args = boxes[:, 1].argsort(descending=True)
    sort_boxes = boxes[args]
    keep_boxes = []

    while len(sort_boxes) > 0:
        _box = sort_boxes[0]
        keep_boxes.append(_box)

        if len(sort_boxes) > 1:
            _boxes = sort_boxes[1:]

            _iou = iou(_box, _boxes, mode)
            sort_boxes=_boxes[_iou< thresh]

        else:
            break

    return keep_boxes


if __name__ == '__main__':

    make_image_data('images/1.jpg')
