# 代码实现 YOLOv3



## 项目概述

基于 PyTorch 实现 YOLOv3 目标检测，涉及到的概念有：iou交并比、nms非极大值抑制等。



## 项目内容

代码主要包括四个模块：dataset、darknet53、trainer和detector。

data：数据来自VOC2007，使用make_data_txt程序将voc数据转化为yolov3可用数据格式。

dataset.py：数据集整理，输出size为 image: ([3, 416, 416])  和 label: ([13, 13, 3, 8], [26, 26, 3, 8], [52, 52, 3, 8])。 其中13，26，52为三个feature_size，3对应三种anchor，8 = (1+4+3)，分别为置信度、cx_offset、cy_offset 、pw、ph和三种类别（人、车、马）。

darknet53.py：模型的网络，输入为 ([batch, 3, 416, 416])  ，输出为 ([batch, 24, 13, 13], [batch, 24, 26, 26], [batch, 24, 52, 52])。

trainer.py：训练函数，其中定义了epoch、学习率、loss函数等。

detector.py：基于训练好的网络和参数darknet_params.pt对data/valid_data中图片进行目标检测。

cfg.py：包含模型配置参数IMG_SIZE、CLASS_NUM和ANCHORS_GROUP 。

utils.py：包含make_image_data、iou和nms函数。



## 参考链接

https://www.bilibili.com/video/BV1Rf4y1n7mG

https://blog.csdn.net/weixin_44751294/article/details/117882906

https://blog.csdn.net/qq_40716944/article/details/114822515
