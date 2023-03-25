import os
import math
import xml.etree.ElementTree as ET

class_dict = {
    'person': 0,
    'horse': 1,
    'bicycle': 2,
}

xml_files = os.listdir('image_voc')
with open('data.txt', 'a') as f:
    for xml_file in xml_files:
        tree = ET.parse(os.path.join('image_voc', xml_file))
        root = tree.getroot()
        
        image_name = root.find('filename')
        class_name = root.findall('object/name')
        
        boxes = root.findall('object/bndbox')
        filename = image_name.text

        data = []
        data.append(filename)
        for cls, box in zip(class_name, boxes):
            cls = class_dict[cls.text]
            cx, cy = math.floor(
                (int(box[0].text)+int(box[2].text))/2), math.floor((int(box[1].text)+int(box[3].text))/2)
            w, h = (int(box[2].text)-int(box[0].text)), (int(box[3].text)-int(box[1].text))
            obj = f'{cls},{cx},{cy},{w},{h}'
            print(obj)
            data.append(obj)
            
        str = ''
        for i in data:
            str = str+i+','
        str = str.replace(',', ' ').strip()
        f.write(str+'\n')