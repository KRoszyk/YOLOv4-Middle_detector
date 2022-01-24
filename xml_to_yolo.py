import xml.etree.ElementTree as ET
import os
from pathlib import Path

classes = ['person', 'people', 'cyclist']

images_folder = Path.cwd() / 'data/KAIST_data/'
xml_folder = Path.cwd() / 'data/annotations_xml/'
output_file = Path.cwd() / 'labels/yolo4_middle_labels.txt'

if not os.path.exists(output_file):
    open(output_file, 'w+')
else:
    f = open(output_file, 'w+')
    f.truncate(0)


def prepare_txt(img_path, xml_path, out_file):
    output = img_path
    in_file = open(xml_path, "r")
    tree = ET.parse(in_file)
    root = tree.getroot()
    for i, obj in enumerate(root.iter('object')):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls in classes and int(difficult) != 1:
            cls_id = classes.index(cls)
            box = obj.find('bndbox')
            x = int(int(box.find('x').text) * 416 / 640)
            w = int(int(box.find('w').text) * 416 / 640)
            y = int(int(box.find('y').text) * 416 / 512)
            h = int(int(box.find('h').text) * 416 / 512)
            if x < 0: x = 0
            if y < 0: y = 0
            out = (x, y, x + w, y + h)
            output += " " + ",".join([str(dim) for dim in out]) + ',' + str(cls_id)

    label_file = open(out_file, 'a')
    label_file.write(output + '\n')
    label_file.close()

    for (image_path, xml_path) in zip(list(images_folder.iterdir()), list(xml_folder.iterdir())):
        prepare_txt(image_path, xml_path, output_file)
