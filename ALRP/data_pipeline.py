import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

class DataPipeline:
    def __init__(self, pascal_voc_dir, output_dir):
        self.pascal_voc_dir = pascal_voc_dir
        self.output_dir = output_dir

    def convert_pascal_to_yolo_format(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()

        image_width = int(root.find('size').find('width').text)
        image_height = int(root.find('size').find('height').text)

        yolo_annotations = []

        for obj in root.iter('object'):
            class_name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            x_center = (xmin + xmax) / 2 / image_width
            y_center = (ymin + ymax) / 2 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            yolo_annotation = f'{class_name} {x_center} {y_center} {width} {height}'
            yolo_annotations.append(yolo_annotation)

        with open(os.path.join(self.output_dir, 'yolo', os.path.splitext(os.path.basename(filename))[0] + '.txt'), 'w') as f:
            f.write('\n'.join(yolo_annotations))

    def convert_pascal_to_coco_format(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()

        image_width = int(root.find('size').find('width').text)
        image_height = int(root.find('size').find('height').text)

        coco_annotations = []

        for obj in root.iter('object'):
            class_name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            coco_annotation = {
                'id': int(obj.find('id').text),
                'image_id': int(root.find('filename').text.split('.')[0]),
                'category_id': int(obj.find('id').text),
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                'area': (xmax - xmin) * (ymax - ymin),
                'iscrowd': 0
            }
            coco_annotations.append(coco_annotation)

        with open(os.path.join(self.output_dir, 'coco', os.path.splitext(os.path.basename(filename))[0] + '.json'), 'w') as f:
            import json
            json.dump(coco_annotations, f)

    def convert_all_pascal_to_yolo_and_coco_format(self):
        for filename in tqdm(os.listdir(self.pascal_voc_dir)):
            if filename.endswith('.xml'):
                self.convert_pascal_to_yolo_format(os.path.join(self.pascal_voc_dir, filename))
                self.convert_pascal_to_coco_format(os.path.join(self.pascal_voc_dir, filename))

