import os
import json
from PIL import Image

# กำหนด category
categories = [
    { "id": 1, "name": "machine open", "supercategory": "" },
	{ "id": 2, "name": "glove", "supercategory": "" },
	{ "id": 3, "name": "black glove", "supercategory": "" },
	{ "id": 4, "name": "mask", "supercategory": "" }
]

# เส้นทางโฟลเดอร์ภาพ
image_folder = 'images'
output_json = 'annotations/instances_train.json'

# Mock annotation สำหรับตัวอย่าง
# (ควรแทนที่ด้วย bbox จริงจาก labeling tool)
def get_annotations_for_image(image_id, width, height):
    return [{
        "id": image_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [50, 30, 100, 200],  # [x, y, width, height]
        "area": 100 * 200,
        "iscrowd": 0,
        "segmentation": []  # หรือ polygon ถ้ามี
    }]

images = []
annotations = []
image_id = 1
annotation_id = 1

for filename in os.listdir(image_folder):
    if filename.endswith((".jpg", ".png")):
        filepath = os.path.join(image_folder, filename)
        with Image.open(filepath) as img:
            width, height = img.size

        images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        anns = get_annotations_for_image(image_id, width, height)
        for ann in anns:
            ann["id"] = annotation_id
            annotations.append(ann)
            annotation_id += 1

        image_id += 1

coco_format = {
    "info": {
        "description": "My COCO dataset",
        "version": "1.0",
        "year": 2025
    },
    "licenses": [],
    "images": images,
    "annotations": annotations,
    "categories": categories
}

# บันทึกเป็น JSON
os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, 'w') as f:
    json.dump(coco_format, f, indent=4)

print(f"✅ บันทึก dataset ในรูปแบบ COCO ที่: {output_json}")
