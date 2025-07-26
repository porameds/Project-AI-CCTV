import json
import os
import shutil

# ====== PATH CONFIGURATION ======
old_json_path = '/home/smart/Project-AI-CCTV/dataset/dataset_oven_machine_b_1/annotations/instances_default.json'
new_json_path = '/home/smart/Project-AI-CCTV/ove_b_retrain_add_some_class/annotations/instances_default.json'
merged_json_path = '/home/smart/Project-AI-CCTV/dataset/dataset_oven_machine_b_1/annotations/merged_01_instances.json'
image_dir = '/home/smart/Project-AI-CCTV/dataset/dataset_oven_machine_b_1/images'               # ที่เก็บภาพทั้งหมด old img + new img
new_image_dir = '/home/smart/Project-AI-CCTV/ove_b_retrain_add_some_class/images/default'       # ที่เก็บภาพชุดใหม่ก่อน merge
# =================================

# โหลด JSON เดิม
with open(old_json_path) as f:
    old_data = json.load(f)

# โหลด JSON ใหม่
with open(new_json_path) as f:
    new_data = json.load(f)

def merge():
    # เริ่มจาก id ล่าสุดใน dataset เดิม
    last_image_id = max([img["id"] for img in old_data["images"]], default=0)
    last_ann_id = max([ann["id"] for ann in old_data["annotations"]], default=0)

    # สร้าง mapping เก่าใหม่
    image_id_map = {}

    # อัปเดต id ของรูปภาพใหม่และ rename ภาพให้ตรง JSON
    for i, img in enumerate(new_data["images"]):
        old_id = img["id"]
        new_id = last_image_id + i + 1
        img["id"] = new_id
        image_id_map[old_id] = new_id

        old_filename = img["file_name"]
        new_filename = f"merged_{new_id:06d}.jpg"  # เปลี่ยนชื่อไฟล์ เช่น merged_000123.jpg 0 is fill to 6 point 6 is max point d is decimal integer
        img["file_name"] = new_filename

        # ตรวจสอบว่าไฟล์ภาพเก่ามีจริงไหม แล้ว copy มาไว้ที่ image_dir พร้อมเปลี่ยนชื่อ
        src_path = os.path.join(new_image_dir, old_filename)
        dst_path = os.path.join(image_dir, new_filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Renamed: {old_filename} → {new_filename}")
        else:
            print(f"Not found: {src_path}")

    # อัปเดต id ของ annotation ให้ไม่ชนกัน และเชื่อมกับ image_id ใหม่
    for j, ann in enumerate(new_data["annotations"]):
        ann["id"] = last_ann_id + j + 1
        ann["image_id"] = image_id_map.get(ann["image_id"], ann["image_id"])

    # รวมทุกอย่าง
    old_data["images"].extend(new_data["images"])
    old_data["annotations"].extend(new_data["annotations"])

    # เขียนไฟล์รวม
    with open(merged_json_path, 'w') as f:
        json.dump(old_data, f, indent=4)

    print("\n รวม dataset สำเร็จแล้ว →", merged_json_path)

if __name__ == "__main__":
    merge()


