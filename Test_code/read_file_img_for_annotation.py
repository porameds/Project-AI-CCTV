import os
import json

# กำหนด path
image_folder = "dataset/images"
json_file = "dataset/annotation.json"

# โหลด annotation
with open(json_file, "r") as f:
    data = json.load(f)

# เปลี่ยนชื่อภาพตาม id → id.PNG
for img_info in data["images"]:
    old_name = img_info["file_name"]
    new_name = f"{img_info['id']}.PNG"

    old_path = os.path.join(image_folder, old_name)
    new_path = os.path.join(image_folder, new_name)

    if os.path.exists(old_path):
        print(f"🔁 {old_name} → {new_name}")
        os.rename(old_path, new_path)
        img_info["file_name"] = new_name  # อัปเดตใน JSON ด้วย
    else:
        print(f"⚠️ ไม่พบไฟล์: {old_name}")

# (เลือก) บันทึก JSON กลับ
with open(json_file, "w") as f:
    json.dump(data, f, indent=2)

print("✅ เปลี่ยนชื่อและอัปเดต annotation.json เรียบร้อย")
