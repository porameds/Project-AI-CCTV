import os
import argparse
from glob import glob
import supervision as sv
import yaml


"""ใช้อันนี้ 
python3 /home/smart/Project-AI-CCTV/Test_code/step_1_coco_to_yolo11_and_split.py \
--images_directory_path="/home/smart/Project-AI-CCTV/dataset/dataset_oven_123/images/default" \
--annotations_path="/home/smart/Project-AI-CCTV/dataset/dataset_oven_123/annotations/instances_default.json" \
--output_path="/home/smart/Project-AI-CCTV/coco_to_yolo11/yolo11_oven_machine_b_123" \
--train_split_ratio=0.8 \
--valid_split_ratio=0.5 \
--clean_data=True 

"""


def find_file_by_name(root_dir, filename_to_find):
    for dirpath, _, filenames in os.walk(root_dir):
        if filename_to_find in filenames:
            return os.path.join(dirpath, filename_to_find)
    return None

def clean_data(output_path):
    print("[INFO] Starting data cleaning...")
    text_files = glob(f"{output_path}/**/*/*.txt", recursive=True)

    for txt_path in text_files:
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        img_path = find_file_by_name(output_path, f"{base_name}.png")

        with open(txt_path, "r") as f:
            content = f.read()
            if not content:
                os.remove(txt_path)
                if img_path:
                    os.remove(img_path)
                print(f"[CLEANED] Removed: {txt_path}, {img_path}")

    print("[SUCCESS] Data cleaning completed.")

def split_data(images_directory_path, annotations_path, train_split_ratio, valid_split_ratio, output_path):
    print("[INFO] Starting dataset loading and splitting...")

    ds = sv.DetectionDataset.from_coco(
        images_directory_path=images_directory_path,
        annotations_path=annotations_path
    )

    train_ds, temp_ds = ds.split(split_ratio=train_split_ratio, random_state=42, shuffle=True)
    val_ds, test_ds = temp_ds.split(split_ratio=valid_split_ratio, random_state=42, shuffle=True)

    print(f"[SUCCESS] Split complete - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    print("[INFO] Exporting YOLO formatted datasets...")
    train_ds.as_yolo(
        images_directory_path=f"{output_path}/train/images",
        annotations_directory_path=f"{output_path}/train/labels",
        data_yaml_path=f"{output_path}/data.yaml"
    )
    val_ds.as_yolo(
        images_directory_path=f"{output_path}/valid/images",
        annotations_directory_path=f"{output_path}/valid/labels",
        data_yaml_path=f"{output_path}/data.yaml"
    )
    test_ds.as_yolo(
        images_directory_path=f"{output_path}/test/images",
        annotations_directory_path=f"{output_path}/test/labels",
        data_yaml_path=f"{output_path}/data.yaml"
    )

    print("[INFO] Fixing YAML paths...")
    with open(f"{output_path}/data.yaml", "r") as f:
        data = yaml.safe_load(f)

    data["train"] = "../train/images"
    data["val"] = "../valid/images"
    data["test"] = "../test/images"

    with open(f"{output_path}/data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print("[SUCCESS] Export and YAML path fixing complete.")

def main():
    parser = argparse.ArgumentParser(description="Convert COCO dataset to YOLO and split.")
    parser.add_argument("--images_directory_path", type=str, default="D:/projects/dataset/Shotcut/LPI/lpi_resize_test/images/Train")
    parser.add_argument("--annotations_path", type=str, default="D:/projects/dataset/Shotcut/LPI/lpi_resize_test/annotations/instances_Train.json")
    parser.add_argument("--output_path", type=str, default="tmp/lpi_resize_test_yolo_test")
    parser.add_argument("--train_split_ratio", type=float, default=0.7)
    parser.add_argument("--valid_split_ratio", type=float, default=0.5)
    parser.add_argument("--clean_data", type=str, default="False")
    
    args = parser.parse_args()

    print("[START] Processing...")

    split_data(
        images_directory_path=args.images_directory_path,
        annotations_path=args.annotations_path,
        train_split_ratio=args.train_split_ratio,
        valid_split_ratio=args.valid_split_ratio,
        output_path=args.output_path
    )

    if args.clean_data.lower() == "true":
        clean_data(args.output_path)
    else:
        print("[INFO] Clean data flag is False. Skipping cleaning step.")

    print("[STOP] Done.")

if __name__ == "__main__":
    main()
