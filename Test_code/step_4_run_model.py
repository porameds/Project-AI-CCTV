import os
import cv2
import csv
import time
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import psycopg2
from psycopg2.extras import execute_values

# CONFIG -----------------------------------------------------
CONFIG = {
    "model_base_path": r"/home/smart/Project_ai_barcode/model/Ai_barcode_model_2",
    "image_path": r"/home/smart/Project_ai_barcode/img_barcode",  # <-- Use image folder here
    "class_names": {0: "bar code"},  # แก้ class id เป็น 0 ตามโมเดลจริง
    "confidence_threshold": 0.2,
    "save_image": True,
    "save_csv": True,
    "insert_db": False,
    "show_frame_predict": True
}
# ------------------------------------------------------------

weights_path = os.path.join(CONFIG["model_base_path"], "weights", "best.onnx")
save_path_with_boxes = os.path.join(CONFIG["model_base_path"], "Capture_image_new", "Clean_with_bbx")
save_path_without_boxes = os.path.join(CONFIG["model_base_path"], "Capture_image_new", "Clean__no_bbx")
csv_output_path = os.path.join(CONFIG["model_base_path"], "result_output.csv")

os.makedirs(save_path_with_boxes, exist_ok=True)
os.makedirs(save_path_without_boxes, exist_ok=True)

def insert_to_postgres(df):
    try:
        values = [
            (
                str(row["predict_time"]),
                str(row["detection_result"]),
                int(row["label_flag"]) if row["label_flag"] is not None else None,
                float(row["avg_conf"]) if row["avg_conf"] is not None else None,
                str(row["file_name"])
            )
            for _, row in df.iterrows()
        ]
        with psycopg2.connect(database="computervision", user="postgres", password="postgres", host="localhost", port="5432") as conn:
            with conn.cursor() as cur:
                insert_query = """
                    INSERT INTO smart_ai.ppe_rdev_r21422_ul
                    (predict_time, detection_result, label_flag, avg_conf, file_name)
                    VALUES %s
                """
                execute_values(cur, insert_query, values)
                conn.commit()
                print(f"[✔] Success insert to DB")
    except Exception as e:
        print(f"[!] Failed to insert to DB: {e}")

def process_images():
    model = YOLO(weights_path, task='detect')  # กำหนด task ชัดเจน
    image_dir = CONFIG["image_path"]
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

    results_list = []

    if CONFIG["save_csv"]:
        csv_file = open(csv_output_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "detection_result", "label_flag", "avg_conf", "file_name"])
    else:
        csv_writer = None

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[!] Cannot read image: {image_path}")
            continue

        results = model.predict(frame, device='cpu')[0]
        boxes = results.boxes

        if boxes is not None and len(boxes) > 0:
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
        else:
            cls_ids = np.array([])
            confs = np.array([])

        timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Debug info
        print(f"[DEBUG] Image: {image_file}")
        print(f"→ Detected class IDs: {cls_ids}")
        print(f"→ Confidences: {confs}")
        print(f"→ Model class names: {model.names}")

        labels = [model.names[cls_id] for cls_id in cls_ids] if cls_ids.size > 0 else []

        detection_result = ", ".join(labels) if labels else "no detections"
        label_set = set(labels)
        label_flag = 1 if {"bar code"}.issubset(label_set) else 0
        clean_confs = [conf for cls_id, conf in zip(cls_ids, confs) if cls_id in CONFIG["class_names"]]
        conf_to_save = round(float(clean_confs[0]), 4) if clean_confs else None

        if CONFIG["save_csv"] and csv_writer:
            csv_writer.writerow([timestamp_now, detection_result, label_flag, conf_to_save, image_file])

        results_list.append({
            "predict_time": timestamp_now,
            "detection_result": detection_result,
            "label_flag": label_flag,
            "avg_conf": conf_to_save,
            "file_name": image_file
        })

        save_image_flag = False
        if boxes is not None and len(boxes) > 0:
            for box, conf, cls in zip(boxes.xyxy.cpu().numpy(), confs, cls_ids):
                if conf >= CONFIG["confidence_threshold"]:
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{model.names[cls]} ({conf:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    save_image_flag = True

        if CONFIG["save_image"] and save_image_flag:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            label = "_".join(set(labels)) if labels else "no_label"
            cv2.imwrite(os.path.join(save_path_with_boxes, f"{timestamp}_{label}_with_boxes.jpg"), frame)
            cv2.imwrite(os.path.join(save_path_without_boxes, f"{timestamp}_{label}_no_boxes.jpg"), frame)

        if CONFIG["show_frame_predict"]:
            cv2.imshow("Prediction", frame)
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break

    if CONFIG["save_csv"]:
        csv_file.close()
    if CONFIG["insert_db"] and results_list:
        df = pd.DataFrame(results_list)
        insert_to_postgres(df)

    cv2.destroyAllWindows()
    print("[✔] All images processed.")

def main():
    start_time = time.time()
    process_images()
    print(f"\n✅ Total runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
