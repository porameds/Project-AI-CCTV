import os
import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import psycopg2
from psycopg2.extras import execute_values
import time
import onnxruntime


CONFIG = {
    "model_path": r"D:/Project_cctv/b210_expb_removeadh_clean_pass_line/weights/best.onnx",
    "input_dirs": {
        "input1": r"D:/Project_cctv/test_1/output"
        # "input2": r"D:/Smart/AI/CCTV/PPE RDEV/dataset1/test2/output"
    },
    "predict_dirs": {
        "input1": r"D:/Project_cctv/test_1/predict"
        # "input2": r"D:/Smart/AI/CCTV/PPE RDEV/dataset1/test2/predict"
    },
    "no_box_dirs": {
    "input1": r"D:/Project_cctv/test_1no_box"
    # "input2": r"D:/Smart/AI/CCTV/PPE RDEV/dataset1/test2/no_box"
    },
    "confidence_threshold": 0.5,
    "save_image": True,
    "save_video": False,
    "save_csv": True,
    "insert_db": False,
    "show_frame_predict": True
}

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

        with psycopg2.connect(
            database="computervision",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432"
        ) as conn:
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

def predict_images(input_dir, output_dir, model, name_tag):
    os.makedirs(output_dir, exist_ok=True)
    no_box_dir = CONFIG["no_box_dirs"][name_tag]
    os.makedirs(no_box_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    results_list = []
    video_writer = None

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        result = model.predict(img, device="cpu")[0]

        boxes = result.boxes
        label_set = set()
        confidences = []
        has_box = False  # ใช้ตรวจว่ามีกล่องใดผ่าน threshold ไหม

        if boxes and len(boxes.conf) > 0:
            for box, conf, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)):
                if conf >= CONFIG["confidence_threshold"]:
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{model.names[cls]} ({conf:.2f})"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    label_set.add(model.names[cls])
                    confidences.append(conf)
                    has_box = True  # อย่างน้อยมี 1 กล่องผ่าน threshold

        # ตั้งค่าธง label_flag ตามเงื่อนไข
        if {"remove_adhesive"}.issubset(label_set):
            label_flag = 1
        elif {"clean", "cart"}.issubset(label_set):
            label_flag = 2
        else:
            label_flag = 0

        detection_result = ', '.join(sorted(label_set)) if label_set else "no detections"
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        results_list.append({
            "predict_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detection_result": detection_result,
            "label_flag": label_flag,
            "avg_conf": round(avg_conf, 2),
            "file_name": img_file
        })

        # ✅ Save image:
        save_path = None
        if CONFIG["save_image"]:
            if label_flag > 0:
                save_path = os.path.join(output_dir, img_file)
            elif not has_box and label_set:  # มี label แต่ไม่มีกรอบ
                save_path = os.path.join(no_box_dir, img_file)

            if save_path:
                try:
                    cv2.imwrite(save_path, img)
                    print(f"[💾] Saved: {save_path}")
                except Exception as e:
                    print(f"[!] Failed to save image {save_path}: {e}")

        # 🔥 ลบภาพต้นฉบับหลังจาก predict เสร็จ
        try:
            os.remove(img_path)
            print(f"[🗑] Removed processed image: {img_path}")
        except Exception as e:
            print(f"[!] Failed to remove {img_path}: {e}")

        if CONFIG["show_frame_predict"]:
            cv2.imshow(f"{name_tag} Predict", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if CONFIG["save_video"]:
            if video_writer is None:
                height, width = img.shape[:2]
                video_path = os.path.join(output_dir, f"{name_tag}_predict_output.avi")
                video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))
            video_writer.write(img)

    if video_writer:
        video_writer.release()

    if CONFIG["save_csv"]:
        df = pd.DataFrame(results_list)
        csv_path = os.path.join(output_dir, f"{name_tag}_results.csv")
        df.to_csv(csv_path, index=False)

    if CONFIG["insert_db"]:
        df = pd.DataFrame(results_list)
        insert_to_postgres(df)

    print(f"[✔] Done predicting for {name_tag}")




# def main():
#     model = YOLO(CONFIG["model_path"])

#     for name_tag, input_dir in CONFIG["input_dirs"].items():
#         output_dir = CONFIG["predict_dirs"][name_tag]
#         predict_images(input_dir, output_dir, model, name_tag)

#     if CONFIG["show_frame_predict"]:
#         cv2.destroyAllWindows()

def main():
    model = YOLO(CONFIG["model_path"])

    try:
        while True:
            for name_tag, input_dir in CONFIG["input_dirs"].items():
                output_dir = CONFIG["predict_dirs"][name_tag]
                predict_images(input_dir, output_dir, model, name_tag)

            if CONFIG["show_frame_predict"]:
                cv2.destroyAllWindows()

            # ✅ หน่วงเวลาเล็กน้อย เพื่อลดการใช้ CPU
            time.sleep(2)  # ตรวจทุก 2 วินาที (ปรับตามความต้องการ)

    except KeyboardInterrupt:
        print("[⛔] Stopped by user")

if __name__ == "__main__":
    main()