import os
import cv2
import csv
import queue
import time
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Queue, Process, set_start_method
from ultralytics import YOLO
import psycopg2
from psycopg2.extras import execute_values

# CONFIG -----------------------------------------------------
CONFIG = {
    "model_base_path": r"/home/user/Project/plasma_working_model/model/plasma_working_model",
    #"video_path": r"/home/user/Project/vdo_test/test_plasma_vdo",
    #"video_path":"rtsp://admin:Plant_1340@192.168.75.39:554/Streaming/Channels/101",
    "video_path_backup": r"/home/user/Project/test_plasma_vdo/backup",
    "class_names": { 1: "Working"},
    "predict_frame_interval": 125,
    "confidence_threshold": 0.5,
    #True or False 
    "save_image":True,
    "save_video": False,
    "save_csv": True,
    "insert_db": False,
    "show_frame_predict": True
}
# ------------------------------------------------------------

weights_path = os.path.join(CONFIG["model_base_path"], "weights", "best.onnx")
save_path_with_boxes = os.path.join(CONFIG["model_base_path"], "Capture_image_new", "Clean_with_bbx")
save_path_without_boxes = os.path.join(CONFIG["model_base_path"], "Capture_image_new", "Clean__no_bbx")
output_video_dir = os.path.join(CONFIG["model_base_path"], "capture_video_with_bbx_new")
csv_output_path = os.path.join(CONFIG["model_base_path"], "result_output.csv")

os.makedirs(save_path_with_boxes, exist_ok=True)
os.makedirs(save_path_without_boxes, exist_ok=True)
os.makedirs(output_video_dir, exist_ok=True)

def insert_to_postgres(df):
    try:
        # records = df.to_records(index=False)
        # values = [tuple(record) for record in records]
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

def connect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        exit()
    return cap

def video_inspection(video_path, q1: Queue):
    cap = connect_video(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame_resized = cv2.resize(frame, (640, 360))
        # frame_resized = cv2.resize(frame, (320, 180))
        frame_resized = cv2.resize(frame, (1280, 720))
        try:
            q1.put(frame_resized, timeout=1)
        except queue.Full:
            continue
    cap.release()

def model_plot(q1: Queue, video_path: str):
    model = YOLO(weights_path)
    filename = os.path.basename(video_path)
    video_writer = None
    results_list = []
    frame_counter = 0

    if CONFIG["save_csv"]:
        csv_file = open(csv_output_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "detection_result", "label_flag", "avg_conf", "file_name"])
    else:
        csv_writer = None

    try:
        while True:
            try:
                frame = q1.get(timeout=3)
            except queue.Empty:
                break

            frame_counter += 1
            if frame_counter % CONFIG["predict_frame_interval"] != 0:
                continue

            results = model.predict(frame, device='cpu')[0]
            boxes = results.boxes

            timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes else []
            confs = boxes.conf.cpu().numpy() if boxes else []
            labels = [model.names[cls_id] for cls_id in cls_ids] if boxes else []

            detection_result = ", ".join(labels) if labels else "no detections"
            # กำหนด label_flag ตาม detection_result
                            # if detection_result == "sheet":
                            #     label_flag = 0
                            # elif detection_result in ["sheet+gloves", "gloves+sheet", "sheet+gloves", "man+sheet+gloves", "man+gloves+sheet", "sheet+gloves+man", "sheet+man+gloves", "gloves+man+sheet", "gloves+sheet+man"]:
                            #     label_flag = 1
                            # elif detection_result in ["sheet+man", "man+sheet"]:
                            #     label_flag = 2
                            # elif detection_result in ["gloves+man", "man+gloves", "man", "gloves"]:
                            #     label_flag = 0
                            # else:
                            #     label_flag = 0
            label_set = set(labels)

            if {"Working"}.issubset(label_set):
                label_flag = 1
        # elif {"clean", "cart"}.issubset(label_set): #and "gloves" not in label_set:
                #label_flag = 2
            else:
                label_flag = 0

            clean_confs = [conf for cls_id, conf in zip(cls_ids, confs) if cls_id in CONFIG["class_names"]]
            conf_to_save = round(float(clean_confs[0]), 4) if clean_confs else ""

            if CONFIG["save_csv"] and csv_writer:
                csv_writer.writerow([timestamp_now, detection_result, label_flag, conf_to_save, filename])

            results_list.append({
                "predict_time": timestamp_now,
                "detection_result": detection_result,
                "label_flag": label_flag,
                "avg_conf": conf_to_save if conf_to_save != "" else None,
                "file_name": filename
            })

            save_image_flag = False
            if boxes:
                for box, conf, cls in zip(boxes.xyxy.cpu().numpy(), confs, cls_ids):
                    if cls in CONFIG["class_names"] and conf >= CONFIG["confidence_threshold"]:
                        x1, y1, x2, y2 = map(int, box)
                        label = f"{model.names[cls]} ({conf:.2f})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        save_image_flag = True

            if CONFIG["save_image"] and save_image_flag:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                label = "_".join(set(labels))
                cv2.imwrite(os.path.join(save_path_with_boxes, f"{timestamp}_{label}_with_boxes.jpg"), frame)
                cv2.imwrite(os.path.join(save_path_without_boxes, f"{timestamp}_{label}_no_boxes.jpg"), frame)

            if CONFIG["save_video"]:
                if not video_writer:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    output_video_path = os.path.join(output_video_dir, f"output_{datetime.now():%Y%m%d_%H%M%S}.avi")
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, 25, (w, h))
                video_writer.write(frame)

            if CONFIG["show_frame_predict"]:
                cv2.imshow("YOLO Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        if CONFIG["save_video"] and video_writer:
            video_writer.release()
        if CONFIG["save_csv"]:
            csv_file.close()
        if CONFIG["insert_db"] and results_list:
            df = pd.DataFrame(results_list)
            insert_to_postgres(df)
        cv2.destroyAllWindows()
        print("[✔] Detection complete.")

def main():
    set_start_method('spawn')
    start_time = time.time()

    video_input = CONFIG["video_path"]
    backup_dir = CONFIG.get("video_path_backup", "")

    # กรณี RTSP stream
    if video_input.lower().startswith("rtsp://"):
        print(f"[▶] Processing RTSP stream: {video_input}")
        q1 = Queue(maxsize=20)
        p1 = Process(target=video_inspection, args=(video_input, q1))
        p2 = Process(target=model_plot, args=(q1, video_input))
        p1.start()
        p2.start()
        p1.join()
        p2.join()

    # กรณีเป็นโฟลเดอร์ที่เก็บวิดีโอ
    elif os.path.isdir(video_input):
        os.makedirs(backup_dir, exist_ok=True)

        video_files = [
            f for f in os.listdir(video_input)
            if f.lower().endswith(('.mp4', '.avi', '.mov'))
        ]

        for video_file in video_files:
            video_path_full = os.path.join(video_input, video_file)
            print(f"[▶] Processing file: {video_path_full}")
            q1 = Queue(maxsize=20)

            p1 = Process(target=video_inspection, args=(video_path_full, q1))
            p2 = Process(target=model_plot, args=(q1, video_path_full))
            p1.start()
            p2.start()
            p1.join()
            p2.join()

            try:
                backup_path = os.path.join(backup_dir, video_file)
                os.rename(video_path_full, backup_path)
                print(f"[✔] Moved to backup: {backup_path}")
            except Exception as e:
                print(f"[!] Failed to move {video_file} to backup: {e}")

    else:
        print(f"[!] Invalid video_path: {video_input}")

    print(f"\n✅ Total runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
