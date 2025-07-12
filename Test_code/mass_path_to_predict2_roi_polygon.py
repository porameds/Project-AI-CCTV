
import os
import torch
import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import psycopg2
from psycopg2.extras import execute_values
import time
import numpy as np
import onnxruntime as ort


"""
ROI_ALL ={
    "R2-07-11": [(308, 110), (456, 130), (445, 271), (301, 312)],
    "R2-07-12": [(399, 125), (525, 138), (511, 249), (388, 280)]
}
"""

#SELECTED_MACHINE = os.getenv("MACHINE_NAME","R2-07-11")


CONFIG = {
    "model_path": "/home/smart/Project-AI-CCTV/model/oven_machine_b_model3/weights/best.pt",
    "input_dirs": {
        "input1": "/home/smart/Project-AI-CCTV/test_oven_1/output" 
    },
    "predict_dirs": {
        "input1": "/home/smart/Project-AI-CCTV/test_oven_1/predict"
    },
    "no_box_dirs": {
        "input1": "/home/smart/Project-AI-CCTV/test_oven_1/no_box"
    },
    
    # "roi_polygon": [  # พิกัด 4 จุดของกรอบ (ROI)
  
    #"roi_polygons": {
        #"R2-07-11": [(308, 110), (456, 130), (445, 271), (301, 312)],
        #"R2-07-12": [(399, 125), (525, 138), (511, 249), (388, 280)]

        "roi_polygons": {
        "R2-07-11": [(175, 19), (273, 9), (406, 17), (406, 174), (218, 219), (177, 118)],
        # "R2-07-11": [(308, 110), (456, 130), (445, 271), (301, 312)],
        # "R2-07-12": [(399, 125), (525, 138), (511, 249), (388, 280)]


       #ROI_ALL[SELECTED_MACHINE]
    },

    "confidence_threshold": 0.5,
    "save_image": False,
    "save_video": True,
    "save_csv": False,
    "insert_db": True,
    "show_frame_predict": True,
}

def insert_to_postgres(df):
    try:
        values = [
            (
                str(row["predict_time"]),
                str(row["detection_result"]),
                int(row["main_signal"]) if row["main_signal"] is not None else None,
                int(row["sub_signal"]) if row["sub_signal"] is not None else None,
                float(row["avg_conf"]) if row["avg_conf"] is not None else None,
                str(row["file_name"]),
                str(row["machine_name"]) if row["machine_name"] is not None else None
            )
            for _, row in df.iterrows()
        ]

        with psycopg2.connect(
            database="computervision",
            user="myuser",
            password="postgres",
            host="localhost",
            port="5432"
        ) as conn:
            with conn.cursor() as cur:
                insert_query = """
                    INSERT INTO smart_ai.oven_machine_test
                    (predict_time, detection_result, main_signal,sub_signal,avg_conf, file_name, machine_name)
                    VALUES %s
                """
                execute_values(cur, insert_query, values)
                conn.commit()
                print(f"[✔] Success insert to DB")
    except Exception as e:
        print(f"[!] Failed to insert to DB: {e}")


def is_box_inside_polygon(box, polygon):
    x1, y1, x2, y2 = box
    box_points = [
        (x1, y1), (x2, y1), (x2, y2), (x1, y2)
    ]
    polygon_np = np.array(polygon, dtype=np.int32)

    #for point in box_points:
      #  if cv2.pointPolygonTest(polygon_np, point, False) < 0:
           # return False
    #return True
    return all(cv2.pointPolygonTest(polygon_np, pt,False) >= 0 for pt in box_points)
def gen_new_filename(machine_name, predict_time, detection_result, ext =".jpg"):
    safe_machine_name = machine_name
    predict_time_nfn = predict_time
    detection_result_nfn = detection_result
    return f"{safe_machine_name}_{predict_time_nfn}_{detection_result_nfn}{ext}"

def predict_images(input_dir, output_dir, model, name_tag, device="cpu"):
    os.makedirs(output_dir, exist_ok=True)
    no_box_dir = CONFIG["no_box_dirs"][name_tag]
    os.makedirs(no_box_dir, exist_ok=True)

    roi_polygons = CONFIG["roi_polygons"]

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    results_list = []

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        original_img = img.copy()
        result = model.predict(img, device=device)[0]

        boxes = result.boxes
        machine_data = {mn: {"labels": set(), "confidences": [], "boxes": []} for mn in roi_polygons.keys()}

        if boxes and len(boxes.conf) > 0:
            for box, conf, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)):
                if conf >= CONFIG["confidence_threshold"]:
                    box_coords = tuple(map(int, box))
                    label_name = model.names[cls]
                    machine_name = get_box_machine_name(box_coords, roi_polygons)
                    if not machine_name:
                        continue
                    
                    machine_data[machine_name]["labels"].add(label_name)
                    machine_data[machine_name]["confidences"].append(conf)
                    machine_data[machine_name]["boxes"].append(box_coords)

                    # วาดกล่องบนภาพ
                    label = f"{label_name} ({conf:.2f})"
                    cv2.rectangle(img, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (0, 255, 0), 2)
                    cv2.putText(img, label, (box_coords[0], box_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # วาด ROI Polygon (เส้นขอบโซนตรวจจับแต่ละเครื่อง)
        for machine_name, polygon in roi_polygons.items():
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

            # แสดงชื่อเครื่องไว้ที่กลาง polygon
            cx = int(np.mean([p[0] for p in polygon]))
            cy = int(np.mean([p[1] for p in polygon]))
            cv2.putText(img, machine_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


        # วนลูปแยกเครื่อง
        for machine_name, data in machine_data.items():
            label_set = data["labels"]
            confidences = data["confidences"]
            boxes = data["boxes"]

            if not label_set:
                continue  # ข้ามถ้าไม่มี label ในเครื่องนี้

            # คำนวณ signal
            main_signal = 0
            sub_signal = 0
            if "machine open" in label_set:
                main_signal = 1
                if {"mask", "glove"}.issubset(label_set) or {"mask", "black glove"}.issubset(label_set):
                    sub_signal = 2
                elif "mask" in label_set:
                    sub_signal = 3
                elif "glove" in label_set or "black glove" in label_set:
                    sub_signal = 4

            detection_result = ', '.join(sorted(label_set))
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            predict_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_filename = gen_new_filename(machine_name, predict_time, detection_result)

            result_dict = {
                "predict_time": predict_time,
                "detection_result": detection_result,
                "main_signal": main_signal,
                "sub_signal": sub_signal,
                "avg_conf": round(avg_conf, 2),
                "file_name": new_filename,
                "machine_name": machine_name
            }
            results_list.append(result_dict)

            # บันทึกภาพ no_box เฉพาะภาพที่เข้าเงื่อนไข
            if sub_signal not in [0] and main_signal != 0:
                try:
                    cv2.imwrite(os.path.join(no_box_dir, new_filename), original_img)
                    print(f" Saved (no box): {new_filename}")
                except Exception as e:
                    print(f"[!] Failed to save no-box image {new_filename}: {e}")

            if CONFIG["save_image"]:
                try:
                    cv2.imwrite(os.path.join(output_dir, new_filename), img)
                except Exception as e:
                    print(f"[!] Failed to save image {new_filename}: {e}")

        if CONFIG["show_frame_predict"]:
            cv2.imshow(f"{name_tag} Predict", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Save CSV
    if CONFIG["save_csv"] and results_list:
        df = pd.DataFrame(results_list)
        csv_path = os.path.join(output_dir, f"{name_tag}_results.csv")
        df.to_csv(csv_path, index=False)

    # Insert DB
    if CONFIG["insert_db"] and results_list:
        insert_to_postgres(pd.DataFrame(results_list))

    # Remove processed images
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        try:
            os.remove(img_path)
        except Exception as e:
            print(f"[!] Failed to remove {img_path}: {e}")

# def is_box_inside_polygon(box, polygon):
#     x1, y1, x2, y2 = box
#     box_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
#     polygon_np = np.array(polygon, dtype=np.int32)
#     return all(cv2.pointPolygonTest(polygon_np, pt, False) >= 0 for pt in box_points)
def get_box_machine_name(box, roi_polygons):
    for machine_name, polygon in roi_polygons.items():
        if is_box_inside_polygon(box, polygon):
            return machine_name
    return None

def run_path_to_predict_roi_polygon_oven():
    # model = YOLO(CONFIG["model_path"])
    # model.to('cuda')
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

    model = YOLO(CONFIG["model_path"])

    try:
        while True:
            for name_tag, input_dir in CONFIG["input_dirs"].items():
                output_dir = CONFIG["predict_dirs"][name_tag]
                predict_images(input_dir, output_dir, model, name_tag)

            if CONFIG["show_frame_predict"]:
                cv2.destroyAllWindows()

            time.sleep(5)
    except KeyboardInterrupt:
        print(" Stopped by user")

if __name__ == "__main__":
    run_path_to_predict_roi_polygon_oven()