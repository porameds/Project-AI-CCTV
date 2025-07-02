import os
import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import psycopg2
from psycopg2.extras import execute_values
import time
import numpy as np

CONFIG = {
    "model_path": r"/home/user/Project/oven_machine_model/model/oven_machine_model/weights/best.onnx",
    "input_dirs": {
        "input1": r"/home/user/Project/test_oven_3/output"
    },
    "predict_dirs": {
        "input1": r"/home/user/Project/test_oven_3/predict"
    },
    "no_box_dirs": {
        "input1": r"/home/user/Project/test_oven_3/no_box"
    },
    # "roi_polygon": [  # พิกัด 4 จุดของกรอบ (ROI)
    #     (215, 159),  # จุด 1
    #     (290, 164),  # จุด 2
    #     (290, 300),  # จุด 3
    #     (98, 300)   # จุด 4
    "roi_polygons": {
        "R2-07-11": [(308, 110), (512, 137), (487, 259), (301, 312)],
        #"R2-07-12": [(290, 164), (373, 154), (456, 299), (290, 300)]
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
                str(row["file_name"]),
                str(row["machine_name"]) if row["machine_name"] is not None else None
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
                    INSERT INTO smart_ai.a31_dpp_clean_by_vacuum
                    (predict_time, detection_result, label_flag, avg_conf, file_name, machine_name)
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

def predict_images(input_dir, output_dir, model, name_tag):
    os.makedirs(output_dir, exist_ok=True)
    no_box_dir = CONFIG["no_box_dirs"][name_tag]
    os.makedirs(no_box_dir, exist_ok=True)

    roi_polygons = CONFIG["roi_polygons"]

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    results_list = []
    video_writer = None

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        original_img = img.copy()
        result = model.predict(img, device="cpu")[0]

        boxes = result.boxes
        label_set = set()
        confidences = []
        has_box = False
        valid_for_saving = False
        #machine_names = set()

        # วาด ROI ทุกวง
        for polygon in roi_polygons.values():
            roi_np = np.array([polygon], dtype=np.int32)
            cv2.polylines(img, roi_np, isClosed=True, color=(255, 0, 0), thickness=2)

        if boxes and len(boxes.conf) > 0:
            for box, conf, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)):
                if conf >= CONFIG["confidence_threshold"]:
                    box_coords = tuple(map(int, box))
                    label_name = model.names[cls]
                   # machine_name = get_box_machine_name(box_coords, roi_polygons)

                   #check box in ROI
                    in_any_polygon = any(is_box_inside_polygon(box_coords, polygon)for polygon in roi_polygons.values())
                    if not in_any_polygon:
                        continue

                    label = f"{label_name} ({conf:.2f})"
                    cv2.rectangle(img, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (0, 255, 0), 2)
                    cv2.putText(img, label, (box_coords[0], box_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                   # if machine_name:
                      #  label = f"{model.names[cls]} ({conf:.2f}) [{machine_name}]"
                      #  cv2.rectangle(img, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (0, 255, 0), 2)
                       # cv2.putText(img, label, (box_coords[0], box_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    label_set.add(label_name)
                    confidences.append(conf)
                    has_box = True
                    valid_for_saving = True
                      #  label_set.add(model.names[cls])
                      #  confidences.append(conf)
                     #   has_box = True
                      #  valid_for_saving = True
                      #  machine_names.add(machine_name)
        arr_signal = ["machine open","mask","glove","black glove"]
        label_flag = []
        for item in arr_signal:
            if item in label_set:
                if item == "machine open":
                    label_flag.append(1)
                elif item == "mask":
                    label_flag.append(2)
                elif item == "glove":
                    label_flag.append(3)
                elif item == "black glove":
                    label_flag.append(4)
                else:
                    label_flag.append(0)
            else:
                label_flag.append(0)

        detection_result = ', '.join(sorted(label_set)) if label_set else "no detections"
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        #machine_name = ', '.join(sorted(machine_names)) if machine_names else None

        result_dict = {
            "predict_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detection_result": detection_result,
            "label_flag": label_flag,
            "avg_conf": round(avg_conf, 2),
            "file_name": img_file,
            #"machine_name": machine_name
        }

        #if valid_for_saving:
        results_list.append(result_dict)

        if CONFIG["save_image"]:
                save_path = os.path.join(output_dir, img_file)
                try:
                    cv2.imwrite(save_path, img)
                    print(f"[💾] Saved: {save_path}")
                except Exception as e:
                    print(f"[!] Failed to save image {save_path}: {e}")
        elif not has_box and label_set:
            save_path = os.path.join(no_box_dir, img_file)
            try:
                cv2.imwrite(save_path, original_img)
                print(f"[💾] Saved (no box): {save_path}")
            except Exception as e:
                print(f"[!] Failed to save no-box image {save_path}: {e}")

        #try:
           # os.remove(img_path)
           # print(f"[🗑] Removed processed image: {img_path}")
        #except Exception as e:
            #print(f"[!] Failed to remove {img_path}: {e}")

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

    if CONFIG["insert_db"] and results_list:
        df = pd.DataFrame(results_list)
        insert_to_postgres(df)

        print(f"[✔] Done predicting for {name_tag}")
"""
def is_box_inside_polygon(box, polygon):
    x1, y1, x2, y2 = box
    box_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    polygon_np = np.array(polygon, dtype=np.int32)
    return all(cv2.pointPolygonTest(polygon_np, pt, False) >= 0 for pt in box_points)

def get_box_machine_name(box, roi_polygons):
    for machine_name, polygon in roi_polygons.items():
        if is_box_inside_polygon(box, polygon):
            return machine_name
    return None
"""
def main():
    model = YOLO(CONFIG["model_path"])
    try:
        #while True:
            for name_tag, input_dir in CONFIG["input_dirs"].items():
                output_dir = CONFIG["predict_dirs"][name_tag]
                predict_images(input_dir, output_dir, model, name_tag)

            if CONFIG["show_frame_predict"]:
                cv2.destroyAllWindows()

            time.sleep(2)
    except KeyboardInterrupt:
        print("[⛔] Stopped by user")

if __name__ == "__main__":
    main()


