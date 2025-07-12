import cv2
import os
import shutil
from datetime import datetime
from multiprocessing import Process

CONFIG = {
    "sources": {
        #"input1": "rtsp://admin:Plant_1340@192.168.75.45:554/Streaming/Channels/801"
        "input1": "/home/smart/Project-AI-CCTV/vdo_test/oven_b"
    },
    "output_dirs": {
        "input1": "/home/smart/Project-AI-CCTV/test_oven_1/output"
        # "input2": "D:/Smart/AI/CCTV/PPE RDEV/dataset1/test2/output"
    },
    "backup_video_dir": "/home/smart/Project-AI-CCTV/test_oven_1/output/input_backup",
    "save_interval": 25,
    "show_frame": True,
    "resize": (640, 384)  # ขนาด resize
}

def is_video_file(path):
    return any(path.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"])

def capture_and_save_frames_from_file(source_name, video_path, output_dir):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[✘] Cannot open video/stream: {video_path}")
            return

        os.makedirs(output_dir, exist_ok=True)
        frame_count = 0
        saved_count = 0

        # ใช้ชื่อไฟล์หากเป็นวิดีโอไฟล์ / หรือชื่อ "live" สำหรับ RTSP
        base_name = os.path.splitext(os.path.basename(video_path))[0] if os.path.isfile(video_path) else "live"

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            resized_frame = cv2.resize(frame, CONFIG["resize"])
            frame_count += 1

            if frame_count % CONFIG["save_interval"] == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{source_name}_{base_name}_{timestamp}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), resized_frame)
                saved_count += 1

            if CONFIG["show_frame"]:
                pass
                cv2.imshow(f"{source_name}", resized_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        print(f"[✔] Saved {saved_count} frames from {video_path}")
        
        # สำหรับวิดีโอไฟล์ → ย้ายไป backup
        if os.path.isfile(video_path):
            os.makedirs(CONFIG["backup_video_dir"], exist_ok=True)
            shutil.move(video_path, os.path.join(CONFIG["backup_video_dir"], os.path.basename(video_path)))
            print(f"[→] Moved {video_path} to backup folder.")
    finally:
        cap.release()
        if CONFIG["show_frame"]:
            cv2.destroyWindow(source_name)

def process_videos_in_folder(source_name, source_folder, output_folder):
    video_files = [f for f in os.listdir(source_folder) if is_video_file(f)]
    if not video_files:
        print(f"[!] No video files in {source_folder}")
        return

    for video_file in video_files:
        video_path = os.path.join(source_folder, video_file)
        capture_and_save_frames_from_file(source_name, video_path, output_folder)
        

def start_process(source_name, source):
    output_folder = CONFIG["output_dirs"][source_name]

    if source.lower().startswith("rtsp://"):
        # ✅ RTSP stream
        capture_and_save_frames_from_file(source_name, source, output_folder)
    elif os.path.isdir(source):
        # ✅ Folder with video files
        process_videos_in_folder(source_name, source, output_folder)
    else:
        print(f"[✘] Invalid source path: {source}")

def run_cam_to_path_oven():
    processes = []

    for source_name, source in CONFIG["sources"].items():
        p = Process(target=start_process, args=(source_name, source))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if CONFIG["show_frame"]:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_cam_to_path_oven()