from multiprocessing import Process
from mass_cam_to_path import run_cam_to_path_oven
from mass_path_to_predict2_roi_polygon import run_path_to_predict_roi_polygon_oven #,mass_path_to_predict2_roi_polygon
#from python_analyze import run_job
import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    # สร้าง process
    p1 = Process(target=run_cam_to_path_oven)
    p2 = Process(target=run_path_to_predict_roi_polygon_oven )
    #p3 = Process(target= run_job)
    # เริ่มทำงาน
    p1.start()
    p2.start()
    #p3.start()

    # รอให้ process ทำงานจนจบ (ถ้ามี loop จะรันไม่จบจนปิดเอง)
    p1.join()
    p2.join()
    #p3.join()