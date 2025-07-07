FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# ติดตั้ง libgl1 และ libglib2.0-0 ที่จำเป็นสำหรับ OpenCV
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# คัดลอก requirements.txt และติดตั้ง dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --default-timeout=100 --retries=3 --no-cache-dir -r requirements.txt

# คัดลอกไฟล์โปรเจกต์
COPY . .

CMD ["python", "main_run_oven_machine.py"]
