# ใช้ PyTorch image ที่รองรับ CUDA + Python 3.10
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# ตั้ง working directory
WORKDIR /app

# คัดลอก requirements.txt และติดตั้ง Python packages
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --default-timeout=100 --retries=3 --no-cache-dir -r requirements.txt && \
    pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# คัดลอกไฟล์โปรเจกต์ทั้งหมดเข้า container
COPY . .

# รัน main.py เมื่อ container เริ่มต้น
CMD ["python"]
