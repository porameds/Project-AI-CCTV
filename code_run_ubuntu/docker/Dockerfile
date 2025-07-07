FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV TZ=Asia/Bangkok
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# ติดตั้ง lib ที่จำเป็นสำหรับ OpenCV และ psql client
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --default-timeout=100 --retries=3 --no-cache-dir -r requirements.txt && \
    pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu122

COPY . .

CMD ["python", "main_run_oven_machine.py"]
