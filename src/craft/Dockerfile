FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
# FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# Cài đặt các gói phụ thuộc khác của bạn
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libxrender1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /craft

# Sao chép requirements.txt và cài đặt các gói Python khác nếu cần
COPY requirements.txt ./ 
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/craft
ENV PYTHONIOENCODING=utf-8

# Khởi động ứng dụng
# CMD ["python", "app.py"]
