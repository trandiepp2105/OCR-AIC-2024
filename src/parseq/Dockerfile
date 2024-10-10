# FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Cài đặt các gói phụ thuộc khác của bạn
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libxrender1 \
    libfontconfig1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /parseq

# Sao chép requirements.txt và cài đặt các gói Python khác nếu cần
COPY requirements.txt ./ 


# Xóa thủ công PyYAML cũ
# RUN rm -rf /usr/local/lib/python3.*/dist-packages/PyYAML* \
#     && rm -rf /usr/lib/python3.*/dist-packages/PyYAML* 
# Cài đặt pip, PyYAML phiên bản 6.0.0 và các gói yêu cầu khác



RUN pip install --upgrade pip \
    # && pip install --no-cache-dir --ignore-installed PyYAML==6.0.0 \
    && pip install --no-cache-dir -r requirements.txt

RUN pip install pip==23.2.1
RUN pip install pytorch-lightning==1.6.5

ENV PYTHONPATH=/parseq
ENV PYTHONIOENCODING=utf-8


