# Sử dụng Python 3.9 làm image base
FROM python:3.9

# Thiết lập thư mục làm việc
WORKDIR /scenetext

# Sao chép tệp requirements.txt vào container
COPY requirements.txt ./

# Cài đặt các gói yêu cầu từ một gương và tăng thời gian timeout
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=100 -i https://pypi.python.org/simple -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY ./ ./

# Thiết lập biến môi trường PYTHONPATH để đảm bảo import modules từ /scenetext
ENV PYTHONPATH=/scenetext
