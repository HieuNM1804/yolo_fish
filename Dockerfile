# Sử dụng image Python phiên bản nhẹ (slim)
FROM python:3.9-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cài đặt các thư viện hệ thống cơ bản cần thiết (nếu có)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements.txt vào trước để tận dụng cache layer của Docker
COPY requirements.txt .

# Cài đặt các gói thư viện Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ project (đã loại bỏ các file thừa nhờ .dockerignore) vào container
COPY . .

# Lệnh mặc định chạy khi container khởi động
CMD ["python", "tracking.py"]
