FROM python:3.9-alpine

# 安装系统依赖
RUN apk add --no-cache build-base

# 设置工作目录
WORKDIR /app

# 复制requirements.txt到工作目录
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码到工作目录
COPY . .

# 暴露端口
EXPOSE 5000

# 定义启动命令
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
