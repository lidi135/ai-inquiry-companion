# AI 问诊陪练助手 —— 生产运行镜像
# 构建：docker build -t ai-inquiry-companion .
# 运行：docker run --rm -p 8501:8501 -e DEEPSEEK_API_KEY=sk-xxx ai-inquiry-companion
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 先拷贝依赖清单并安装，充分利用镜像层缓存
COPY src/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 拷贝应用源码（含 app.py / core / eval / cases / .streamlit 主题）
COPY src/ ./

# 多模型密钥通过环境变量注入（DEEPSEEK_API_KEY 等），不在镜像内固化 secrets.toml
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

# 健康检查：Streamlit 内置 _stcore/health 端点
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5).status==200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.headless", "true"]