FROM python:3.11-slim
WORKDIR /app
COPY constraints.txt ./
# PyTorch first, in its own layer (CPU wheel, version from constraints.txt): code changes then
# rebuild without downloading it again
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "$(grep '^torch==' constraints.txt)"
COPY pyproject.toml README.md ./
COPY flowtwin ./flowtwin
RUN pip install --no-cache-dir -c constraints.txt .
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "flowtwin.api:app", "--host", "0.0.0.0", "--port", "8000"]
