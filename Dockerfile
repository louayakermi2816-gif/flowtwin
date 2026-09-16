FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY flowtwin ./flowtwin
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir .
EXPOSE 8000
CMD ["uvicorn", "flowtwin.api:app", "--host", "0.0.0.0", "--port", "8000"]
