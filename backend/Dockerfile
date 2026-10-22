FROM python:3.13-slim
WORKDIR /workspace
COPY backend/requirements.txt /workspace/backend/requirements.txt
COPY backend/requirements.lock.txt /workspace/backend/requirements.lock.txt
RUN pip install --no-cache-dir -r backend/requirements.lock.txt
COPY backend/requirements-reranker.txt /workspace/backend/requirements-reranker.txt
RUN pip install --no-cache-dir 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r backend/requirements-reranker.txt
COPY backend /workspace/backend
COPY evaluation /workspace/evaluation
RUN mkdir -p /workspace/data
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
