FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
# Install dependencies, handling bitsandbytes gracefully if it fails
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir torch transformers peft accelerate numpy scipy fastapi "uvicorn[standard]" pydantic

COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV CHECKPOINT_PATH=/app/Model/Final_Model

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

