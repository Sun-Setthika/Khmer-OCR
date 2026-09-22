FROM python:3.11-slim
#docker also needs dir to store the dependecies, and COPY means transfer file
WORKDIR /app 
RUN python -m pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY inference.py vocab.json checkpoint.pth ./

EXPOSE 7861
CMD ["python", "inference.py"]