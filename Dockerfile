FROM saharifrah1/sahar:v4

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    "monai[all]>=1.3.0" \
    "numpy>=1.24.0" \
    "tqdm>=4.66.0"
