#!/bin/bash
npm install
npm run compile

cd backend
pip install -r requirements.txt
python -m uvicorn app:app --reload

###
docker pull paddlepaddle/paddle:3.3.0-gpu-cuda12.6-cudnn9.5

# Run the container interactively with GPU support
# Mounts the current directory to /paddle in the container
docker run -it \
    --name paddleocr_gpu \
    --gpus all \
    -v "$PWD":/paddle \
    paddlepaddle/paddle:3.3.0-gpu-cuda12.6-cudnn9.5 \
    //bin/bash