#!/bin/bash
cd frontend
npm install
npm run build
if [ $? -eq 0 ]; then
    echo -e "\033[1;32m\nfrontend build success \033[0m"
else
    echo -e "\033[1;31m\nfrontend build failed\n\033[0m" >&2  exit 1
fi
cd ../
CUDA_VISIBLE_DEVICES=0 python main.py --port 7860 --host 0.0.0.0 --num_gpus 1 --step 1 --model_type T2V-1.3B --enable-metrics