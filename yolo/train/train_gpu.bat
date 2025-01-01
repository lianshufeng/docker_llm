docker run --gpus all --shm-size 8G --rm -it -v /var/run/docker.sock:/var/run/docker.sock -v %CD%:/work  -w /work lianshufeng/llm:yolo bash

:: 设置工作路径
:: yolo settings datasets_dir=/work


:: 开始训练 device=cpu
:: yolo detect train data=datasets/yolo/dataset.yaml model=yolo11n.pt epochs=100 imgsz=640 

:: 恢复训练
:: yolo train resume model=runs/detect/train/weights/last.pt