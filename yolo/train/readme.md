
## 数据集
- voc转换为labelme
```python
python voc2labelme.py datasets\fire_smoke
```

- labelme 转换为yolo格式
```python
python labelme2yolo.py datasets/fire_smoke/images datasets/yolo
```

## 训练
- 进入docker容器
```shell
docker run --gpus all --shm-size 8G --rm -it -v /var/run/docker.sock:/var/run/docker.sock -v %CD%:/work  -w /work lianshufeng/llm:yolo bash
```

- 设置数据集默认路径(结合挂载目录)
```shell
yolo settings datasets_dir=/work
```

- 开始训练
```shell
yolo detect train data=datasets/yolo/dataset.yaml model=yolo11n.pt epochs=100 imgsz=640 
```