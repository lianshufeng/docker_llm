from ultralytics import YOLO

# 加载模型
model = YOLO("./yolo11n.pt")
# model.to('cpu')
# model.to('cuda')




# 训练模型
# train_results = model.train(
#     data="coco8.yaml",  # 数据集 YAML 路径
#     epochs=100,  # 训练轮次
#     imgsz=640,  # 训练图像尺寸
#     device="cpu",  # 运行设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
# )

# 评估模型在验证集上的性能
# metrics = model.val()
source = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# 在图像上执行对象检测
results = model(source=source)
results[0].show()

# 将模型导出为 ONNX 格式
# path = model.export(format="onnx")  # 返回导出模型的路径
