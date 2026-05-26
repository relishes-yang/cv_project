# ===================== YOLO11n 专属训练启动脚本 =====================
# 作用：一键启动训练，无需手动敲命令，参数已适配你的云主机+数据集
# 用法：直接运行 python train.py 即可
# ==================================================================

from ultralytics import YOLO
import torch
# 1. 基础配置（固定你的路径，无需修改）
DATA_CONFIG = "/data/coding/cv_project/coco.yaml"  # 你的数据集配置文件
SAVE_PROJECT = "/data/coding/cv_project/runs"     # 训练结果保存目录
DEVICE = 0                                        # 使用你的RTX5060Ti GPU

# 2. 加载YOLO11n预训练模型（轻量版，适配16G显存）
model = YOLO("yolo11s.pt")

# ===================== 二选一：测试训练 / 正式训练 =====================
# 🔴 选项1：快速测试训练（10轮，验证流程是否正常，推荐先跑这个）
# model.train(
#     data=DATA_CONFIG,
#     epochs=10,        # 训练轮数
#     imgsz=640,        # 图片分辨率
#     batch=16,         # 批次大小（16G显存最优）
#     device=0,
#     project=SAVE_PROJECT,
#     name="test_train",# 训练结果文件夹名称
#     exist_ok=True,    # 覆盖旧文件
#     workers=4         # 数据加载线程
# )

# 🟢 选项2：正式训练（100轮，训练完直接用！测试没问题后启用）
# 启用方法：把上面 选项1 的代码注释掉，取消下面代码的注释
model.train(
    data=DATA_CONFIG,
    epochs=30,       # 正式训练轮数
    imgsz=1280,      # 输入图像尺寸：统一将图片缩放到640×640，适配电力小目标检测
    batch=16,       # 批次大小：每轮训练一次性加载16张图片，平衡训练速度与显存占用
    device=DEVICE,  # 运行设备：指定使用GPU训练（DEVICE=0代表第一块显卡，加速训练）
    project=SAVE_PROJECT,       # 训练结果保存根目录：所有训练日志、权重文件统一存放
    name="5.25add_data30",        # 本次训练任务名称：按日期+数据增强命名，方便区分版本
    exist_ok=True,          # 覆盖同名文件夹：重复训练时直接覆盖旧文件，避免生成冗余文件夹
    workers=4,          # 数据加载线程数：4线程并行加载图片，提升训练速度，不卡顿
    patience=30,       # 早停：50轮不提升自动停止，防止过拟合

      # --- 学习率（适配短轮训练，加快收敛） ---
    lr0=0.0008,        # 略微提高初始学习率，30轮学得更快
    lrf=0.01,          # 末尾学习率比例，适配短轮衰减
    optimizer="AdamW",
    weight_decay=0.001,

    # --- 损失权重（保留！专门针对小目标+少数类提分） ---
    box=10.0,
    cls=3.0,
    dfl=2.0,
    warmup_epochs=2,  # 预热缩短为2轮（总共只30轮，不用长时间预热）

    # --- 数据增强（适配30轮，后半段关闭Mosaic） ---
    mosaic=0.2,
    close_mosaic=10,   # 最后10轮关闭Mosaic（30-10=前20轮用增强，合理）
    degrees=15.0,
    flipud=0.5,
    fliplr=0.5,
    scale=0.5,
    hsv_h=0.01,
    hsv_s=0.5,
    hsv_v=0.3,
    copy_paste=0.3     # 保留Copy-Paste，缓解样本不均衡，提分关键
)
# ==================================================================