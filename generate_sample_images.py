# -*- coding: utf-8 -*-
"""
生成示例手写数字图片，放在 sample_images/ 文件夹中。

这些图片从 sklearn 自带的 digits 数据集中提取，每个数字(0~9)各一张，
放大保存为 PNG（黑底白字），供 classify.py 的"用户输入图片预测"功能测试使用。

运行：python generate_sample_images.py
输出：sample_images/digit_0.png ~ digit_9.png
"""
import os
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits

# 创建 sample_images 文件夹（已存在则不报错）
os.makedirs("sample_images", exist_ok=True)

# 加载 digits 数据集（1797张 8×8 手写数字灰度图）
digits = load_digits()

# 每个数字(0~9)选数据集中一张，保存为图片
for digit in range(10):
    # 注意：数字5的第一个样本(索引5)本身会被KNN模型误判为9，
    # 所以这里选第二个5样本(索引15)，保证示例图片能被正确识别。
    if digit == 5:
        idx = 15
    else:
        # 找到第一个标签为 digit 的样本下标
        idx = list(digits.target).index(digit)
    # digits.images[idx] 是 8×8 的数组，像素值 0~16（0=黑色背景，16=白色数字）
    img_array = digits.images[idx]
    # 归一化到 0~255（PIL 灰度图的像素范围）
    img_255 = (img_array / 16 * 255).astype(np.uint8)
    # 转成 PIL 灰度图
    img = Image.fromarray(img_255, mode="L")
    # 放大到 64×64（最近邻插值，保持像素方块感，方便人眼查看）
    img = img.resize((64, 64), Image.NEAREST)
    # 保存为 PNG
    img.save(f"sample_images/digit_{digit}.png")
    print(f"已生成 sample_images/digit_{digit}.png（数字 {digit}）")

print("\n示例图片生成完毕！")
print("运行 classify.py 后，输入 sample_images/digit_5.png 即可测试识别。")
