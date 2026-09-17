# -*- coding: utf-8 -*-
"""
方案A：图片分类——手写数字识别（监督学习，sklearn）

【对应高考考点】
  · 监督学习：每张图片都带标签（标注了是哪个数字），模型从带标签数据中学习规律
  · 数据集划分：训练集（训练模型）+ 测试集（评估模型效果），两者不能混用
  · 计算机视觉：将手写数字图片的像素值作为特征输入模型
  · 模型评估：用准确率和混淆矩阵衡量模型好坏

【本次新增功能】
  · 程序先用 digits 数据集训练 KNN 模型
  · 然后用户通过输入图片文件路径，程序读取图片、预处理、识别、输出结果
  · 支持黑底白字和白底黑字两种图片（自动判断反色）

使用 sklearn 自带的 digits 数据集（1797张 8×8 手写数字灰度图片），
用 K近邻(KNN) 分类器训练模型，识别手写数字 0~9。
不使用神经网络、不做深度学习，完全高考生水平。

运行：python classify.py
"""

# ===== 导入需要的库 =====
import numpy as np                       # numpy：数值计算，图片像素数组处理
from PIL import Image                    # PIL：读取用户图片、图片缩放预处理
import matplotlib.pyplot as plt          # 绘图库，用来画混淆矩阵
from sklearn.datasets import load_digits          # 加载sklearn自带的手写数字数据集
from sklearn.model_selection import train_test_split   # 划分训练集和测试集的工具
from sklearn.neighbors import KNeighborsClassifier     # K近邻分类器
from sklearn.metrics import accuracy_score, confusion_matrix  # 准确率、混淆矩阵评估工具

# ===== 设置matplotlib中文字体（Windows用SimHei），防止图中中文乱码 =====
plt.rcParams["font.sans-serif"] = ["SimHei"]    # 设置中文字体为黑体
plt.rcParams["axes.unicode_minus"] = False       # 解决负号显示为方块的问题


# ============================================================
# 图片预处理函数：把用户的任意图片转成模型能识别的格式
# ============================================================
# 模型训练时用的是 8×8 像素、像素值 0~16、黑底白字 的图片。
# 用户上传的图片可能是任意大小、任意底色，所以必须预处理成一致的格式。
def preprocess_image(image_path):
    """
    读取用户图片，预处理成与 digits 数据集一致的 1×64 特征向量。
    参数：image_path —— 图片文件路径
    返回：1×64 的 numpy 数组（64个像素值，0~16）
    """
    # 第1步：打开图片并转灰度图（"L"模式 = 8位灰度，像素值0~255）
    img = Image.open(image_path).convert("L")

    # 第2步：缩放到 8×8（和 digits 数据集尺寸完全一致）
    # NEAREST 最近邻插值：和生成示例图片时的放大方式对称，
    # 不会引入双线性插值的灰度渐变，最大程度保留原始像素值
    img = img.resize((8, 8), Image.NEAREST)

    # 第3步：转成 numpy 数组，像素值范围 0~255
    arr = np.array(img, dtype=float)

    # 第4步：自动判断是白底黑字还是黑底白字
    # 判断方法：看图片四个角的平均亮度，如果 >127 说明角是白色（白底），
    # digits 数据集是黑底白字，所以白底黑字需要反色。
    corners = [arr[0, 0], arr[0, 7], arr[7, 0], arr[7, 7]]
    if np.mean(corners) > 127:
        arr = 255 - arr    # 反色：白底黑字 → 黑底白字

    # 第5步：归一化到 0~16（digits 数据集的像素范围，不是 0~255）
    arr = arr / 255 * 16

    # 第6步：展平成 1×64 的二维数组（sklearn 的 predict 要求输入是二维）
    # reshape(1, -1)：1行，列数自动计算（即64）
    return arr.reshape(1, -1)


# ============================================================
# 第1步：加载数据集
# ============================================================
# digits 包含：
#   data   : 1797张图片的像素数据，形状 (1797, 64)，每行64个像素值(0~16)
#   target : 1797个标签，形状 (1797,)，每个值是 0~9 的数字
#   images : 1797张图片的二维形式，形状 (1797, 8, 8)，用于可视化
digits = load_digits()

X = digits.data      # X：特征矩阵，每行是一张图片的64个像素值（模型的"输入"）
y = digits.target    # y：标签向量，每个元素是图片对应的真实数字（模型要学习的"答案"）

print("数据集总样本数：", len(X))
print("每张图片特征数（像素数）：", X.shape[1])

# ============================================================
# 第2步：划分训练集和测试集【高考核心考点】
# ============================================================
# train_test_split 参数：
#   X, y          : 要划分的特征和标签
#   test_size=0.3 : 测试集占30%，训练集占70%
#   random_state=42: 随机种子，固定后每次划分结果相同（方便复现）
# 【注意】训练集和测试集必须严格分开，不能用测试集训练，否则评估结果不准（相当于作弊）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

print("训练集样本数：", len(X_train))
print("测试集样本数：", len(X_test))

# ============================================================
# 第3步：训练KNN分类器（监督学习）
# ============================================================
# KNeighborsClassifier(n_neighbors=3)：K=3，找距离最近的3个邻居投票决定分类
# KNN原理：对于一张新图片，找训练集中和它最像的3张图片，
#          这3张里出现次数最多的数字，就是预测结果。
model = KNeighborsClassifier(n_neighbors=3)

# fit()：用训练集训练模型（模型从 X_train 学习像素规律，对照 y_train 的答案）
model.fit(X_train, y_train)
print("模型训练完成")

# ============================================================
# 第4步：用测试集预测，评估准确率
# ============================================================
# predict()：让模型对测试集图片进行预测，返回预测结果数组
y_pred = model.predict(X_test)

# accuracy_score(真实标签, 预测标签)：计算预测正确的比例（准确率）
acc = accuracy_score(y_test, y_pred)
print("测试集准确率：", round(acc * 100, 2), "%")

# ============================================================
# 第5步：混淆矩阵——看哪些数字容易被认错
# ============================================================
# confusion_matrix 返回 10×10 矩阵：
#   行 i 表示真实数字是 i 的图片，列 j 表示模型预测为 j 的图片数量
#   对角线上的数字越大说明识别越准确，非对角线表示认错的数量
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵（行=真实数字，列=预测数字）：")
print(cm)

# ============================================================
# 第6步：保存混淆矩阵图（不弹窗阻塞，直接保存文件）
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_title("混淆矩阵（准确率 %.2f%%）" % (acc * 100))
ax.set_xlabel("预测数字")
ax.set_ylabel("真实数字")
ax.set_xticks(range(10))
ax.set_yticks(range(10))
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("output/digits_result.png", dpi=150)
print("混淆矩阵图已保存 → output/digits_result.png")

# ============================================================
# 第7步：【新增】用户输入图片路径，识别并输出结果
# ============================================================
print("\n" + "=" * 50)
print("手写数字识别系统")
print("=" * 50)
print("示例图片在 sample_images/ 文件夹中，例如：sample_images/digit_5.png")
print("请输入图片文件路径进行识别，输入 quit 退出程序。")

while True:
    # input()：从键盘读取用户输入的一行文字，strip()去掉首尾空格
    path = input("\n请输入图片路径：").strip()

    # 用户输入 quit 则退出循环
    if path.lower() == "quit":
        print("程序结束，再见！")
        break

    # 空输入则跳过，重新提示
    if not path:
        continue

    try:
        # 调用预处理函数，把用户图片转成模型能识别的特征向量
        feature = preprocess_image(path)

        # 用训练好的模型预测，[0] 取第一个（也是唯一一个）预测结果
        pred = model.predict(feature)[0]

        # 输出识别结果
        print(f"✅ 识别结果：数字 {pred}")

    except FileNotFoundError:
        # 文件不存在时的友好提示
        print(f"❌ 错误：找不到文件 '{path}'，请检查路径是否正确。")
    except Exception as e:
        # 其他异常（图片损坏、格式不支持等）的提示
        print(f"❌ 处理图片时出错：{e}")
