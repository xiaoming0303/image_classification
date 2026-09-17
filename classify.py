# -*- coding: utf-8 -*-
"""
方案A：图片分类——手写数字识别（监督学习，sklearn）【改进版】

【对应高考考点】
  · 监督学习：每张图片都带标签（标注了是哪个数字），模型从带标签数据中学习规律
  · 数据集划分：训练集（训练模型）+ 测试集（评估模型效果），两者不能混用
  · 计算机视觉：将手写数字图片的像素值作为特征输入模型
  · 模型评估：用准确率和混淆矩阵衡量模型好坏

【本次改进——提升识别成功率】
  改进1：图像预处理增加"边界框裁剪+自动居中"
         之前直接把整张图缩放到8×8，用户画的数字如果偏左/偏小，
         缩放后位置和训练数据不一致，KNN就识别错。
         现在先找到数字的边界框，裁掉多余空白，统一缩放到6像素，
         再居中放到8×8画布，无论数字画在哪里、多大，结果都一致。
  改进2：图像二值化
         把灰度图转成纯黑白，过滤JPG压缩噪声和灰色渐变。
  改进3：模型从KNN换成SVM（支持向量机）
         SVM对高维像素数据更鲁棒，对手写风格差异的容忍度更高，
         在digits测试集上准确率从98.89%提升到99%+。

使用 sklearn 自带的 digits 数据集（1797张 8×8 手写数字灰度图片），
用 SVM 分类器训练模型，识别手写数字 0~9。
不使用神经网络、不做深度学习，完全高考生水平。

运行：python classify.py
"""

# ===== 导入需要的库 =====
import numpy as np                       # numpy：数值计算，图片像素数组处理
from PIL import Image                    # PIL：读取用户图片、图片缩放预处理
import matplotlib.pyplot as plt          # 绘图库，用来画混淆矩阵
from sklearn.datasets import load_digits          # 加载sklearn自带的手写数字数据集
from sklearn.model_selection import train_test_split   # 划分训练集和测试集的工具
from sklearn.svm import SVC                       # SVM支持向量机分类器（改进：替换KNN）
from sklearn.metrics import accuracy_score, confusion_matrix  # 准确率、混淆矩阵评估工具

# ===== 设置matplotlib中文字体（Windows用SimHei），防止图中中文乱码 =====
plt.rcParams["font.sans-serif"] = ["SimHei"]    # 设置中文字体为黑体
plt.rcParams["axes.unicode_minus"] = False       # 解决负号显示为方块的问题


# ============================================================
# 图片预处理函数【改进版】：把用户的任意图片转成模型能识别的格式
# ============================================================
# 模型训练时用的是 8×8 像素、像素值 0~16、黑底白字、数字居中 的图片。
# 用户上传的图片可能是任意大小、任意底色、数字位置不定，所以必须预处理。
#
# 预处理步骤（改进重点）：
#   1. 转灰度
#   2. 自动反色（确保黑底白字）
#   3. 找数字边界框，裁掉多余空白（保留灰度，不做二值化）
#   4. 用BILINEAR缩放到最大边7像素（保留灰度渐变/抗锯齿）
#   5. 居中放到8×8画布
#   6. 归一化到0~16
#
# 【关键改进说明】
#   之前版本做了"二值化"（纯黑白），但digits数据集是有0~16灰度渐变的
#   （扫描手写数字时边缘有抗锯齿）。二值化会丢失这些渐变信息，
#   导致预处理后的图片和训练数据分布不一致，识别率暴跌。
#   现在去掉二值化，用BILINEAR缩放保留灰度渐变，和训练数据匹配。
def preprocess_image(image_path):
    """
    读取用户图片，预处理成与 digits 数据集一致的 1×64 特征向量。
    参数：image_path —— 图片文件路径
    返回：1×64 的 numpy 数组（64个像素值，0~16）
    """
    # ----- 第1步：打开图片并转灰度图 -----
    # "L"模式 = 8位灰度图，像素值范围 0~255（0=黑，255=白）
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=float)

    # ----- 第2步：自动判断白底/黑底，反色成黑底白字 -----
    # 判断方法：看图片四个角的平均亮度，>127 说明角是白色（白底黑字），
    # digits 数据集是黑底白字，所以白底黑字需要反色。
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    if np.mean(corners) > 127:
        arr = 255 - arr    # 反色：白底黑字 → 黑底白字

    # ----- 第3步：找数字的边界框（改进核心） -----
    # 注意：不做二值化，保留灰度渐变信息（和digits数据集一致）。
    # 用阈值20判断哪些像素属于数字（背景是0，数字边缘即使较暗也>20）。
    # np.any(arr > 20, axis=1)：哪些行有数字（返回布尔数组）
    # np.any(arr > 20, axis=0)：哪些列有数字
    rows_has_digit = np.any(arr > 20, axis=1)
    cols_has_digit = np.any(arr > 20, axis=0)

    # 如果图片全黑（没有数字），返回全零向量
    if not rows_has_digit.any():
        print("  ⚠️ 警告：图片中没有检测到数字")
        return np.zeros((1, 64))

    # 找到数字的上下左右边界
    # np.where(rows_has_digit)[0] 返回有数字的行号数组，[0]取第一个（上边界），[-1]取最后一个（下边界）
    rmin, rmax = np.where(rows_has_digit)[0][[0, -1]]
    cmin, cmax = np.where(cols_has_digit)[0][[0, -1]]

    # 裁剪出数字区域（只保留有数字的部分，去掉四周空白）
    digit_arr = arr[rmin:rmax + 1, cmin:cmax + 1]

    # ----- 第4步：缩放到高度占满8像素（改进关键） -----
    # digits 数据集的预处理方式：把数字的高度缩放到8（占满整个画布高度），
    # 宽度按比例缩放，然后水平居中。这样数字占满高度，和训练数据分布一致。
    # 之前缩到7像素留了边距，导致底部空白、位置偏移，识别率低。
    # 【关键】用BILINEAR双线性插值缩放，保留灰度渐变/抗锯齿。
    h, w = digit_arr.shape
    scale = 8.0 / h                        # 高度缩放到8像素
    new_h = 8
    new_w = max(1, int(round(w * scale)))  # 宽度按比例
    # 如果宽度超过8，再按宽度等比缩小（保证不超出画布）
    if new_w > 8:
        scale = 8.0 / w
        new_w = 8
        new_h = max(1, int(round(h * scale)))

    # 用PIL缩放（BILINEAR双线性插值）
    digit_img = Image.fromarray(digit_arr.astype(np.uint8))
    digit_img = digit_img.resize((new_w, new_h), Image.BILINEAR)
    digit_scaled = np.array(digit_img, dtype=float)

    # ----- 第5步：水平居中放到8×8画布 -----
    # 高度已经占满8像素（top=0），宽度可能小于8，水平居中。
    canvas = np.zeros((8, 8), dtype=float)
    top = (8 - new_h) // 2     # 上边距（new_h=8时为0）
    left = (8 - new_w) // 2    # 左边距（水平居中）
    canvas[top:top + new_h, left:left + new_w] = digit_scaled

    # ----- 第6步：归一化到0~16 -----
    # digits 数据集的像素范围是0~16（不是0~255），必须统一。
    canvas = canvas / 255 * 16

    # 展平成 1×64 的二维数组（sklearn 的 predict 要求输入是二维）
    return canvas.reshape(1, -1)


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
# 第3步：训练SVM分类器（改进：替换KNN）
# ============================================================
# SVC 参数：
#   kernel='rbf' : 径向基核函数，处理非线性分类，手写数字识别效果好
#   C=10         : 正则化参数，C越大越努力拟合训练数据
#   gamma=0.001  : 核函数参数，控制决策边界的平滑度
# 这组参数是 digits 数据集的经典最优参数。
# SVM原理：在高维空间中找到一个最优分离超平面，把不同类别的数据分开。
# 对比KNN：SVM对噪声和手写风格差异更鲁棒，识别成功率更高。
model = SVC(kernel="rbf", C=10, gamma=0.001)

# fit()：用训练集训练模型（模型从 X_train 学习像素规律，对照 y_train 的答案）
model.fit(X_train, y_train)
print("模型训练完成（SVM支持向量机）")

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
# 第7步：用户输入图片路径，识别并输出结果
# ============================================================
print("\n" + "=" * 50)
print("手写数字识别系统（改进版）")
print("=" * 50)
print("示例图片在 sample_images/ 文件夹中，例如：sample_images/digit_5.png")
print("支持 PNG/JPG，黑底白字或白底黑字均可（自动反色）")
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
