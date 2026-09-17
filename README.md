# 方案A：图片分类——手写数字识别（监督学习）

对应浙江信息技术高考考点：**监督学习、训练集/测试集划分、计算机视觉、模型评估**。

## 做了什么
使用 sklearn 自带的手写数字数据集（1797张 8×8 手写数字图片，标签0~9），
用 K近邻(KNN) 分类器训练模型。训练完成后，**用户可以通过输入图片文件路径，让程序识别图片中的手写数字并输出结果**。

## 核心流程（与高考考点对应）
1. 加载带标签的数据集 → **监督学习**（数据有标签）
2. 划分训练集(70%)和测试集(30%) → **数据集划分**（高考高频考点）
3. 用训练集训练KNN模型 → **模型训练**
4. 用测试集预测，计算准确率 → **模型评估**
5. 混淆矩阵查看哪些数字容易认错 → **错误分析**
6. 用户输入图片路径 → 图片预处理 → 模型识别 → 输出结果

## 新增功能：用户输入图片识别
- 程序运行后，提示用户输入图片文件路径
- 支持 PNG、JPG 等常见图片格式
- 自动判断白底黑字/黑底白字，自动反色处理
- 自动缩放到 8×8、归一化像素值，与训练数据格式一致
- 输入 `quit` 退出程序

## 文件结构
```
A_image_classification/
├── classify.py                 # 主程序：训练模型 + 用户输入图片识别
├── generate_sample_images.py   # 生成示例数字图片（0~9各一张）
├── requirements.txt
├── README.md
├── sample_images/              # 示例数字图片（运行generate脚本后生成）
│   ├── digit_0.png ~ digit_9.png
└── output/
    └── digits_result.png       # 混淆矩阵图
```

## 运行步骤
```
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成示例数字图片（只需运行一次）
python generate_sample_images.py

# 3. 运行主程序
python classify.py

# 4. 程序提示输入图片路径时，输入示例图片路径，例如：
sample_images/digit_5.png

# 5. 输入 quit 退出
```

## 输出
- 控制台：数据集大小、训练/测试集数量、准确率、混淆矩阵
- 交互提示：输入图片路径后，输出识别结果（如"✅ 识别结果：数字 5"）
- `output/digits_result.png`：混淆矩阵图

## 高考生学习要点
- 只用了 sklearn 的：`load_digits`、`train_test_split`、`KNeighborsClassifier`、`accuracy_score`、`confusion_matrix`
- 图片预处理用了 PIL（`Image.open`、`convert`、`resize`）和 numpy
- 没有神经网络、没有深度学习，全部在考纲范围内
- 关键概念：训练集用来"教"，测试集用来"考"，两者不能混用
- 图片预处理是计算机视觉的重要步骤：用户图片必须转成和训练数据一致的格式，模型才能识别
