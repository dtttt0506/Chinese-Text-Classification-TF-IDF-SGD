# Chinese-Text-Classification-TF-IDF-SGD

一个基于 TF-IDF 特征与多种经典机器学习分类器（SGD/Naive Bayes/SVM/Logistic Regression）的中文文本分类项目，支持 20 类中文语料。

## 功能特性
- 中文分词与停用词过滤（jieba + 自定义停用词表）
- TF-IDF 特征提取（支持 unigram/bigram，可配置特征数与阈值）
- 三种训练模式：
  - `sgd`：增量训练（`SGDClassifier`），适合大数据与快速迭代
  - `grid`：朴素贝叶斯/线性SVM/逻辑回归的网格搜索（含交叉验证）
  - `fast`：`HashingVectorizer` + `SGDClassifier` 的高效模式
- 训练过程日志、模型保存、评估报告与混淆矩阵
- 交互式预测脚本与标签空间保存

## 目录结构
```
├── config/
│   ├── cn_stopwords.txt        # 中文停用词
│   └── config.yaml             # 项目配置
├── data/
│   ├── train/                  # 训练数据（按类别文件夹组织）
│   ├── val/                    # 验证数据（可选，或自动划分）
│   └── test/                   # 测试数据（按类别文件夹组织）
├── src/
│   ├── data.py                 # 数据加载与划分
│   ├── preprocess.py           # 分词、向量器构建
│   ├── trainers.py             # 三种训练模式的实现
│   ├── evaluate.py             # 评估与可视化
│   ├── predict.py              # 预测与标签空间保存
│   └── utils.py                # 工具与日志
├── run.py                      # 主入口：读取配置并运行训练评估
├── interactive_predict.py      # 交互式预测脚本
├── requirements.txt            # 依赖
└── .gitignore
```

## 环境准备
- Python ≥ 3.10（推荐）
- 安装依赖：
```
pip install -r requirements.txt
```

## 快速开始
1. 准备数据集（按类别目录组织）：
```
data/
  train/
    C11-Space/
    C15-Energy/
    ...
  test/
    C11-Space/
    C15-Energy/
    ...
  val/        # 可为空，若为空则自动按比例划分
```

2. 配置训练模式（`config/config.yaml`）：
```
train:
  mode: "sgd"  # 可选: "sgd" | "grid" | "fast"
```

3. 运行训练与评估：
```
python run.py
```

完成后会在以下位置生成输出：
- `checkpoints/best_pipeline.joblib`：训练好的最佳模型
- `output/confusion_matrix.png`：混淆矩阵图
- `output/report.txt`：分类报告（精度、召回、F1等）
- `logs/`：运行日志

## 训练模式说明
### 1) SGD 模式（默认）
- 模型：`SGDClassifier`
- 损失：`log_loss`（逻辑回归）
- 关键配置（`config.yaml`）：
```
preprocess:
  use_bigram: true
  max_features: 30000
  min_df: 2

train:
  sgd:
    loss: "log_loss"
    max_epochs: 5
    batch_size: 512
    eval_every: 1
    learning_rate: 0.0005
```

### 2) Grid 模式
- 模型：`MultinomialNB` / `LinearSVC` / `LogisticRegression`
- 使用分层交叉验证（`cv_folds` 可配），自动搜索最优参数并保存最佳管道。

### 3) Fast 模式
- 模型：`HashingVectorizer` + `SGDClassifier`
- 特点：无需拟合向量器，内存占用低，适合快速原型与大规模文本。

## 预测使用
### 方式一：脚本交互
```
python interactive_predict.py
```
按提示输入中文文本，输出预测类别。

### 方式二：代码调用
```python
import joblib

model = joblib.load("checkpoints/best_pipeline.joblib")
text = "这是一个关于计算机技术的中文段落。"
pred = model.predict([text])[0]
print(pred)
```

## 配置文件说明（`config/config.yaml`）
- `paths.*`：数据与输出目录
- `preprocess.*`：TF-IDF特征参数（是否使用bigram、特征数、最小df）
- `train.mode`：三种训练模式切换
- `train.sgd.*`：SGD相关超参数
- `grid_params.*`：网格搜索各模型的参数空间

## 常见问题
- 序列化问题：项目内通过 `JiebaCutTransformer` + `tokenizer=str.split` 的方式，确保完整 `Pipeline` 可使用 `joblib` 安全保存与加载。
- 目录不存在：确保 `config.yaml` 中的路径与实际目录一致，`run.py` 会创建必要目录。
- 类别不均衡：可考虑在数据准备阶段做采样或在模型上调整类别权重。

## 许可证
未设置开源许可证。如需开源，请自行添加 `LICENSE` 文件。

## 致谢
- 结巴分词（jieba）
- scikit-learn