# GSL-GCRNN 频谱预测算法复现

本目录实现论文《Spectrum Prediction via Graph Structure Learning》中的 GSL-GCRNN 核心流程：利用图卷积网络（GCN）提取频段关联，通过图结构学习（GSL）更新频段间连接，再使用门控循环单元（GRU）和多层感知机（MLP）预测未来功率谱密度（PSD）。

本文档仅针对 `gsl_gcrnn/`，不涉及其他改进版本。该代码是论文方法的实现，实验结果是否达到论文水平需要通过完整训练验证。

## 1. 文件结构

```text
gsl_gcrnn/
├── README.md        # 使用说明
├── config.py        # 数据路径、模型参数与训练参数
├── data_loader.py   # CSV 读取、标准化和滑动窗口数据集
├── graph.py         # 相关系数计算、Top-k 构图和邻接矩阵加载
├── models.py        # GCN、图结构估计器、GRU 和 MLP
├── train.py         # 单轮训练、评估和指标计算
└── main.py          # 命令行入口与训练流程
```

## 2. 环境准备

需要 Python，以及以下依赖：

```powershell
python -m pip install torch numpy pandas scipy
```

代码使用原生 PyTorch 实现图卷积，无需安装 PyTorch Geometric。上述命令不锁定依赖版本；使用 GPU 时，需要确保安装的 PyTorch 支持本机 CUDA 环境。默认自动选择可用的 CUDA，否则使用 CPU。

## 3. 数据准备

数据位于项目根目录的 `data/` 下，而不是本目录中。程序分别读取已经划分的训练集和测试集，不会再次按 80% / 20% 切分。

```text
data/
├── trainData/
│   ├── test_yago/
│   ├── test_rpi4/
│   ├── rack_2/
│   └── bcn_L/
└── testData/
    ├── test_yago/
    ├── test_rpi4/
    ├── rack_2/
    └── bcn_L/
```

每个传感器文件夹中应包含对应的频谱 CSV：

| 传感器参数 | CSV 文件名 |
| --- | --- |
| `test_yago` | `test_yago_202106010800-202106080800_1mins_2MHz.csv` |
| `test_rpi4` | `test_rpi4_202106010800-202106080800_1mins_2MHz.csv` |
| `rack_2` | `rack_2_202106010800-202106080800_1mins_2MHz.csv` |
| `bcn_L` | `bcn-L_202106010800-202106080800_1mins_2MHz.csv` |

CSV 读取时固定跳过第一行；其余内容必须是数值矩阵，形状为 `(时间步数, 101)`，不应包含额外的时间戳列。更换数据时需相应调整 `config.py` 中的 `N_BANDS`、路径及入口中的文件命名规则。

预处理流程：

1. 仅用训练集计算每个频段的均值和标准差。
2. 使用相同统计量标准化训练集和测试集；标准差小于 `1e-8` 时设为 1。
3. 默认仅用训练集原始 PSD 构建初始相关图。
4. 分别在训练集和测试集内部生成滑动窗口，不跨越两者边界。

默认每个样本使用过去 16 个时间步预测未来 1 个时间步。输入形状为 `(B, 16, 101)`，标签和输出形状为 `(B, 1, 101)`；长度为 `L` 的序列生成 `L - WINDOW - HORIZON + 1` 个样本。

## 4. 快速运行

以下命令均在项目根目录执行：

```powershell
# 查看命令行参数
python gsl_gcrnn/main.py --help

# 快速检查数据读取、训练和评估流程
python gsl_gcrnn/main.py --sensor bcn_L --epochs 2 --device cpu

# 按默认配置训练单个传感器（3000 轮）
python gsl_gcrnn/main.py --sensor bcn_L

# 依次训练四个传感器，每个传感器单独建立模型
python gsl_gcrnn/main.py

# 指定 GPU 和构图方式
python gsl_gcrnn/main.py --sensor test_rpi4 --device cuda --graph-method pearson
```

也可进入本目录后执行 `python main.py ...`。当前代码采用同目录模块导入，建议直接运行脚本，不使用 `python -m gsl_gcrnn.main`。

两轮运行用于检查流程，不代表模型已收敛。运行会写入或覆盖对应的输出文件。

### 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
