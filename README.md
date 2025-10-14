# richCode - 体彩预测项目

基于 Transformer 的彩票号码预测深度学习项目。 

⚠️ 本项目仅供学习交流使用，彩票预测结果不具有实际参考价值。

## 项目简介

1、本项目使用 Transformer 编码器架构，通过学习历史彩票数据来预测下一期的彩票号码。

2、模型使用 Embedding 层将数字转换为高维向量，通过多层 Transformer 进行特征提取，最终为 7 个位置分别预测 0-9 的数字。

3、数据来源：中国体育彩票网，江苏区域 7 位彩票最近 20 年的彩票数据。

## 项目结构

```
lottery-demo/
├── data/                           # 数据目录
│   └── lottery_data.xlsx          # 彩票历史数据
├── datasets/                       # 数据集模块
│   ├── custom_datasets.py         # 自定义数据集类
│   ├── get_all_data_request.py    # 数据爬取（requests）
│   └── get_all_data_selenium.py   # 数据爬取（selenium）
├── models/                         # 模型模块
│   └── base_model.py              # Transformer 基础模型
├── utils/                          # 工具模块
│   ├── train_demo.py              # 训练脚本
│   └── write_caipiao_data_to_excel.py
├── model_checkpoint/               # 模型保存目录
└── train_logs/                     # TensorBoard 日志目录
```

## 模型架构

- **输入**: `[batch_size, seq_len, 7]` - 历史 N 期数据的 7 个位置
- **Embedding**: 将每个数字（0-9）映射到高维向量空间
- **Transformer Encoder**: 多层注意力机制提取时序特征
- **输出**: `[batch_size, 7, 10]` - 7 个位置各 10 个类别的概率分布


## 配置参数

在 `utils/train_demo.py` 中可调整：

- `embedding_dims`: 嵌入维度（默认 1024）
- `batch_size`: 批次大小（默认 32）
- `epochs`: 训练轮数（默认 10）
- `learning_rate`: 学习率（默认 0.0001）
- `need_day`: 使用历史数据期数（默认 5）

#  <p align="center">Star history</p>
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lrc-zcl/richCode&type=Timeline)](https://star-history.com/#lrc-zcl/richCode&Timeline)