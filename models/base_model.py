"""
搭建基础模型
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # 创建位置编码矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 创建位置索引

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 注册为buffer，不参与梯度更新

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BaseModel(nn.Module):
    def __init__(self, embedding_dims):
        super(BaseModel, self).__init__()
        self.embedding_dims = embedding_dims
        self.embeddingLinear = nn.Embedding(num_embeddings=10, embedding_dim=self.embedding_dims)
        self.embedding_norm = nn.LayerNorm(self.embedding_dims)

        # 添加正弦位置编码
        self.pos_encoding = PositionalEncoding(d_model=self.embedding_dims, max_len=100, dropout=0.1)

        self.transformerEncoderLayer = nn.TransformerEncoderLayer(d_model=self.embedding_dims, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformerEncoderLayer, num_layers=3)
        self.output_layers = nn.ModuleList([
            nn.Linear(self.embedding_dims, 10) for _ in range(7)  # 7个位置分别预测
        ])

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dims))  # 设置一个可学习的token 用于最后输送至 分类层,类似一个标志位

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        num_positions = x.size(2)  # 输入的数据,batch seq_len(5), 7
        outputs = []

        for i in range(num_positions):
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, embedding_dims]
            position_data = x[:, :, i]  # 提取第i个位置的历史序列 [batch, 10]
            embedded = self.embeddingLinear(position_data.long())  # [batch, 10, embedding_dims]
            embedded = self.embedding_norm(embedded)  # [batch, 10, embedding_dims]

            # 添加位置编码
            embedded = self.pos_encoding(embedded)  # [batch, 10, embedding_dims]

            embedded_with_cls = torch.cat([cls_tokens, embedded], dim=1)  # [batch, 11, embedding_dims]

            transformer_out = self.transformer(embedded_with_cls)  # [batch, 5, embedding_dims]
            cls_key_step = transformer_out[:, 0, :]  # [batch, embedding_dims]  取关键的标志位送至分类层
            output = self.output_layers[i](cls_key_step)  # [batch, 10]
            #output = self.softmax(output)
            outputs.append(output)
        return torch.stack(outputs, dim=1)  # [batch, 7, 5]


if __name__ == "__main__":
    input_data = torch.randint(0, 9, (64, 10, 7))
    model = BaseModel(embedding_dims=1024)
    reslt = model(input_data)
    print(reslt)
