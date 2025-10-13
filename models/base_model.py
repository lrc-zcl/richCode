"""
搭建基础模型
"""
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, embedding_dims):
        super(BaseModel, self).__init__()
        self.embedding_dims = embedding_dims
        self.embeddingLinear = nn.Embedding(num_embeddings=10, embedding_dim=self.embedding_dims)
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(d_model=self.embedding_dims, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformerEncoderLayer, num_layers=6)
        self.output_layers = nn.ModuleList([
            nn.Linear(self.embedding_dims, 10) for _ in range(7)  # 7个位置分别预测
        ])

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dims))  # 设置一个可学习的token 用于最后输送至 分类层,类似一个标志位

    def forward(self, x):
        batch_size = x.size(0)
        num_positions = x.size(2)  # 输入的数据,batch seq_len(5), 7
        outputs = []

        for i in range(num_positions):
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, embedding_dims]
            position_data = x[:, :, i]  # 提取第i个位置的历史序列 [batch, 5]
            embedded = self.embeddingLinear(position_data.long())  # [batch, 5, embedding_dims]
            embedded_with_cls = torch.cat([cls_tokens, embedded], dim=1)  # [batch, 6, embedding_dims]

            transformer_out = self.transformer(embedded_with_cls)  # [batch, 5, embedding_dims]
            cls_key_step = transformer_out[:, 0, :]  # [batch, embedding_dims]  取关键的标志位送至分类层
            output = self.output_layers[i](cls_key_step)  # [batch, 5]
            outputs.append(output)
        return torch.stack(outputs, dim=1)  # [batch, 7, 5]


if __name__ == "__main__":
    model = BaseModel(embedding_dims=1024)
