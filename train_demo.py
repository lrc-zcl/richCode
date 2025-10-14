import os
import torch
from torch.utils.data import DataLoader
from models.base_model import BaseModel
from datasets.custom_datasets import MyDatasets
from torch.nn import functional as F
from tensorboardX import SummaryWriter

writer = SummaryWriter(logdir='./train_logs/20251013')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    excel_path = "./data/lottery_data.xlsx"
    datasets = MyDatasets(excel_path)
    if os.path.exists("./model_checkpoint/model.pth"):
        model = torch.load("./model_checkpoint/model.pth")
    else:
        model = BaseModel(1024)
    model.to(device=device)
    custom_dataloader = DataLoader(datasets, batch_size=64, shuffle=False, drop_last=True)
    epochs = 10
    loss_function = F.cross_entropy
    learning_rate = 0.00001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    step_counts = 0
    for epoch in range(epochs):
        for train_x, train_y in custom_dataloader:
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            step_counts = step_counts + 1
            optimizer.zero_grad()
            model_result = model.forward(train_x)
            loss_value = loss_function(model_result.transpose(1, 2), train_y.long())
            loss_value.backward()  # 执行反向传播
            optimizer.step()

            writer.add_scalar('train_loss', loss_value.item(), step_counts)
            print(f"{epoch + 1}/{epochs} ------------------------------{loss_value.item()}")

    # 保存模型
    torch.save(model, "./model_checkpoint/model.pth")

    # 保存模型结构图
    dummy_input = torch.randint(0, 10, (64, 10, 7))  # 生成0-9之间的随机整数，匹配embedding层的范围
    writer.add_graph(model=model, input_to_model=dummy_input)
