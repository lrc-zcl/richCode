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
    if os.path.exists("model_checkpoints/model.pth"):
        model = torch.load("model_checkpoints/model.pth")
    else:
        model = BaseModel(256)
    model.to(device=device)
    custom_dataloader = DataLoader(datasets, batch_size=64, shuffle=False, drop_last=True)
    epochs = 100
    loss_function = F.cross_entropy
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    step_counts = 0
    for epoch in range(epochs):
        for train_x, train_y in custom_dataloader:
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            step_counts = step_counts + 1
            optimizer.zero_grad()
            model_result = model.forward(train_x)
            calculate_result = model_result.transpose(1, 2)
            loss_value = loss_function(calculate_result, train_y.long())
            loss_value.backward()  # 执行反向传播
            optimizer.step()
            writer.add_scalar('train_loss', loss_value.item(), step_counts)
            print(
                f"{epoch + 1}/{epochs} ------------------------------{loss_value.item()} ----- 当前学习率是 {optimizer.param_groups[0].get('lr')}")
        scheduler.step()  # 学习率更新
    torch.save(model, "model_checkpoints/model.pth")
    # 保存模型结构图
    dummy_input = torch.randint(0, 10, (64, 10, 7)).to(device)  # 生成0-9之间的随机整数，匹配embedding层的范围
    writer.add_graph(model=model, input_to_model=dummy_input)
