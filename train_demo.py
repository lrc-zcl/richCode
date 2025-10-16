import os
import torch
from torch.utils.data import DataLoader
from models.base_model import BaseModel
from datasets.custom_datasets import MyDatasets, MyTestDatasets
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from utils.train_utils import calculate_detailed_accuracy
writer = SummaryWriter(logdir='./train_logs/20251013')



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    excel_path = "./data/lottery_data.xlsx"

    # 训练数据集
    datasets = MyDatasets(excel_path)
    custom_dataloader = DataLoader(datasets, batch_size=64, shuffle=True, drop_last=True)

    # 验证数据集
    val_datasets = MyTestDatasets(excel_path)
    val_dataloader = DataLoader(val_datasets, batch_size=64, shuffle=True, drop_last=True)
    val_dataloader_iter = iter(val_dataloader)

    if os.path.exists("model_checkpoints/model.pth"):
        model = torch.load("model_checkpoints/model.pth")
    else:
        model = BaseModel(128)
    model.to(device=device)
    epochs = 300
    loss_function = F.cross_entropy
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    step_counts = 0

    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        param_count = p.numel()  # 获取tensor中元素的个数
        total_params += param_count
        if p.requires_grad:
            trainable_params += param_count
    print(f"模型总计参数量 {total_params} ------ 模型可训练参数量 {trainable_params}")
    for epoch in range(epochs):
        for train_x, train_y in custom_dataloader:
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            step_counts = step_counts + 1

            # 训练步骤
            optimizer.zero_grad()
            model_result = model.forward(train_x)
            calculate_result = model_result.transpose(1, 2)
            loss_value = loss_function(calculate_result, train_y.long())
            loss_value.backward()  # 执行反向传播
            optimizer.step()

            if step_counts % 10 == 0:
                model.eval()
                with torch.no_grad():
                    try:
                        test_x, test_y, _, _, _ = next(val_dataloader_iter)
                    except StopIteration:
                        val_dataloader_iter = iter(val_dataloader)
                        test_x, test_y, _, _, _ = next(val_dataloader_iter)
                    test_x = test_x.to(device)
                    test_y = test_y.to(device)
                    model_predict = model(test_x)
                    calculate_result = model_predict.transpose(1, 2)
                    val_loss_value = loss_function(calculate_result, test_y.long())
                    current_step_accuracy = calculate_detailed_accuracy(predictions=model_predict, targets=test_y)
                    print(
                        f"{epoch + 1}/{epochs}--------train_loss--{loss_value.item()}--------val_loss--{val_loss_value.item()}---acc--{current_step_accuracy[0]}---current learning rate--{optimizer.param_groups[0].get('lr')}")
                    writer.add_scalar('val_loss', val_loss_value.item(), step_counts)
                    writer.add_scalar('train_loss', loss_value.item(), step_counts)
                model.train()
        scheduler.step()
    torch.save(model, "model_checkpoints/model.pth")
