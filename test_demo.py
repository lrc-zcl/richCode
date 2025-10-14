import os
import torch
from torch.utils.data import DataLoader
from models.base_model import BaseModel
from datasets.custom_datasets import MyDatasets, MyTestDatasets
from torch.nn import functional as F
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    excel_path = "./data/lottery_data.xlsx"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = MyTestDatasets(excel_path)
    model = torch.load("./model_checkpoints/model.pth")
    test_dataloader = DataLoader(datasets, batch_size=64, shuffle=False, drop_last=True)
    for test_x, test_y in test_dataloader:
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        model_predict = model(test_x)
        print(f"当前的预测结果是{model_predict} ---- 正确结果是 {test_y}")

    print("执行完成")
