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
    model = torch.load("model_checkpoints/model.pth")
    test_dataloader = DataLoader(datasets, batch_size=1, shuffle=False, drop_last=True)
    model.eval()
    with torch.no_grad():
        for test_x, test_y, start_index, end_index, predict_index in test_dataloader:
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            model_predict = model(test_x)
            predict_number = torch.argmax(model_predict, dim=2)
            print("*" * 80)
            print(f"使用期号 === {start_index[0]} === {end_index[0]} ===进行预测=== {predict_index[0]}")
            print(f"当前的预测结果是{predict_number.cpu().numpy()[0,]} ---- 正确结果是 {test_y.cpu().numpy()[0,]}")
            print("*" * 80)
            pass
    print("执行完成")
