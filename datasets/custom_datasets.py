import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset


class MyDatasets(Dataset):
    def __init__(self, excel_path=None, need_day=10):
        self.excel_path = excel_path
        super(MyDatasets, self).__init__()
        self.need_data = need_day
        self.excel_data = pd.read_excel(self.excel_path)
        self.train_data = self.excel_data.values[0:3300, 0:9]
        self.reversed_data = self.train_data[::-1].copy()
        self.final_train_data = self.reversed_data[:, 2:9].astype('float32')

    def __len__(self):
        return self.final_train_data.shape[0] - self.need_data

    def __getitem__(self, idx):
        train_data = self.final_train_data[idx:idx + self.need_data, :]
        train_label_data = self.final_train_data[idx + self.need_data, :]
        train_data = torch.from_numpy(train_data)
        train_label_data = torch.from_numpy(train_label_data)
        return train_data, train_label_data


class MyTestDatasets(Dataset):
    def __init__(self, excel_path=None, need_day=10):
        self.excel_path = excel_path
        super(MyTestDatasets, self).__init__()
        self.need_data = need_day
        self.excel_data = pd.read_excel(self.excel_path)
        self.test_data = self.excel_data.values[3300:, 0:9]
        self.reversed_data = self.test_data[::-1].copy()
        self.final_test_data = self.reversed_data[:, 2:9].astype('float32')

    def __len__(self):
        return self.final_test_data.shape[0] - self.need_data

    def __getitem__(self, idx):
        test_data = self.final_test_data[idx:idx + self.need_data, :]
        test_label_data = self.final_test_data[idx + self.need_data, :]
        test_data = torch.from_numpy(test_data)
        test_label_data = torch.from_numpy(test_label_data)
        return test_data, test_label_data


if __name__ == "__main__":
    excel_path = "../data/lottery_data.xlsx"
    datasets = MyDatasets(excel_path)
    le = datasets.__len__()
    # dataloader = DataLoader(datasets, batch_size=16, shuffle=False, drop_last=True)
    # for x, y in dataloader:
    #     print(f"当前的训练 X 数据是 ------{x}")
    #     print(f"当前的训练 Y 数据是 ------{y}")

    """
    测试数据
    """
    test_datasets = MyTestDatasets(excel_path)
    test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False, drop_last=True)
    for test_x, test_y in test_dataloader:
        print(f"当前的训练 X 数据是 ------{test_x}")
        print(f"当前的训练 Y 数据是 ------{test_y}")
