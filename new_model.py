import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class new_model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.dimensionality_dense = nn.Sequential(
            nn.Linear(3, 100),
            nn.GELU(),
            nn.Linear(100, 20))
        self.dense1 = nn.Linear(3, 20)
        self.norm = nn.LayerNorm(20)
        self.dense = nn.Linear(20, 1)

    def forward(self, x):
        x = self.dense1(x) + self.dimensionality_dense(x)
        x = self.norm(x)
        x = self.dense(x)
        return x


def Train(model, dataset_x, dataset_y, test_x, test_y, EPOCH):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_x = torch.from_numpy(dataset_x).to(torch.float).to(device)
    train_y = torch.from_numpy(dataset_y.reshape(-1, 1)).to(torch.float).to(device)
    test_x = torch.from_numpy(test_x).to(torch.float).to(device)
    test_y = test_y
    loss_fn = nn.MSELoss()
    loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    # 组建数据开始训练
    model.to(device)
    torch_dataset = TensorDataset(train_x, train_y)
    BATCH_SIZE = 100
    model = model.train()
    train_loss = []
    print('Start training...')
    for i in tqdm(range(EPOCH), colour='red'):
        loader = DataLoader(dataset=torch_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)
        temp_1 = []
        for step, (batch_x, batch_y) in enumerate(loader):
            out = model(batch_x)
            optimizer.zero_grad()
            loss = loss_fn(out, batch_y)
            temp_1.append(loss.item())
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        train_loss.append(np.mean(np.array(temp_1)))
    # if i % 10 == 0:
    # 	print("The {} is end...".format(i))
    print('Training end...')
    model = model.eval()
    # 拟合数据
    pred = model(test_x)
    pred = pred.cpu().data.numpy()
    plt.plot(train_loss)

    plt.figure()
    train_res = model(train_x)
    train_res = train_res.cpu().data.numpy()
    plt.plot(train_res, label='train')
    plt.plot(train_y.cpu().data.numpy(), label='true')
    plt.legend()

    plt.figure()
    plt.plot(test_y.flatten(), label='true')
    plt.plot(pred, label='pred')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    df = pd.read_excel('Az_data.xlsx')
    data_x = np.array(df)[:, :-1]
    data_y = np.array(df)[:, -1]
    N = data_x.shape[0]
    train_x = data_x[:int(N * 0.8), :]
    train_y = data_y[:int(N * 0.8)]
    test_x = data_x[int(N * 0.8):, :]
    test_y = data_y[int(N * 0.8):]

    model = new_model()
    EPOCH = 1000
    Train(model, train_x, train_y, test_x, test_y, EPOCH)
