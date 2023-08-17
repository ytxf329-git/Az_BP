import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

fix_seed = 9
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


class new_model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.dimensionality_dense = nn.Sequential(
            nn.Linear(3, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 3),
            nn.Tanh())
            # nn.Linear(6, 6),
            # nn.Tanh(),
            # nn.Linear(6, 3),
            # nn.Tanh())
        self.norm = nn.LayerNorm(3)
        self.dense = nn.Linear(3, 1)

    def forward(self, x):
        x = self.dimensionality_dense(x)
        x = self.norm(x)
        x = self.dense(x)
        return x


def Train(model, dataset_x, dataset_y, test_x, test_y, EPOCH):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # scale1 = StandardScaler()
    # dataset_x = scale1.fit_transform(dataset_x)
    # scale2 = StandardScaler()
    # test_x = scale2.fit_transform(test_x)
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
    BATCH_SIZE = 32
    model = model.train()
    train_loss = []
    print('Start training...')
    print('cuda(GPU)是否可用:', torch.cuda.is_available())
    print('torch的版本:', torch.__version__)
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
    torch.save(model, "model.pkl")
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

    plt.figure()
    Az_error = test_y.flatten() - pred.flatten()
    plt.plot(Az_error, label='Az_error')
    plt.legend()

    fig, ax = plt.subplots()
    plt.plot(pred.flatten(), test_y.flatten(), ls='', marker='o', color='#003C9D', markersize=5, alpha=0.2)
    min_, max_ = min(np.min(pred), np.min(test_y)), max(np.max(pred), np.max(test_y))
    plt.plot([min_, max_], [min_, max_], color='black', ls='--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.show()


if __name__ == "__main__":
    df = pd.read_excel('Az_data.xlsx')
    data = np.array(df)
    np.random.shuffle(data)
    data_x = np.array(data)[:, 1:]
    data_y = np.array(data)[:, 0]
    N = data_x.shape[0]

    train_x = data_x[:int(N * 0.8), :]
    train_y = data_y[:int(N * 0.8)]
    test_x = data_x[int(N * 0.8):, :]
    test_y = data_y[int(N * 0.8):]

    model = new_model()
    EPOCH = 500
    Train(model, train_x, train_y, test_x, test_y, EPOCH)
