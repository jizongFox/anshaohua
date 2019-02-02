from operator import itemgetter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
from tqdm import tqdm
from torchnet.meter import AverageValueMeter

txrx = scipy.io.loadmat('TxRxQAMSymbols4SPS.mat')['TxRxQAMSymbols4SPS']

received_signal = np.stack([txrx[:, 1].real, txrx[:, 1].imag], axis=1)
## preprocessing data

Y_ = txrx[:, 0]
Y_[1::4] = Y_[0::4]
Y_[2::4] = Y_[0::4]
Y_[3::4] = Y_[0::4]

Y_coding = {str(i): j for i, j in zip(np.sort(np.unique(txrx[:, 0])), range(0, 20))}
mapping = lambda x: Y_coding[str(x)]
vmapping = np.vectorize(mapping)
gt = vmapping(Y_)

assert received_signal.shape[0] == gt.shape[0]

train_X = received_signal[:int(received_signal.__len__() * 0.8)]
train_y = gt[:int(received_signal.__len__() * 0.8)]
test_X = received_signal[int(received_signal.__len__() * 0.8):]
test_y = gt[int(received_signal.__len__() * 0.8):]
print('train dataset length:', train_y.__len__())
print('test dataset length:', test_y.__len__())

assert train_y.__len__() + test_y.__len__() == received_signal.__len__()


## dataset
class SignalDataset(Dataset):

    def __init__(self, X: np.ndarray, Y: np.ndarray, range=25) -> None:
        super().__init__()
        assert X.shape[0] == Y.shape[0]
        self.X = X
        self.Y = Y
        self.range = range

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if index <= self.range + 1 or index >= self.__len__() - self.range - 1:
            index = np.random.randint(self.range + 1, self.__len__() - self.range - 1)
            return self.__getitem__(index)
        else:
            x, y = self.X[index - self.range:index + self.range].T, self.Y[index]
            return x, y, index


def center_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batchindexgetter = itemgetter(2)
    batch = list(filter(lambda x: batchindexgetter(x) % 4 == 0, batch))
    return default_collate(batch)


train_dataset = SignalDataset(train_X, train_y)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=100, )  # collate_fn=center_collate)
test_dataset = SignalDataset(test_X, test_y)
test_loader = DataLoader(test_dataset, 100, collate_fn=center_collate)


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=7, ),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
        )
        self.feature2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
        )
        self.feature3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )
        self.feature4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )
        self.classfier = nn.Sequential(
            nn.Linear(64, 16)
        )

    def forward(self, input):
        out = self.feature1(input)
        out = self.feature2(out)
        out = self.feature3(out)
        out = self.feature4(out)
        out = F.adaptive_avg_pool1d(out, output_size=1)
        out = out.view(out.shape[0], -1)
        out = self.classfier(out)
        return out


cnn = CNN()
optimiser = torch.optim.Adam(cnn.parameters())

# output = cnn(X.float())
# print(output.shape)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
citerion = nn.CrossEntropyLoss()


def _main_loop(dataloader, mode='train'):
    acc_meter = AverageValueMeter()
    dataloader = tqdm(dataloader)
    for i, [signal, gt, _] in enumerate(dataloader):

        signal, gt = signal.float().to(device), gt.long().to(device)
        if mode == "train":
            cnn.train()
        else:
            cnn.eval()
        pred = cnn(signal)
        if mode == "train":
            loss = citerion(pred, gt)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        acc_meter.add(compute_acc(pred, gt))
        dataloader.set_postfix({"acc": float(acc_meter.value()[0])})

    return acc_meter.value()[0]


def compute_acc(pred, gt):
    pred = pred.max(1)[1]
    assert pred.shape == gt.shape
    acc = (pred == gt).sum().float() / int(gt.view(-1).size()[0])

    return acc


def main():
    for i in range(1000):
        train_acc = _main_loop(train_loader, mode='train')
        test_acc = _main_loop(test_loader, mode='eval')

    print(f'@ epoch {i}, train_acc:{train_acc}, test_acc{test_acc}\t')


if __name__ == '__main__':
    main()
