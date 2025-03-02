# coding = gbk

import torch
from torch import nn
from torch.nn import init
import numpy as np
import pandas as pd
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler
from utils import addattack


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 256

#data preprocess
data_train = pd.read_csv("../data_train_NSL_KDD.csv")
data_test = pd.read_csv("../data_test_NSL_KDD.csv")
data_train = np.array(data_train)
data_test = np.array(data_test)
X_train, y_train = data_train[:, :41], data_train[:, [41]]
X_test, y_test = data_test[:, :41], data_test[:, [41]]

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dataset_train = np.hstack((X_train_scaled, y_train))
dataset_test = np.hstack((X_test_scaled, y_test))

dataset_train_data = torch.from_numpy(dataset_train[:,0:41]).float()
dataset_train_label = torch.from_numpy(dataset_train[:,41:42]).long()

dataset_test_data = torch.from_numpy(dataset_test[:,0:41]).float()
dataset_test_label = torch.from_numpy(dataset_test[:,41:42]).long()

train_dataset = Data.TensorDataset(dataset_train_data, dataset_train_label)
train_iter = Data.DataLoader(train_dataset, batch_size, shuffle=True)

test_dataset = Data.TensorDataset(dataset_test_data, dataset_test_label)
test_iter = Data.DataLoader(test_dataset, batch_size, shuffle=True)

# DNN64
net = nn.Sequential(
    nn.Linear(41, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 5)
).to(device)

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
num_epochs = 50
train_list_acc = []
test_list_acc = []

# training
# for epoch in range(1, num_epochs + 1):
#
#     train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
#     test_acc_sum = 0.0
#
#     for data, label in train_iter:
#         data = data.to(device)
#         label = label.to(device)
#         label = label.squeeze()
#         output = net(data)
#         l = loss(output, label).sum()
#         optimizer.zero_grad()
#         l.backward()
#         optimizer.step()
#         train_l_sum += l.item()
#         train_acc_sum += (output.argmax(dim=1) == label).sum().item()
#         n += label.shape[0]
#
#     with torch.no_grad():
#         dataset_test_data = dataset_test_data.to(device)
#         dataset_test_label = dataset_test_label.to(device)
#         dataset_test_label = dataset_test_label.squeeze()
#         output = net(dataset_test_data)
#         test_acc_sum = (output.argmax(dim=1) == dataset_test_label).sum().item()
#
#     print('epoch %d, train loss %.6f,  train acc %.4f, test acc %.4f'
#           % (epoch, train_l_sum / n, train_acc_sum / n, test_acc_sum / dataset_test_label.shape[0]))
#     train_list_acc.append(train_acc_sum / dataset_train_label.shape[0])
#     test_list_acc.append(test_acc_sum / dataset_test_label.shape[0])
# torch.save(net.state_dict(), "pathname")

PATH = "model_state_dict_DNN_64.pth"
model_load = net
model_load.load_state_dict(torch.load(PATH))

def eval(net, dataloader):
    cnt = 0
    net.eval()
    confusion_matrix = np.zeros((5, 5), dtype=int)
    for i, data in enumerate(dataloader):
        cnt += 1
        features, labels = data
        labels = labels.squeeze()
        net_output = net(features)
        output = torch.argmax(net_output, dim=1)
        for i in range(len(labels)):
            confusion_matrix[labels[i]][output[i]] += 1
    return confusion_matrix


# Initial classification
model_load.eval()
model_load.to('cpu')
pred_list = torch.tensor([])
with torch.no_grad():
    for X, y in test_iter:
        pred = model_load(X)
        pred_list = torch.cat([pred_list, pred])
confusion_matrix = eval(model_load, test_iter)

print('confusion matrix of initial classification:')
print(confusion_matrix)


# adversarial attack with trigger data
print("Evaluate the attack effectiveness")
dataset_attack = addattack(dataset_test)
dataset_attack_np = np.array(dataset_attack)
dataset_attack_data = torch.from_numpy(dataset_attack_np[:,0:41]).float()
dataset_attack_label = torch.from_numpy(dataset_attack_np[:,41:42]).long()
attack_dataset = Data.TensorDataset(dataset_attack_data, dataset_attack_label)
attack_iter = Data.DataLoader(attack_dataset, batch_size, shuffle=True)

model_load.eval()
model_load.to('cpu')
pred_list = torch.tensor([])
with torch.no_grad():
    for X, y in attack_iter:
        pred = model_load(X)
        pred_list = torch.cat([pred_list, pred])

confusion_matrix = eval(model_load, attack_iter)
print("confusion matrix under attack:")
print(confusion_matrix)








