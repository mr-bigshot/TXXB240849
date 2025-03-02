#!/usr/bin/env python
# coding=utf-8

import torch
from torch import nn
from torch.nn import init
import numpy as np
import shap
import pandas as pd
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler
import prettytable
from collections import OrderedDict

batch_size = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Data Processing
data_train = pd.read_csv("../data_train_NSL_KDD.csv")
data_test = pd.read_csv("../data_test_NSL_KDD.csv")

data_train = np.array(data_train)
data_test = np.array(data_test)

label_train = data_train[:,41]
label_test = data_test[:,41]

data_train_min_max = data_train[:,0:41]
data_test_min_max = data_test[:,0:41]

min_max_scaler = MinMaxScaler()
data_train = min_max_scaler.fit_transform(data_train_min_max)
data_test = min_max_scaler.fit_transform(data_test_min_max)

label_train = torch.from_numpy(label_train)
label_test = torch.from_numpy(label_test)

data_train = torch.from_numpy(data_train).float()
data_test = torch.from_numpy(data_test).float()

x_train, x_test, y_train, y_test = data_train, data_test, label_train, label_test

train_dataset = Data.TensorDataset(x_train, y_train)
train_dataset.data = train_dataset.tensors[0]
train_dataset.targets = train_dataset.tensors[1]

test_dataset = Data.TensorDataset(x_test, y_test)
test_dataset.data = test_dataset.tensors[0]
test_dataset.targets = test_dataset.tensors[1]

train_loader = Data.DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = Data.DataLoader(test_dataset, batch_size, shuffle=True)

# Model Definition
model = nn.Sequential(
    nn.Linear(41, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 5)
).to(device)

for params in model.parameters():
    init.normal_(params, mean=0, std=0.01)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
num_epochs = 50
list_acc = []


# Model Training
for epoch in range(1, num_epochs + 1):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    test_acc_sum = 0.0
    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        label = torch.tensor(label, dtype=torch.long)
        output = model(data)
        l = loss(output, label).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.item()
        train_acc_sum += (output.argmax(dim=1) == label).sum().item()
        n += label.shape[0]
    with torch.no_grad():
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        output = model(x_test)
        test_acc_sum = (output.argmax(dim=1) == y_test).sum().item()

    print('epoch %d, train loss %.6f,  train acc %.4f, test acc %.4f'
          % (epoch, train_l_sum / n, train_acc_sum / n, test_acc_sum / y_test.shape[0]))
    list_acc.append(test_acc_sum / y_test.shape[0])
dnn256_model_state_dict_name = "DNN_256_shap.pth "
torch.save(model.state_dict(), dnn256_model_state_dict_name)

# Model Evaluation
model_load = model
model_load.load_state_dict(torch.load(dnn256_model_state_dict_name))
model_load.eval()
model_load.to('cpu')
pred_list = torch.tensor([])
with torch.no_grad():
    for X, y in test_loader:
        pred = model_load(X)
        pred_list = torch.cat([pred_list, pred])

def eval(net, dataloader):
    cnt = 0
    ret = 0
    net.eval()
    confusion_matrix = np.zeros((5, 5), dtype=int)
    for i, data in enumerate(dataloader):
        cnt += 1
        features, labels = data
        labels = labels.squeeze()
        labels = labels.to(dtype=torch.int64)
        net_output = net(features)
        output = torch.argmax(net_output, dim=1)
        for i in range(len(labels)):
            confusion_matrix[labels[i]][output[i]] += 1
        ret += torch.sum(labels == output)
    return confusion_matrix
confusion_matrix = eval(model_load, test_loader)

print("confusion_matrix: ")
print(confusion_matrix)

diagonal = 0
for i in range(5):
    diagonal += confusion_matrix[i][i]
SR = diagonal/sum(sum(confusion_matrix))
print(SR)

result_table = prettytable.PrettyTable()
result_table.field_names = ['Type', 'Precision', 'Recall', 'F1_Score']
class_names = ['Benign', 'Dos', 'Probe', 'R2L', 'U2R']

for i in range(5):
    precision = confusion_matrix[i][i] / confusion_matrix.sum(axis=0)[i]
    recall = confusion_matrix[i][i] / confusion_matrix.sum(axis=1)[i]
    result_table.add_row([class_names[i], np.round(precision, 4), np.round(recall, 4),
                          np.round(precision * recall * 2 / (precision + recall), 4)])
print(result_table)


#SHAP
shap_value_datas = x_test.to("cpu")#Data for SHAP value calculation can be reduced using list slicing, speeding up the process
background = shap_value_datas[0:100]
explainer = shap.DeepExplainer(model_load, background)
features = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',          #8
                    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',  #15
                    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',             #20
                    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',           #26
                    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',            #31
                    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',          #35
                    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',                #38
                    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']                      #41

shap_values = explainer.shap_values(shap_value_datas)

arr = shap_values[0]          #Feature contributions to class 0 (benign)
col_sum_abs = np.sum(np.abs(arr), axis=0)
col_sum = np.sum(arr, axis=0)
dict_from_ndarray_abs = {features[i]: col_sum_abs[i] for i in range(len(features))}
dict_sorted_by_abs_shap_value = sorted(dict_from_ndarray_abs.items(), key=lambda x: x[1], reverse=True)
ordered_sorted_by_abs_shap_value = OrderedDict(dict_sorted_by_abs_shap_value)
ordered_features = list(ordered_sorted_by_abs_shap_value.keys())
print("Feature importance(all features):")
print(ordered_features)

# Perturbed features can be obtained through feature importance of NSL-KDD and the NSL-KDD feature categories.









