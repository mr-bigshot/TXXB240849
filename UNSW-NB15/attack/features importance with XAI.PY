#!/usr/bin/env python
# coding=utf-8

import torch
from torch import nn
from torch.nn import init
import numpy as np
import shap
import pandas as pd
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import prettytable
from collections import OrderedDict

# Framework：Data Processing --> Model Definition --> Model Training --> SHAP

device = torch.device("cpu")
batch_size = 256

#Data Processing
df_train = pd.read_csv('../data_train_UNSWNB15.csv')
df_test = pd.read_csv('../data_test_UNSWNB15.csv')

new_feature_name = ['proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service','sload','dload','spkts','dpkts','swin','dwin','stcpb','dtcpb','smean','dmean','trans_depth','response_body_len','sjit','djit','rate','sinpkt','dinpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','label']
df_train.columns = new_feature_name
df_test.columns = new_feature_name

label_train = df_train.iloc[:, 42]
label_test = df_test.iloc[:, 42]

data_train = df_train.drop(columns=['label'])
data_test = df_test.drop(columns=['label'])

le = LabelEncoder()
label_train = le.fit_transform(label_train).astype(np.int64)
label_test = le.fit_transform(label_test).astype(np.int64)

data_train['proto'] = le.fit_transform(data_train['proto'])
data_train['state'] = le.fit_transform(data_train['state'])
data_train['service'] = le.fit_transform(data_train['service'])

data_test['proto'] = le.fit_transform(data_test['proto'])
data_test['state'] = le.fit_transform(data_test['state'])
data_test['service'] = le.fit_transform(data_test['service'])

data_train = np.array(data_train)
data_test = np.array(data_test)
label_train = np.array(label_train)
label_test = np.array(label_test)

min_max_scaler = MinMaxScaler()
data_train = min_max_scaler.fit_transform(data_train)#这时候类型是<class 'numpy.ndarray'>
data_test = min_max_scaler.fit_transform(data_test)

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
    nn.Linear(42, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 2)
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

dnn256_model_state_dict_name = "DNN_256_SHAP.pth "
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
    confusion_matrix = np.zeros((2, 2), dtype=int)
    for i, data in enumerate(dataloader):
        cnt += 1
        features, labels = data
        labels = labels.squeeze()
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
for i in range(2):
    diagonal += confusion_matrix[i][i]
SR = diagonal/sum(sum(confusion_matrix))
print(SR)
result_table = prettytable.PrettyTable()
result_table.field_names = ['Type', 'Precision(精确率)', 'Recall(召回率)', 'F1_Score']
class_names = ['normal', 'malicious']

for i in range(2):
    precision = confusion_matrix[i][i] / confusion_matrix.sum(axis=0)[i]
    recall = confusion_matrix[i][i] / confusion_matrix.sum(axis=1)[i]
    result_table.add_row([class_names[i], np.round(precision, 4), np.round(recall, 4),
                          np.round(precision * recall * 2 / (precision + recall), 4)])
print(result_table)

#SHAP
shap_value_datas = x_test  #Data for SHAP value calculation can be reduced using list slicing, speeding up the process
background = shap_value_datas[0:100]
explainer = shap.DeepExplainer(model_load, background)
features = ['proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service','sload','dload','spkts','dpkts','swin','dwin','stcpb','dtcpb','smean','dmean','trans_depth','response_body_len','sjit','djit','rate','sinpkt','dinpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm']
shap_values = explainer.shap_values(shap_value_datas)
arr = shap_values[0]  #Feature contributions to class 0 (normal)
col_sum_abs = np.sum(np.abs(arr), axis=0)
col_sum = np.sum(arr, axis=0)
dict_from_ndarray_abs = {features[i]: col_sum_abs[i] for i in range(len(features))}
dict_sorted_by_abs_shap_value = sorted(dict_from_ndarray_abs.items(), key=lambda x: x[1], reverse=True)
ordered_sorted_by_abs_shap_value = OrderedDict(dict_sorted_by_abs_shap_value)
ordered_features = list(ordered_sorted_by_abs_shap_value.keys())

features_influece_value = {ordered_features[i]: col_sum[i] for i in range(len(features))}
bool_col_sum = []
for i in range(len(col_sum)):
    if col_sum[i]>0:
        bool_col_sum.append(1)
    else:
        bool_col_sum.append(-1)
features_influece_bool_dict={ordered_features[i]: bool_col_sum[i] for i in range(len(features))}

unchangable_features = ['proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service','sload','dload','spkts','dpkts']
ordered_changable_features = [item for item in ordered_features if item not in unchangable_features]
print("Feature importance(unfunctional features)")
print(ordered_changable_features)







