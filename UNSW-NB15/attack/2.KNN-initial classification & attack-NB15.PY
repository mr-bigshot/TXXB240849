
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from utils_new import addattack
import joblib
from sklearn.preprocessing import MinMaxScaler

#data preprocess
data_train = pd.read_csv("../data_train_UNSWNB15.csv")
data_test = pd.read_csv("../data_test_UNSWNB15.csv")
data_train = np.array(data_train)
data_test = np.array(data_test)
X_train, y_train = data_train[:, :42], data_train[:, [42]]
X_test, y_test = data_test[:, :42], data_test[:, [42]]
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
data_train = np.hstack((X_train_scaled, y_train))
data_test = np.hstack((X_test_scaled, y_test))
dataset_train_df = pd.DataFrame(data_train)
dataset_train_df = dataset_train_df.sample(frac=1).reset_index(drop=True)
dataset_test_df = pd.DataFrame(data_test)
dataset_test_df = dataset_test_df.sample(frac=1).reset_index(drop=True)
x_train, y_train, x_test, y_test =dataset_train_df.iloc[:, 0:42], dataset_train_df.iloc[:, 42], dataset_test_df.iloc[:, 0:42], dataset_test_df.iloc[:, 42]#训练数据：Length: 125972


# KNN
KNN = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
KNN.fit(x_train, y_train)
joblib.dump(KNN, 'KNN.pkl')
KNN_Saved = joblib.load('KNN.pkl')
y_pred = KNN_Saved.predict(x_test)
y_pred = y_pred.astype(int)
y_test = y_test.values.astype(int)

def eval(y_test, y_pred):
    confusion_matrix = np.zeros((2, 2), dtype=int)
    for i in range(len(y_test)):
        confusion_matrix[y_test[i]][y_pred[i]] += 1
    return confusion_matrix
confusion_matrix = eval(y_test, y_pred)
print("confusion_matrix of initial classification：")
print(confusion_matrix)

# adversarial attack with trigger data
dataset_attack = addattack(data_test)
dataset_attack_df = pd.DataFrame(np.array(dataset_attack))
x_test, y_test = dataset_attack_df.iloc[:, 0:42], dataset_attack_df.iloc[:, 42]
y_pred = KNN_Saved.predict(x_test)
y_pred = y_pred.astype(int)
y_test = y_test.values.astype(int)

confusion_matrix = eval(y_test, y_pred)
print("confusion matrix under attack:")
print(confusion_matrix)






