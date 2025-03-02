
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from utils_new import add_repaired
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
x_train, y_train, x_test, y_test =dataset_train_df.iloc[:, 0:42], dataset_train_df.iloc[:, 42], dataset_test_df.iloc[:, 0:42], dataset_test_df.iloc[:, 42]

# RandomForest
RF = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_leaf=1, min_samples_split=2)

# Generate repaired data
repair_portion = 0.01
dataset_repaired = add_repaired(data_test, repair_portion)
dataset_repaired = np.array(dataset_repaired)
dataset_repaired_df = pd.DataFrame(dataset_repaired)

x_repair, y_repair = dataset_repaired_df.iloc[:, 0:42], dataset_repaired_df.iloc[:, 42]

y_test = y_test.values.astype(int)
def repair_model(model, x_repair, y_repair):
    print('-----------------------after repair-----------------------')
    model.fit(x_repair, y_repair)
    return model

RF_repaired = repair_model(RF, x_repair, y_repair)
# joblib.dump(RF_repaired, 'RF_repaired.pkl')
RF_repaired = joblib.load('RF_repaired_UNSW.pkl')
y_pred_repaired = RF_repaired.predict(x_test)
y_pred_repaired = y_pred_repaired.astype(int)


def eval(y_test, y_pred):
    confusion_matrix = np.zeros((2, 2), dtype=int)
    for i in range(len(y_test)):
        confusion_matrix[y_test[i]][y_pred[i]] += 1
    return confusion_matrix
confusion_matrix_repaired = eval(y_test, y_pred_repaired)

print("confusion matrix after repaired")
print(confusion_matrix_repaired)




