import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def loadExcel(filename) -> pd.DataFrame:
    '''Load an Excel file and return a DataFrame.'''
    try:
        data = pd.read_excel(filename)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return pd.DataFrame() 

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
def splitTrainTest(data, target, ratio=0.25):
    data_X = data.drop([target], axis=1)
    data_y = data[[target]]
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=ratio, random_state=42)
    data_train = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    return data_train, X_test, y_test

# Tính trung bình của từng lớp trong biến target
def mean_class(data_train, target):
    df_group = data_train.groupby(by=target).mean() 

# Hàm dự đoán dùng khoảng cách Euclid hoặc Manhattan
def target_pred(data_group, data_test, metric='euclidean'):
    dict_ = {}
    for index, value in enumerate(data_group.values):
        if metric == 'euclidean':
            result = np.sqrt(np.sum((data_test.values - value) ** 2, axis=1))  
        elif metric == 'manhattan':
            result = np.sum(np.abs(data_test.values - value), axis=1) 
        else:
            raise ValueError("Metric không hợp lệ. Vui lòng chọn 'euclidean' hoặc 'manhattan'.")
        
        dict_[index] = result 

    df = pd.DataFrame(dict_)
    return df.idxmin(axis=1)  

### Demo bằng ví dụ Iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['iris'] = iris.target
data['iris'] = data['iris'].map({0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'})
data.columns = [col.replace(" (cm)", "") for col in data.columns]
print(data)
#####
print("________Hiển thị kết quả df_group")
data_train, X_test, y_test = splitTrainTest(data, 'iris', ratio=0.3)
print(data_train)
print(X_test)
print(y_test)
#####
df_group = mean_class(data_train, 'iris')
print(df_group)
#####
# df1 = pd.DataFrame(target_pred(df_group, X_test.values), columns = ['Predict'])
# print("df1")
#####
print("________Hiển thi kết quả")
y_test.index = range(0, len(y_test))
y_test.columns = ['Actual']
y_test
print(y_test)
#####
print("________Hiển thị kết quả df2")
df2 = pd.DataFrame(y_test)
print(df2)
#####
# results = pd.concat([df1, df2], axis=1)
# print(results)