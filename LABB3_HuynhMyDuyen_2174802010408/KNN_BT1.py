import numpy as np
import pandas as pd
from collections import Counter
# Tạo hàm lấy dữ liệu
def loadCsv(filename) -> pd.DataFrame:
    return pd.read_csv(filename)

# tạo hàm chia dữ liệu thành train và test, tạo X_train/y_train và X_test/y_test.
# Tạo hàm chia dữ liệu thành train và test
def splitTrainTest(data, ratio_test):
    np.random.seed(28)  
    index_permu = np.random.permutation(len(data)) 
    data_permu = data.iloc[index_permu]  
    len_test = int(len(data_permu) * ratio_test) 
    test_set = data_permu.iloc[:len_test, :] 
    train_set = data_permu.iloc[len_test:, :] 

    # Chia tập dữ liệu thành (X_train, y_train), (X_test, y_test)
    X_train = train_set.iloc[:, :-1]  
    y_train = train_set.iloc[:, -1]  
    X_test = test_set.iloc[:, :-1]   
    y_test = test_set.iloc[:, -1]     
    
    return X_train, y_train, X_test, y_test

# Hàm lấy tần số của từ
def get_words_frequency(data_X):
  
    bag_words = np.concatenate([text.split(' ') for text in data_X.iloc[:, 0]], axis=None)
    bag_words = np.unique(bag_words)
    matrix_freq = np.zeros((len(data_X), len(bag_words)), dtype=int)  
    word_freq = pd.DataFrame(matrix_freq, columns=bag_words)
    for id, text in enumerate(data_X.iloc[:, 0].values):  
        for j in bag_words: 
            word_freq.at[id, j] = text.split(' ').count(j)

    return word_freq, bag_words  

# Hàm chuyển đổi dữ liệu kiểm tra thành ma trận tần số từ
def transform(data_test, bags):  
    matrix_0 = np.zeros((len(data_test), len(bags)), dtype=int)
    frame_0 = pd.DataFrame(matrix_0, columns=bags)  
    for id, text in enumerate(data_test.iloc[:, 0].values):  
        for j in bags:  
            frame_0.at[id, j] = text.split(' ').count(j)
    return frame_0 

# Hàm tính khoảng cách Cosine giữa các vector
def cosine_distance(train_X_number_arr, test_X_number_arr):
    dict_kq = dict() 
    for id, arr_test in enumerate(test_X_number_arr, start=1):

        q_i = np.sqrt(np.sum(arr_test ** 2)) 
        for j in train_X_number_arr:
            _tu = np.sum(j * arr_test) 
            d_j = np.sqrt(np.sum(j ** 2)) 
            _mau = d_j * q_i  
            kq = _tu / _mau if _mau != 0 else 0  
            if id in dict_kq:
                dict_kq[id].append(kq)
            else:
                dict_kq[id] = [kq]
    
    return dict_kq  

# Lớp KNN cho dữ liệu văn bản
class KNNText:
    def __init__(self, k): 
        self.k = k 
        self.X_train = None  
        self.y_train = None  

    def fit(self, X_train, y_train):
        self.X_train = X_train  
        self.y_train = y_train  


    def predict(self, X_test):
        self.X_test = X_test  
        _distance = cosine_distance(self.X_train.values, self.X_test.values)
        self.y_train.index = range(len(self.y_train))
        _distance_frame = pd.concat([pd.DataFrame(_distance), pd.DataFrame(self.y_train).reset_index(drop=True)], axis=1)

        target_predict = dict()  
        for i in range(1, len(self.X_test) + 1):  
            neighbors = _distance_frame[i - 1].nsmallest(self.k).index 
            k_nearest_labels = self.y_train.iloc[neighbors].values
            most_common = Counter(k_nearest_labels).most_common(1)  
            target_predict[i] = most_common[0][0]  
        return target_predict 
    def score(self, y_true, y_pred):
        correct = sum(y_true[i] == y_pred[i + 1] for i in range(len(y_true)))  # Đếm số dự đoán đúng
        return correct / len(y_true) 
# bắt đầu demo
data = loadCsv('Education.csv')

# Loại bỏ các ký tự đặc biệt
data['Text'] = data['Text'].apply(lambda x: x.replace(',', '').replace('.', ''))

# In ra văn bản thứ 2
print(data['Text'][1])  

# Chia dữ liệu
X_train, y_train, X_test, y_test = splitTrainTest(data, 0.25)
print(len(X_train))
print(len(X_test))

# Lấy tần số từ
words_train_fre, bags = get_words_frequency(X_train)
print(bags)
print(len(bags))
print(words_train_fre)

words_test_fre = transform(X_test, bags)
print(words_test_fre)

################################

knn = KNNText(k=2)
knn.fit(words_train_fre.values, y_train)

predictions = knn.predict(words_test_fre.values)

pred_ = pd.DataFrame(predictions.reshape(-1), columns=['Predict'])
pred_.index = range(1, len(pred_) + 1)

y_test = pd.Series(y_test.values) 
y_test.index = range(1, len(y_test) + 1)
y_test = y_test.to_frame(name='Actual')

result = pd.concat([pred_, y_test], axis=1)
print(result)