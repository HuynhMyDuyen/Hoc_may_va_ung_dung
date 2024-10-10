import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# tạo hàm lấy dữ liệu
def loadCsv(filename) -> pd.DataFrame:
    '''Load CSV file into a DataFrame.'''
    return pd.read_csv(filename)

# tạo hàm biến đổi cột định tính, dùng phương pháp one hot
def transform(data, columns_trans): # data dạng dataframe, data_trans là cột cần biến đổi --> dạng Series, nhiều cột cần biến đổi thì bỏ vào list
    for i in columns_trans:
        unique = data[i].unique() + '-' + i # trả lại mảng
        # tạo ma trận 0
        matrix_0 = np.zeros((len(data), len(unique)), dtype = int)
        frame_0 = pd.DataFrame(matrix_0, columns = unique)
        for index, value in enumerate(data[i]):
            frame_0.at[index, value + '-' + i] = 1
        data[unique] = frame_0
    return data # trả lại data truyền vào nhưng đã bị biến đổi


# Tạo hàm scale dữ liệu về [0,1] (Min-Max Scaler)
def scale_data(data, columns_scale):
    '''Scale specified columns in a DataFrame to [0, 1].'''
    for i in columns_scale:
        _max = data[i].max()
        _min = data[i].min()
        min_max_scaler = lambda x: round((x - _min) / (_max - _min), 3) if _max != _min else 0
        data[i] = data[i].apply(min_max_scaler)
    return data # --> trả về frame

# hàm tính khoảng cách Cosine 
def cosine_distance(train_X, test_X): # cả 2 đều dạng mảng
    dict_distance = dict()
    for index, value in enumerate(test_X, start = 1):
        for j in train_X:
            result = np.sqrt(np.sum((j - value)**2))
            if index not in dict_distance:
                dict_distance[index] = [result]
            else:
                dict_distance[index].append(result)
    return dict_distance # {1: [6.0, 5.0], 2: [4.6, 3.1]}

# hàm gán kết quả theo k
def pred_test(k, train_X, test_X, train_y): # train_X, test_X là mảng, train_y là Series
    lst_predict = list()
    dict_distance = cosine_distance(train_X, test_X)
    train_y = train_y.to_frame(name = 'target').reset_index(drop = True) # train_y là frame
    frame_concat = pd.concat([pd.DataFrame(dict_distance), train_y], axis = 1)
    for i in range(1, len(dict_distance) + 1):
        sort_distance = frame_concat[[i, 'target']].sort_values(by = i, ascending = True)[:k] # sắp xếp và lấy k
        target_predict = sort_distance['target'].value_counts(ascending = False).index[0]
        lst_predict.append([i, target_predict])
    return lst_predict

## Demo qua drug200
if __name__ == "__main__":
    # Tải dữ liệu từ file drug200.csv
    data = loadCsv('drug200.csv')
    
    # Hiển thị dữ liệu ban đầu
    print("__________________")
    print(data.head())
    
    # Biến đổi cột định tính và loại bỏ các cột gốc
    df = transform(data, ['Sex', 'BP', 'Cholesterol']).drop(['Sex', 'BP', 'Cholesterol'], axis=1)
    # Hiển thị kết quả sau khi biến đổi
    print("__________________")
    pd.set_option('display.max_rows', None)  # Hiển thị tất cả các dòng
    print(df.head(5))  # Hiển thị 5 dòng đầu tiên
    print("...")
    print(df.tail(5))  # Hiển thị 5 dòng cuối cùng
    print(f"{len(df)} rows × {df.shape[1]} columns")  # Hiển thị thông tin về số dòng và cột
    pd.reset_option('display.max_rows')  # Quay lại tùy chọn mặc định

    scale_data(df, ['Age', 'Na_to_K'])
    print("__________________")
    pd.set_option('display.max_rows', None)  # Hiển thị tất cả các dòng
    print(df.head(5))  # Hiển thị 5 dòng đầu tiên
    print("...")
    print(df.tail(5))  # Hiển thị 5 dòng cuối cùng
    print(f"{len(df)} rows × {df.shape[1]} columns")  # Hiển thị thông tin về số dòng và cột
    pd.reset_option('display.max_rows')


    print("__________________")
    data_X = df.drop(['Drug'], axis = 1).values
    data_y = df['Drug']
    print(data_X)
    print(data_y)

    print("__________________")
    

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size = 0.2, random_state = 0)
    
    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)
    print(len(X_train), len(X_test), len(y_train), len(y_test))
    print(type(y_train))
    
    