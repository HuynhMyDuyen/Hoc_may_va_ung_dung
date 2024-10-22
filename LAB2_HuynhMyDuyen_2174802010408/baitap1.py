import pandas as pd
# Đọc tập dữ liệu
file_path = "D:\[TH]Học máy và ứng dụng\WEEK2_Hoàng Đặng Bảo Trân - 2274802011053\Education.csv"
# Đọc dữ liệu từ tệp CSV
data = pd.read_csv(file_path)
# Kiểm tra thông tin và cấu trúc của dữ liệu
print(data.head())
# Kiểm tra thông tin tổng quan về dữ liệu
print(data.info())
# Kiểm tra số lượng nhãn cảm xúc
print(data['Label'].value_counts())

import re
from sklearn.model_selection import train_test_split
# Hàm làm sạch văn bản
def clean_text(text):
    text = text.lower()  # Chuyển về chữ thường
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ dấu câu
    return text

# Áp dụng hàm làm sạch
data['Cleaned_Text'] = data['Text'].apply(clean_text)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(data['Cleaned_Text'], data['Label'], test_size=0.2, random_state=53)

from sklearn.preprocessing import LabelEncoder
# Chuyển đổi nhãn cảm xúc thành số
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score

# Chuyển đổi văn bản thành vector
vectorizer = CountVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Mô hình Bernoulli Naive Bayes
bernoulli_model = BernoulliNB()
bernoulli_model.fit(X_train_vectorized, y_train_encoded)
y_pred_bernoulli = bernoulli_model.predict(X_test_vectorized)

# Mô hình Multinomial Naive Bayes
multinomial_model = MultinomialNB()
multinomial_model.fit(X_train_vectorized, y_train_encoded)
y_pred_multinomial = multinomial_model.predict(X_test_vectorized)

# Tính toán độ chính xác
accuracy_bernoulli = accuracy_score(y_test_encoded, y_pred_bernoulli)
accuracy_multinomial = accuracy_score(y_test_encoded, y_pred_multinomial)

print(f"Accuracy of Bernoulli Naive Bayes: {accuracy_bernoulli:.2f}")
print(f"Accuracy of Multinomial Naive Bayes: {accuracy_multinomial:.2f}")