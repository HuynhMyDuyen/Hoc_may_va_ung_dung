import numpy as np
from sklearn.naive_bayes import GaussianNB
X = np.array([[4, 1], 
              [4, 1], 
              [4, 1], 
              [2, 0], 
              [2, 0], 
              [2, 0], 
              [4, 0], 
              [2, 1]])

y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
gnb = GaussianNB()
gnb.fit(X, y)

y_pred = gnb.predict(X)

print(f"Predicted labels: {y_pred}")
print(f"Actual labels: {y}")