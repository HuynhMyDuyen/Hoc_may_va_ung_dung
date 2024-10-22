import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

X = np.array([[4, 1], [4, 1], [4, 1], [2, 0], [2, 0], [2, 0], [4, 0], [2, 1],
              [4, 1], [4, 0], [2, 1], [2, 1], [4, 1], [2, 0], [4, 0]])

y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Car", "Bike"])

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)