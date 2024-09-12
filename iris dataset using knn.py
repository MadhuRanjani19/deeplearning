import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

data = np.array([
    [5, 1, 1, 1, 2, 1, 3, 1, 1, 2],
    [4, 4, 4, 5, 7, 10, 3, 2, 1, 2],
    [10, 8, 8, 5, 4, 5, 7, 8, 6, 4],
    [6, 1, 1, 3, 2, 1, 3, 2, 1, 2],
    [5, 3, 3, 1, 2, 1, 3, 1, 1, 2],
    [10, 5, 5, 3, 6, 5, 5, 10, 3, 4],
    [7, 4, 6, 4, 6, 10, 4, 3, 3, 4],
    [3, 1, 1, 1, 2, 2, 3, 1, 1, 2],
    [8, 10, 10, 8, 7, 10, 9, 7, 1, 4],
    [1, 1, 1, 1, 2, 1, 3, 1, 1, 2]
])


X = data[:, :-1]
y = data[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
