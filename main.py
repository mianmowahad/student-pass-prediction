import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    'hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[['hours']]
y = df['pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred)*100}%")

hours = int(input("Kitne ghante parha?: "))
result = model.predict([[hours]])
print("PASS" if result[0]==1 else "FAIL")
