from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from common import load_data, evaluate

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)
print("Gaussian Naive Bayes:", evaluate(model, X_test, y_test))
