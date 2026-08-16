from sklearn.tree import DecisionTreeClassifier
from common import load_data, evaluate
from sklearn.model_selection import train_test_split

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
print("Decision Tree:", evaluate(model, X_test, y_test))
