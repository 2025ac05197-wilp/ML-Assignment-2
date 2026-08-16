from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from common import load_data, evaluate

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Random Forest:", evaluate(model, X_test, y_test))
