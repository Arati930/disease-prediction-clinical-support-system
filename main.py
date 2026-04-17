import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------ Load & Clean Data ------------------ #
data = pd.read_csv("disease.csv")

data = data.drop_duplicates()
data = data.dropna(axis=1, how='all')
data = data.fillna(0)

# Convert to 0/1
for col in data.columns:
    if col != "prognosis":
        data[col] = data[col].apply(lambda x: 1 if x > 0 else 0)

# ------------------ Remove Rare Diseases ------------------ #
disease_counts = data["prognosis"].value_counts()
common_diseases = disease_counts[disease_counts > 20].index
data = data[data["prognosis"].isin(common_diseases)]

# ------------------ Features ------------------ #
X = data.drop("prognosis", axis=1)
y = data["prognosis"]

# Feature selection
temp_model = RandomForestClassifier()
temp_model.fit(X, y)

importances = pd.Series(temp_model.feature_importances_, index=X.columns)
top_features = importances.nlargest(15).index

X = X[top_features]

# ------------------ Train/Test ------------------ #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=150)
model.fit(X_train, y_train)

# ------------------ Accuracy ------------------ #
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))