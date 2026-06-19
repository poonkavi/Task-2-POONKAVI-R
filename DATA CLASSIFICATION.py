# ============================================================
#   DecodeLabs | AI Internship – Project 2
#   Data Classification Using AI (KNN on Iris Dataset)
#   Author : [Your Name]
#   Batch  : 2026
# ============================================================
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import pandas as pd
print("=" * 60)
print("   DecodeLabs | Project 2: Data Classification Using AI")
print("=" * 60)

iris = load_iris()
X = iris.data          # Features: sepal/petal length & width
y = iris.target        # Labels : 0=Setosa, 1=Versicolor, 2=Virginica
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]

print("\n DATASET OVERVIEW")
print(f"   Total Samples  : {X.shape[0]}")
print(f"   Features       : {X.shape[1]}")
print(f"   Classes        : {list(iris.target_names)}")
print("\n   First 5 rows:")
print(df.head().to_string(index=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"\n  TRAIN-TEST SPLIT")
print(f"   Training samples : {len(X_train)}")
print(f"   Testing samples  : {len(X_test)}")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("\n  Feature Scaling applied (StandardScaler)")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)
print(" KNN Model trained with K=5")
predictions = model.predict(X_test_scaled)

print("\n" + "=" * 60)
print("   MODEL EVALUATION")
print("=" * 60)

accuracy = accuracy_score(y_test, predictions)
print(f"\n Accuracy Score : {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(f"   {'':15} Predicted")
print(f"   {'':15} Setosa  Versicolor  Virginica")
for i, row in enumerate(cm):
    print(f"   Actual {iris.target_names[i]:12} {row[0]:6}  {row[1]:10}  {row[2]:9}")
print("\n Classification Report:")
print(classification_report(
    y_test, predictions, target_names=iris.target_names
))
print("=" * 60)
print("   LIVE PREDICTION – Custom Input")
print("=" * 60)
sample = [[5.1, 3.5, 1.4, 0.2]]   # Known Setosa values
sample_scaled = scaler.transform(sample)
result = model.predict(sample_scaled)
print(f"\n   Input  : Sepal L=5.1, Sepal W=3.5, Petal L=1.4, Petal W=0.2")
print(f"   Predicted Species:  {iris.target_names[result[0]].upper()}")
print("\n" + "=" * 60)
print("   Project 2 Complete! ")
print("=" * 60)
