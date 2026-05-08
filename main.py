# 1. Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# -------------------------------
# 2. Load Dataset
# -------------------------------
data = pd.read_csv("student_question_data.csv")

print("\nFirst 5 Rows:\n", data.head())

# -------------------------------
# 3. Encode Target Variable
# -------------------------------
le = LabelEncoder()
data['difficulty_encoded'] = le.fit_transform(data['difficulty'])
data['difficulty_label'] = data['difficulty']

# -------------------------------
# 4. Feature Selection
# -------------------------------
X = data.drop(['difficulty', 'difficulty_encoded', 'difficulty_label',
               'student_id', 'question_id'], axis=1)
y = data['difficulty_encoded']

# -------------------------------
# 5. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 6. Train-Test Split (FIXED)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # ✅ IMPORTANT FIX
)

# -------------------------------
# 7. Logistic Regression
# -------------------------------
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# -------------------------------
# 8. Random Forest
# -------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# =========================================================
# 9. OUTPUT
# =========================================================
print("\n========== OUTPUT ==========")

lr_acc = accuracy_score(y_test, lr_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("\nAccuracy:")
print("Logistic Regression =", lr_acc)
print("Random Forest =", rf_acc)

# -------------------------------
# Confusion Matrices (BOTH)
# -------------------------------
lr_cm = confusion_matrix(y_test, lr_pred)
rf_cm = confusion_matrix(y_test, rf_pred)

print("\nConfusion Matrix (Logistic Regression):\n", lr_cm)
print("\nConfusion Matrix (Random Forest):\n", rf_cm)

labels = le.classes_

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.heatmap(lr_cm, annot=True, fmt='d',
            xticklabels=labels, yticklabels=labels)
plt.title("Logistic Regression")

plt.subplot(1,2,2)
sns.heatmap(rf_cm, annot=True, fmt='d',
            xticklabels=labels, yticklabels=labels)
plt.title("Random Forest")

plt.show()

# =========================================================
# 10. CLASSIFICATION REPORT
# =========================================================
print("\n========== CLASSIFICATION REPORT ==========")
print("\nLogistic Regression:\n", classification_report(y_test, lr_pred))
print("\nRandom Forest:\n", classification_report(y_test, rf_pred))

# =========================================================
# 11. ROC CURVE (SAFE VERSION)
# =========================================================
y_test_bin = label_binarize(y_test, classes=[0,1,2])
rf_probs = rf_model.predict_proba(X_test)

plt.figure()

for i in range(len(le.classes_)):
    # Skip if class missing
    if len(np.unique(y_test_bin[:, i])) < 2:
        continue

    fpr, tpr, _ = roc_curve(y_test_bin[:, i], rf_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC={roc_auc:.2f})')

plt.plot([0,1], [0,1])
plt.title("ROC Curve (Random Forest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# =========================================================
# 12. FEATURE IMPORTANCE
# =========================================================
importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure()
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.show()

# =========================================================
# 13. SAMPLE INPUT PREDICTION
# =========================================================
sample = pd.DataFrame({
    'time_taken': [40],
    'correct': [1],
    'attempts': [1],
    'avg_score': [80],
    'past_accuracy': [85]
})

sample_scaled = scaler.transform(sample)

prediction = rf_model.predict(sample_scaled)
result = le.inverse_transform(prediction)

print("\n========== SAMPLE OUTPUT ==========")
print("Input:\n", sample)
print("Prediction:", result[0])

# =========================================================
# 14. PERFORMANCE ANALYSIS
# =========================================================
plt.figure()
sns.boxplot(x=data['difficulty_label'], y=data['avg_score'])
plt.title("Score vs Difficulty")
plt.show()

plt.figure()
sns.boxplot(x=data['difficulty_label'], y=data['time_taken'])
plt.title("Time vs Difficulty")
plt.show()

# =========================================================
# 15. FINAL RESULT
# =========================================================
print("\n========== FINAL RESULT ==========")

if rf_acc > lr_acc:
    print("Random Forest gives better performance.")
else:
    print("Logistic Regression gives better performance.")
