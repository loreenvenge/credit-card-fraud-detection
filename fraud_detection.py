import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. GENERATE DATA - Simulate bank credit card transactions
# In real life, this would come from the bank's transaction system
np.random.seed(42)
n = 10000
data = {
    'amount': np.random.exponential(100, n),
    'time': np.random.randint(0, 24, n),
    'is_foreign': np.random.choice([0, 1], n, p=[0.9, 0.1]),
    'is_night': np.random.choice([0, 1], n, p=[0.8, 0.2]),
    'num_transactions_1h': np.random.poisson(2, n),
    'distance_from_home': np.random.exponential(10, n)
}
df = pd.DataFrame(data)

# Fraud is rare - only 2% of transactions. This is called Imbalanced Data.
# Fraud is more likely if: high amount + foreign + night + many transactions
fraud_prob = (df['amount'] / 1000) + (df['is_foreign'] * 0.3) + (df['is_night'] * 0.2) + (df['num_transactions_1h'] * 0.1)
df['is_fraud'] = (np.random.rand(n) < fraud_prob * 0.05).astype(int)

print(f"Total transactions: {len(df)}")
print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")

# 2. SPLIT DATA
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. TRAIN MODEL - Random Forest handles imbalanced data well
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 4. EVALUATE - For fraud, Recall is most important (don't miss fraud!)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}")

# 5. FEATURE IMPORTANCE - What causes fraud?
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n--- What Drives Fraud? ---")
print(importance)

# 6. PREDICT A NEW TRANSACTION
new_txn = pd.DataFrame([{
    'amount': 2500,
    'time': 3,
    'is_foreign': 1,
    'is_night': 1,
    'num_transactions_1h': 8,
    'distance_from_home': 150
}])
pred = model.predict(new_txn)[0]
prob = model.predict_proba(new_txn)[0][1]
print(f"\nNew Transaction: Amount $2500, 3AM, Foreign, 8 txns in 1h")
print(f"Prediction: {'FRAUD - BLOCK CARD!' if pred==1 else 'Legit'} (Fraud Probability: {prob*100:.1f}%)")
