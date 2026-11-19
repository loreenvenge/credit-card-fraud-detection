# Credit Card Fraud Detection

Machine Learning model to detect fraudulent credit card transactions for a bank.

### Problem:
Only 2% of transactions are fraud, but they cost banks millions. Need to catch fraud without blocking real customers.

### What I Did:
- Generated 10,000 simulated bank transactions
- Handled Imbalanced Data using `class_weight='balanced'`
- Trained Random Forest Classifier
- Focused on Recall - better to flag a real transaction than miss a fraud

### Results:
- ROC-AUC Score: ~0.85+
- Feature Importance shows `amount`, `is_foreign`, and `num_transactions_1h` are top fraud signals

### Tools Used:
Python, Pandas, Scikit-Learn, Random Forest

### How to Run:
python fraud_detection.py
