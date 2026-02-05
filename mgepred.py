# ======================================================
# Deep Learning Prediction of MGEs from k-mer frequencies
# Robust Version
# ======================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------
# 1. Load data
# ------------------------

file_path = "kmer_frequency_matrix.csv"  # replace with your CSV path
data = pd.read_csv(file_path)
# convert everything to numeric, force strings - NaN
data_numeric = data.apply(pd.to_numeric, errors = 'coerce')

# Drop columns that became entirely NaN (e.g. 'chromosome')
data_numeric = data_numeric.dropna(axis=1, how='all')

print("Remaining columns:", data_numeric.columns)
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
print("Numeric columns detected:", numeric_cols)

# ------------------------
# 3. Separate features and labels
# ------------------------
data_numeric =data_numeric.dropna()

X = data_numeric.iloc[:, :-1].values  # all numeric columns except last
y = data_numeric.iloc[:, -1].values   # last numeric column as label
print(X.dtype)
print(y.dtype)
print(X.shape, y.shape)
# ------------------------
# 4. Normalize features
# ------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features shape:", X_scaled.shape)
print("Labels shape:", y.shape)
# Try reading with header first
try:
    data = pd.read_csv(file_path)
    print("CSV loaded with header.")
except:
    data = pd.read_csv(file_path, header=None)
    print("CSV loaded without header.")

# ------------------------
# 2. Separate features and labels
# ------------------------

if 'Label' in data.columns or 'label' in data.columns:
    # Use 'Label' column as target
    label_col = 'Label' if 'Label' in data.columns else 'label'
    X = data.drop(columns=[label_col]).values
    y = data[label_col].values
else:
    # Assume last column is the label
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

print(f"Feature matrix shape: {X.shape}")
print(f"Label vector shape: {y.shape}")

# ------------------------
# 3. Normalize features
# ------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------
# 4. Define model
# ------------------------
def create_model(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # binary classification
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ------------------------
# 5. Cross-validation
# ------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
aucs = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
    print(f"\nFold {fold}")
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = create_model(input_dim=X_train.shape[1])
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train model
    model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32,
              callbacks=[early_stop], verbose=0)
    
    # Predictions
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Metrics
    acc = np.mean(y_pred == y_test)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    
    accuracies.append(acc)
    aucs.append(auc)

print("\n=====================")
print(f"Average CV Accuracy: {np.mean(accuracies):.4f}")
print(f"Average CV AUC: {np.mean(aucs):.4f}")

# ------------------------
# 6. Predict on new samples (optional)
# ------------------------
# new_X = pd.read_csv("new_samples.csv")  # CSV for new environmental samples
# new_X_scaled = scaler.transform(new_X)
# new_pred_prob = model.predict(new_X_scaled).flatten()
# new_pred = (new_pred_prob > 0.5).astype(int)

