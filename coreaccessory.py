import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# =============================
# 1. Load combined k-mer data
# =============================
path = "/gpfs/home/ekk25vyu/document/samdata/core_all_strain.txt"
df = pd.read_csv(path, sep="\t")  # or sep=" " depending on your file
df.set_index('kmer', inplace=True)

# Features: presence/absence per strain
X = df.values.astype(np.float32)

# Label: 1 if present in all strains (core), 0 otherwise (accessory)
y = (X.min(axis=1) == 1).astype(np.float32)

# =============================
# 2. Train/validation split
# =============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =============================
# 3. Build model
# =============================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# =============================
# 4. Train
# =============================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=1
)

# =============================
# 5. Evaluate & save
# =============================
loss, acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {acc:.3f}")

model.save("core_accessory_model")



