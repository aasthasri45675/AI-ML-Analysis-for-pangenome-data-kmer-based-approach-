import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib
from tensorflow.keras.callbacks import EarlyStopping
matplotlib.use('Agg')  # Safe backend for servers
import matplotlib.pyplot as plt

# ==============================
# 1. Load and label data
# ==============================

# Core k-mers → label 1
core = pd.read_csv(
    "core_kmers_all_strains.txt",
    sep=" ",
    header=None,
    names=["kmer", "flag"]
)
core["label"] = 1
core = core[["kmer", "label"]]

# Accessory k-mers → label 0
accessory = pd.read_csv(
    "accessory_kmers_one_strain.txt",
    sep=" ",
    header=None,
    names=["kmer", "count"]
)
accessory["label"] = 0
accessory = accessory[["kmer", "label"]]

# Balance classes
n = min(len(core), len(accessory))
dataset = pd.concat([
    core.sample(n, random_state=42),
    accessory.sample(n, random_state=42)
]).reset_index(drop=True)

print("Class counts:")
print(dataset["label"].value_counts())

# ==============================
# 2. Reverse complement augmentation
# ==============================

def revcomp(seq):
    comp = str.maketrans("ACGT", "TGCA")
    return seq.translate(comp)[::-1]

aug = dataset.copy()
aug["kmer"] = aug["kmer"].apply(revcomp)
dataset = pd.concat([dataset, aug]).reset_index(drop=True)

# ==============================
# 3. One-hot encoding
# ==============================

BASE2IDX = {"A":0, "C":1, "G":2, "T":3}

def one_hot(seq):
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq):
        arr[i, BASE2IDX[b]] = 1.0
    return arr

X = np.stack(dataset["kmer"].apply(one_hot))
y = dataset["label"].values.astype(np.float32)

print("Input shape:", X.shape)

# ==============================
# 4. Train / validation split
# ==============================

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==============================
# 5. TensorFlow CNN model
# ==============================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], 4)),
    tf.keras.layers.Conv1D(64, 5, activation="relu"),
    tf.keras.layers.Conv1D(128, 3, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(128, activation="sigmoid"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# 6. Train model (once)
# ==============================

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights= True 
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=256,
    verbose=1
)

# ==============================
# 7. Save model with correct extension
# ==============================

model.save("kmer_core_accessory_model.keras")
print("Model saved as kmer_core_accessory_model.keras")

# ==============================
# 8. Evaluate model
# ==============================

train_scores = model.evaluate(X_train, y_train, verbose=0)
val_scores = model.evaluate(X_val, y_val, verbose=0)
print("Training Accuracy: %.2f%%" % (train_scores[1]*100))
print("Validation Accuracy: %.2f%%" % (val_scores[1]*100))

# ==============================
# 9. Plot and save training & validation curves
# ==============================

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='x')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='x')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_validation_curve.png", dpi=300, bbox_inches='tight')
print("Graph saved as training_validation_curve.png")

