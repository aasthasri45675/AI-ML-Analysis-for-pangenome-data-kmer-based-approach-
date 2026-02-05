#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import csv
import os
import shutil
import tensorflow as tf

# ----------------------------
# 1. Arguments
# ----------------------------
parser = argparse.ArgumentParser(description="CNN prediction (TFLite, CPU-safe, HPC-friendly)")
parser.add_argument("--input", required=True, help="Input TXT/CSV for k-mers")
parser.add_argument("--model", required=True, help="Path to TFLite model (.tflite)")
parser.add_argument("--output", required=True, help="Output CSV for CNN predictions")
args = parser.parse_args()

# ----------------------------
# 2. Copy TFLite model to /tmp only if needed
# ----------------------------
tmp_model_path = f"/tmp/{os.path.basename(args.model)}"
if os.path.abspath(args.model) != os.path.abspath(tmp_model_path):
    shutil.copy2(args.model, tmp_model_path)

# ----------------------------
# 3. Load TFLite model
# ----------------------------
interpreter = tf.lite.Interpreter(model_path=tmp_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
max_len = input_details[0]['shape'][1]

# ----------------------------
# 4. One-hot encoding for a single k-mer
# ----------------------------
def encode_kmer_onehot(seq, max_len=max_len):
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    X = np.zeros((1, max_len, 4), dtype=np.float32)
    seq = seq[:max_len]
    for j, base in enumerate(seq):
        if base in mapping:
            X[0, j, mapping[base]] = 1.0
    return X

# ----------------------------
# 5. Prepare output CSV
# ----------------------------
os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w", newline="") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["kmer_seq", "core_accessory", "core_accessory_prob"])

# ----------------------------
# 6. Predict line by line (CPU-safe)
# ----------------------------
for chunk in pd.read_csv(args.input, sep="\t", chunksize=1):
    # Ensure column is named correctly
    if chunk.columns[0] != "kmer_seq":
        chunk.rename(columns={chunk.columns[0]: "kmer_seq"}, inplace=True)

    seq = chunk.iloc[0]["kmer_seq"]
    X = encode_kmer_onehot(seq)

    # Run TFLite inference
    interpreter.set_tensor(input_details[0]['index'], X)
    interpreter.invoke()
    prob = interpreter.get_tensor(output_details[0]['index'])[0][0]
    label = int(prob > 0.5)

    # Append immediately to CSV
    with open(args.output, "a", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([seq, label, prob])

print(f"TFLite CNN predictions completed. Results saved to: {args.output}")
