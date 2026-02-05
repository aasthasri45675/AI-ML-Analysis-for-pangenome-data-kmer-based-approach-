from tensorflow import keras
import os

# -----------------------------
# 1. Paths (UPDATE these)
# -----------------------------
# Folder containing saved_model.pb and variables/
saved_model_folder = "/gpfs/home/ekk25vyu/document/samdata/kmer_core_accessory_model"

# Path where the .h5 file will be saved
h5_output_path = "/gpfs/home/ekk25vyu/projects/kmer_model.h5"

# -----------------------------
# 2. Check if folder exists
# -----------------------------
if not os.path.exists(saved_model_folder):
    raise FileNotFoundError(f"SavedModel folder not found: {saved_model_folder}")

# -----------------------------
# 3. Load the SavedModel
# -----------------------------
print("Loading SavedModel...")
model = keras.models.load_model(saved_model_folder)
print("Model loaded successfully!")

# -----------------------------
# 4. Save as HDF5 (.h5)
# -----------------------------
print(f"Saving model as HDF5 to: {h5_output_path}")
model.save(h5_output_path)
print("Model saved as .h5 successfully!")

# -----------------------------
# 5. Optional: Load the .h5 to test
# -----------------------------
# model_h5 = keras.models.load_model(h5_output_path)
# print("H5 model loaded successfully!")

