import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

# =========================
# 1. Folder with your plots
# =========================
folder = "/gpfs/home/ekk25vyu/document/samdata/kmer_sharing_plots"
files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])  # adjust if .jpg

# =========================
# 2. Create multi-panel figure
# =========================
n = len(files)
cols = 3  # Chromosome, Plasmid, Phage
rows = (n + cols - 1) // cols

plt.figure(figsize=(cols*5, rows*4))

for i, file in enumerate(files):
    img = imread(os.path.join(folder, file))
    ax = plt.subplot(rows, cols, i+1)
    ax.imshow(img)
    ax.axis('off')  # hide axes
    ax.set_title(file.replace(".png", ""), fontsize=10)

plt.suptitle("Median k-mer Sharing Across Compartments and k-mer Lengths", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle

# =========================
# 3. Save figure to HPC
# =========================
output_path = "/gpfs/home/ekk25vyu/document/samdata/combined_kmer_sharing.png"
plt.savefig(output_path, dpi=800)  # high resolution
plt.close()

print(f"Combined figure saved at: {output_path}")

