import pandas as pd
import matplotlib
matplotlib.use('Agg')  # HPC-friendly backend
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Step 1: Read k-mer presence/absence files
# -----------------------------
df_chr = pd.read_csv("chr/kmer count-21/matrix/kmer_presence_absence.txt", sep="\t", index_col=0)
df_pl  = pd.read_csv("plasmid/kmer count-21/matrix/kmer_presence_absence.txt", sep="\t", index_col=0)
df_ph  = pd.read_csv("phage/kmer count-21/matrix/kmer_presence_absence.txt", sep="\t", index_col=0)

# -----------------------------
# Step 2: Create environment x k-mer frequency table
# -----------------------------
freq_matrix = pd.DataFrame({
    'chromosome': df_chr.sum(axis=1),
    'plasmid': df_pl.sum(axis=1),
    'phage': df_ph.sum(axis=1)
})

# Transpose for heatmap: rows=environment, columns=k-mers
heat_data = freq_matrix.T

# -----------------------------
# Step 3: Plot heatmap
# -----------------------------
plt.figure(figsize=(24, 10), dpi=300)

sns.heatmap(
    heat_data,
    cmap="Blues",
    cbar_kws={'label': 'K-mer counts'},
    linewidths=0.2,
    rasterized=True
)

plt.xlabel("K-mers", fontsize=14)
plt.ylabel("Environments", fontsize=14)
plt.title("K-mer Frequency Heatmap Across Environments", fontsize=16)

# Reduce x-axis tick density
plt.xticks(
    ticks=range(0, heat_data.shape[1], 10),
    labels=heat_data.columns[::10],
    rotation=90,
    fontsize=8
)

plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig("kmer_frequency_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
