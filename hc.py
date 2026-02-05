import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import random

# -------------------------------
# Load k-mer presence/absence matrices
# -------------------------------
chrom_df = pd.read_csv(
    '/gpfs/home/ekk25vyu/document/samdata/chr/kmer count-21/matrix/kmer_presence_absence.txt',
    sep='\t'
)
plasmid_df = pd.read_csv(
    '/gpfs/home/ekk25vyu/document/samdata/plasmid/kmer count-21/matrix/kmer_presence_absence.txt',
    sep='\t'
)
phage_df = pd.read_csv(
    '/gpfs/home/ekk25vyu/document/samdata/phage/kmer count-21/matrix/kmer_presence_absence.txt',
    sep='\t'
)

# -------------------------------
# Convert counts to binary presence/absence
# -------------------------------
chrom_df.iloc[:, 1:] = (chrom_df.iloc[:, 1:] >= 1).astype(int)
plasmid_df.iloc[:, 1:] = (plasmid_df.iloc[:, 1:] >= 1).astype(int)
phage_df.iloc[:, 1:] = (phage_df.iloc[:, 1:] >= 1).astype(int)

# -------------------------------
# Set k-mer as index
# -------------------------------
chrom_df = chrom_df.set_index('kmer')
plasmid_df = plasmid_df.set_index('kmer')
phage_df = phage_df.set_index('kmer')

# -------------------------------
# Label genome type
# -------------------------------
chrom_df['genome_type'] = 'core'
plasmid_df['genome_type'] = 'accessory'
phage_df['genome_type'] = 'accessory'

# -------------------------------
# Combine datasets
# -------------------------------
combined_df = pd.concat([chrom_df, plasmid_df, phage_df], axis=0)

# Remove duplicate k-mers
combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

# Separate genome_type
genome_type = combined_df['genome_type']
combined_df = combined_df.drop(columns=['genome_type'])

# -------------------------------
# Subsample 200 kmers
# -------------------------------
if combined_df.shape[0] > 200:
    sampled_indices = random.sample(list(combined_df.index), 200)
    combined_df = combined_df.loc[sampled_indices]
    genome_type = genome_type.loc[sampled_indices]

# -------------------------------
# Compute Jaccard distance and linkage
# -------------------------------
dist_matrix = pdist(combined_df.values, metric='jaccard')

# IMPORTANT: ward is invalid for Jaccard ? use average
linkage_matrix = linkage(dist_matrix, method='average')

# -------------------------------
# Plot dendrogram (biologically correct)
# -------------------------------
plt.figure(figsize=(15, 10))

dendrogram(
    linkage_matrix,
    labels=combined_df.index.tolist(),
    leaf_rotation=90,
    leaf_font_size=8,
    color_threshold=0,              # force single color
    above_threshold_color='black'   # all branches black
)

ax = plt.gca()

# Color leaf labels by biological category
for lbl in ax.get_xmajorticklabels():
    kmer = lbl.get_text()
    if genome_type.loc[kmer] == 'core':
        lbl.set_color('blue')
    else:
        lbl.set_color('red')

# Legend (label color only)
plt.scatter([], [], c='blue', label='Core k-mers')
plt.scatter([], [], c='red', label='Accessory k-mers')
plt.legend(title='K-mer Type (label color)', loc='upper right')

plt.title('Hierarchical clustering of core and accessory k-mers (Jaccard distance)')
plt.xlabel('K-mers')
plt.ylabel('Jaccard distance')

plt.tight_layout()
plt.savefig('core_accessory_dendrogram_200kmers_corrected.png', dpi=300)
plt.close()

print("Dendrogram saved as 'core_accessory_dendrogram_200kmers_corrected.png'")
