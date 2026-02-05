import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
import os

# -----------------------------
# 1️⃣ Functions
# -----------------------------

# Core vs Accessory percentages
def core_accessory_stats(df):
    n_strains = df.shape[1]
    row_sum = df.sum(axis=1)
    total = len(row_sum)
    core = (row_sum == n_strains).sum()
    accessory = ((row_sum >= 1) & (row_sum < n_strains)).sum()
    return core/total*100, accessory/total*100

# Median sharing
def median_sharing(df):
    return df.sum(axis=1).median()

# MGE enrichment (plasmid/phage vs chromosome)
def mge_enrichment(df_chrom, df_mge):
    df_chrom.index = df_chrom.index.str.strip()
    df_mge.index = df_mge.index.str.strip()
    common_kmers = df_chrom.index.intersection(df_mge.index)
    enriched = []
    for km in common_kmers:
        table = [[df_mge.loc[km].sum(), df_mge.shape[1]-df_mge.loc[km].sum()],
                 [df_chrom.loc[km].sum(), df_chrom.shape[1]-df_chrom.loc[km].sum()]]
        _, p = fisher_exact(table, alternative='greater')
        if p < 0.05:
            enriched.append(km)
    return len(enriched)

# Plot k-mer sharing distribution
def plot_sharing(df, k, compartment):
    row_sum = df.sum(axis=1)
    counts = row_sum.value_counts().sort_index()
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(6,4))
    counts.plot(kind='bar')
    plt.title(f"k-mer sharing distribution: k={k}, {compartment}")
    plt.xlabel("Number of strains sharing k-mer")
    plt.ylabel("Number of k-mers")
    plt.tight_layout()
    plt.savefig(f"plots/k{k}_{compartment}_sharing.png")
    plt.close()

# -----------------------------
# 2️⃣ GPFS file paths
# Update paths as necessary
# -----------------------------
files = {
    19: {'Chromosome': '/gpfs/home/ekk25vyu/document/samdata/chr/Ker count -19/matrix/kmer_presence_absence.txt',
         'Plasmid': '/gpfs/home/ekk25vyu/document/samdata/plasmid/kmer count-19/matrix/kmer_presence_absence.txt',
         'Phage': '/gpfs/home/ekk25vyu/document/samdata/phage/kmer counts -19/matrix/kmer_presence_absence.txt'},
    21: {'Chromosome': '/gpfs/home/ekk25vyu/document/samdata/chr/kmer count-21/matrix/kmer_presence_absence.txt',
         'Plasmid': '/gpfs/home/ekk25vyu/document/samdata/plasmid/kmer count-21/matrix/kmer_presence_absence.txt',
         'Phage': '/gpfs/home/ekk25vyu/document/samdata/phage/kmer count-21/matrix/kmer_presence_absence.txt'},
    24: {'Chromosome': '/gpfs/home/ekk25vyu/document/samdata/chr/kmer count -24/matrix/kmer_presence_absence.txt',
         'Plasmid': '/gpfs/home/ekk25vyu/document/samdata/plasmid/kmer count-24/matrix/kmer_presence_absence.txt',
         'Phage': '/gpfs/home/ekk25vyu/document/samdata/phage/kmer counts-24/matrix/kmer_presence_absence.txt'}
}

# -----------------------------
# 3️⃣ Run data profiling
# -----------------------------
results = []

for k, comps in files.items():
    dfs = {}
    # Load files
    for c, path in comps.items():
        df = pd.read_csv(path, sep="\t", index_col=0)
        df.index = df.index.str.strip()
        dfs[c] = df
        # Plot sharing
        plot_sharing(df, k, c)
    
    # Core vs Accessory & median sharing
    for c, df in dfs.items():
        core_pct, acc_pct = core_accessory_stats(df)
        median_s = median_sharing(df)
        results.append({
            "k": k,
            "Compartment": c,
            "Core (%)": round(core_pct,2),
            "Accessory (%)": round(acc_pct,2),
            "Median_Sharing": round(median_s,2)
        })
    
    # MGE enrichment for plasmid/phage
    for comp in ["Plasmid", "Phage"]:
        enriched_count = mge_enrichment(dfs['Chromosome'], dfs[comp])
        # Update the results table
        idx = next(i for i, r in enumerate(results) if r["k"]==k and r["Compartment"]==comp)
        results[idx]["Enriched_kmers"] = enriched_count

# Chromosome has no MGE enrichment
for r in results:
    if r["Compartment"]=="Chromosome":
        r["Enriched_kmers"] = 0

# -----------------------------
# 4️⃣ Save combined table
# -----------------------------
combined_df = pd.DataFrame(results)
# Sort
comp_order = ["Chromosome", "Plasmid", "Phage"]
combined_df['Compartment'] = pd.Categorical(combined_df['Compartment'], categories=comp_order, ordered=True)
combined_df = combined_df.sort_values(by=['k','Compartment'])

combined_df.to_csv("kmer_data_profiling_summary.csv", index=False)
print(combined_df)

