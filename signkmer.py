import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
import glob
import os

# -----------------------------
# Step 0: Define file paths
# -----------------------------
chr_file = "chr/kmer count-21/matrix/kmer_presence_absence.txt"
pl_file  = "plasmid/kmer count-21/matrix/kmer_presence_absence.txt"
ph_file  = "phage/kmer count-21/matrix/kmer_presence_absence.txt"

# -----------------------------
# Step 1: Read files
# -----------------------------
df_chr = pd.read_csv(chr_file, sep="\t").set_index('kmer')
df_pl  = pd.read_csv(pl_file, sep="\t").set_index('kmer')
df_ph  = pd.read_csv(ph_file, sep="\t").set_index('kmer')

# -----------------------------
# Step 2: Merge all k-mers (outer join)
# -----------------------------
df_all = df_chr.join(df_pl, how='outer').join(df_ph, how='outer')
df_all = df_all.fillna(0)  # missing k-mers = 0

# Identify which columns belong to each environment
chr_cols = df_chr.columns.tolist()
pl_cols  = df_pl.columns.tolist()
ph_cols  = df_ph.columns.tolist()

# -----------------------------
# Step 3: Chi-square per k-mer
# -----------------------------
results = []
for kmer, row in df_all.iterrows():
    present_chr = row[chr_cols].sum()
    absent_chr = len(chr_cols) - present_chr
    present_pl = row[pl_cols].sum()
    absent_pl = len(pl_cols) - present_pl
    present_ph = row[ph_cols].sum()
    absent_ph = len(ph_cols) - present_ph

    # skip k-mers with no counts
    if (present_chr + present_pl + present_ph) == 0:
        continue

    # contingency table
    table = np.array([
        [present_chr, present_pl, present_ph],
        [absent_chr, absent_pl, absent_ph]
    ]) + 0.5  # pseudocount

    chi2, p, _, _ = chi2_contingency(table)
    results.append((kmer, p, present_chr, present_pl, present_ph))

# -----------------------------
# Step 4: Multiple testing correction
# -----------------------------
res_df = pd.DataFrame(results, columns=['kmer','pvalue','chr_count','plasmid_count','phage_count'])
res_df['adj_p'] = multipletests(res_df['pvalue'], method='fdr_bh')[1]

# -----------------------------
# Step 5: Filter significant k-mers and label environment
# -----------------------------
sig_df = res_df[res_df['adj_p'] < 0.05].copy()
if sig_df.empty:
    print("No significant k-mers found.")
else:
    sig_df['enriched_env'] = sig_df[['chr_count','plasmid_count','phage_count']].idxmax()
    sig_df['enriched_env'] = sig_df['enriched_env'].replace({
        'chr_count':'chromosome',
        'plasmid_count':'plasmid',
        'phage_count':'phage'
    })

    sig_df.sort_values('adj_p', inplace=True)
    sig_df.to_csv('significant_kmers_env.csv', index=False)
    print(f"{len(sig_df)} significant k-mers saved to 'significant_kmers_env.csv'")

