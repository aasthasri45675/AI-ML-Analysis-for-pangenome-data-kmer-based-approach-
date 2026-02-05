import pandas as pd
import glob
import os

# ensure matrix folder exists
os.makedirs("matrix", exist_ok=True)

# get all .txt files in this folder
files = glob.glob("*.txt")

dfs = []
for f in files:
    df = pd.read_csv(f, sep="\t", header=None, names=["kmer","count"])
    df["presence"] = 1
    df = df[["kmer","presence"]].set_index("kmer")
    df.columns = [f.replace(".txt","")]
    dfs.append(df)

# merge all genomes into a single matrix
matrix = pd.concat(dfs, axis=1).fillna(0).astype(int)

# save matrix
matrix.to_csv("matrix/kmer_presence_absence.txt", sep="\t")
print("Presence/absence matrix saved to matrix/kmer_presence_absence.txt")

