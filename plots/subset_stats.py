import matplotlib.pyplot as plt
import subprocess
import pandas as pd
from io import StringIO
import os
import sys
import matplotlib.ticker as mtick

try: os.mkdir('plots_out')
except FileExistsError: pass

if len(sys.argv) <= 1:
    ducttape = subprocess.Popen(["ducttape", "main.tape", "-C", "main.tconf", "-p", 'subset_stats', "summary"], stdout=subprocess.PIPE)
    tabular = subprocess.Popen(['tabular'], stdin=ducttape.stdout, stdout=subprocess.PIPE)
    csv = subprocess.check_output(["grep", "-o", "^[^#]*"], stdin=tabular.stdout).decode('ascii')
else:
    with open(sys.argv[1], "r")  as fd:
        csv = ""
        for line in fd.readlines():
            if line[0] != "#":
                csv += line

table = pd.read_csv(StringIO(csv), sep="\s+")
table['src_coverage'] = pd.to_numeric(table['src_coverage'], errors='coerce')
table['trg_coverage'] = pd.to_numeric(table['trg_coverage'], errors='coerce')
table['vocab_overlap_src'] = pd.to_numeric(table['vocab_overlap_src'], errors='coerce')
table['vocab_overlap_trg'] = pd.to_numeric(table['vocab_overlap_trg'], errors='coerce')

for lang in table['Lang'].unique():

    print(f'Lang: {lang}')
    rel = table[table['Lang'] == lang].sort_values('data_bytes')

    fig, axes = plt.subplots()
    axes.set_xlabel("Data Size (Bytes)")
    axes.set_xscale('log')
    axes.yaxis.set_major_formatter(mtick.PercentFormatter())
    axes.set_ylim((0,101))
    axes.set_title(f'Training Data Subset Statistics {lang}')

    axes.plot(rel['data_bytes'], (rel['src_coverage'] + rel['trg_coverage']) / 2 * 100, label="Development Tokens Seen During Training")
    axes.plot(rel['data_bytes'], (rel['vocab_overlap_src'] + rel['vocab_overlap_trg']) / 2 * 100, label="Subset BPE Vocab Overlap w/ Full BPE Vocab")

    axes.legend()
    fig.savefig(f'plots_out/{lang}_dev_coverage.png')
