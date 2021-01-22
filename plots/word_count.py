import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from io import StringIO
import os
import sys
from main import read_file, ducttape_summary, format_num

try: os.mkdir('plots_out')
except FileExistsError: pass

csv = read_file(sys.argv[1]) if len(sys.argv) > 1 else ducttape_summary('word_count')
table = pd.read_csv(StringIO(csv), sep="\s+")
table['unique_word_count'] = pd.to_numeric(table['unique_word_count'], errors='coerce')
table['data_bytes'] = pd.to_numeric(table['data_bytes'], errors='coerce')

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(5,2.5))
axes[0].set_ylabel("Unique Words")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_num(x)))
axes[0].set_title("30k BPE")
axes[1].set_title("2k BPE")

table = table[table['DataPercent'] < 25]

for idx, bpe in enumerate((30000, 2000)):
    for lang in table['Lang'].unique():
        rel = table[(table['Lang'] == lang) & (table['BPE'] == bpe)].sort_values('data_bytes')
        axes[idx].plot(rel['data_bytes'], rel['unique_word_count'], label=lang)
    axes[idx].axvline(5242880, linestyle="--", label="5 MB")
    axes[idx].set_ylim((0,None))
    axes[idx].set_xlabel("Dataset Size (Bytes)")
    axes[idx].set_xscale('log')
    axes[idx].legend()

fig.savefig(f'plots_out/unique_words.png', bbox_inches="tight")
