import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import subprocess
import pandas as pd
import numpy as np
from io import StringIO
import os
import scipy.optimize
import scipy.stats
import math
import sys
from main import format_num, exp_decay, ducttape_summary, read_file, joint_modeling_fn


try: os.mkdir('plots_out')
except FileExistsError: pass

tables = []
for idx, plan in enumerate(['data_scaling', 'smaller_bpe']):
    csv = read_file(sys.argv[idx+1]) if len(sys.argv) > idx+1 else ducttape_summary(plan)

    table = pd.read_csv(StringIO(csv), sep="\s+")
    table['bleu_dev'] = pd.to_numeric(table['bleu_dev'], errors='coerce')
    table['ent_dev'] = pd.to_numeric(table['ent_dev'], errors='coerce')
    table['params'] = 2 * table['Layers'] * (4 * table['ModelSize']**2 + 2 * table['ModelSize'] * table['FeedForward'])
    print(f'Missing values: \n {table[table.isnull().any(axis=1)]}')
    table = table.dropna()

    tables.append(table)


for lang in ['deen']: #table['Lang'].unique():
    # print(f'Lang: {lang}')

    fig, axes = plt.subplots(1, 2, figsize=(6,2.5))
    axes[0].set_ylabel("Dev Cross Entropy")
    axes[0].set_title("BPE 30k")
    axes[1].set_title("BPE 2k")

    min_params, max_params = tables[0]['params'].min(), tables[0]['params'].max()
    cmap = mpl.cm.viridis
    norm = mpl.colors.LogNorm(vmin=min_params, vmax=max_params)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm)
    cbar.set_label("# Non-Embedding Parameters")

    for idx, table in enumerate(tables):
        rel = table[table['data_bytes'] > 5242880] # Only use data greater than 5 MB for fitting the curves
        if idx == 0: # TODO: always do this filtering once we have multiple languages with smaller bpes
            rel = rel[rel['Lang'] == lang]

        (a_N, log_N_C, a_D, log_D_C), _ = scipy.optimize.curve_fit(joint_modeling_fn, (rel['data_bytes'], rel['params']), rel['ent_dev'], maxfev=5000, p0=[0.071, math.log(8.8e10), 0.3, math.log(5e6)])

        print(f'Fit: {(a_N, log_N_C, a_D, log_D_C)}')
        if idx == 0: # TODO: always do this filtering once we have multiple languages with smaller bpes
            rel = table[table['Lang'] == lang]
        else:
            rel = table

        for params in sorted(table['params'].unique()):
            params_rel = rel[rel['params'] == params].sort_values('data_bytes')
            scatter = axes[idx].scatter(params_rel['data_bytes'], params_rel['ent_dev'], s=15, color=cmap(norm(params)))
            predicted_line = joint_modeling_fn((params_rel['data_bytes'], params_rel['params']), a_N, log_N_C, a_D, log_D_C)
            # axes[idx].plot(params_rel['data_bytes'], predicted_line, linestyle=':', color=scatter.get_facecolor()[0])

        axes[idx].axvline(5242880, linestyle="--", label="5 MB")
        axes[idx].set_xlabel("Data Size (Bytes)")
        axes[idx].set_xscale('log')
        axes[idx].set_ylim((2,8))
        # axes[idx].set_yscale('log')
        axes[idx].legend()

    # Note: have to set formatters after setting scale
    axes[0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))
    axes[1].yaxis.set_minor_formatter(mticker.StrMethodFormatter('{x:.0f}'))
    fig.savefig(f'plots_out/{lang}_smaller_bpe.png', bbox_inches='tight')
