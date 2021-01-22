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
from main import ducttape_summary, read_file, exp_decay


def plot_detok(table):
    table = table[table['data_bytes'] > 5242880]
    fig, axes = plt.subplots()
    axes.set_xlabel("Dev Cross Entropy")
    axes.set_ylabel("Dev BLEU")
    axes.set_xlim((table['ent_dev'].max() + 0.5, 0))

    bleu_params = {}
    for lang in table['Lang'].unique():

        print(f'Lang: {lang}')
        rel = table[(table['Lang'] == lang)]

        scatter = axes.scatter(rel['ent_dev'], rel['bleu_dev'], s=15, label=f'{lang[0:2]+"-"+lang[2:]}')

        (C, k), _ = scipy.optimize.curve_fit(exp_decay, rel['ent_dev'], rel['bleu_dev'], maxfev=5000, p0=[1, -0.5])
        print(f"C: {C}, k: {k}")
        bleu_params[lang] = (C, k)
        lb = math.log(50 / C) / k # Plot lines to max BLEU of 50
        lin = np.linspace(lb, 6)
        axes.plot(lin, exp_decay(lin, C, k), linestyle=":", color=scatter.get_facecolor()[0])

    axes.legend()

    fig.savefig(f'plots_out/detok_bleu_vs_xent.png', bbox_inches='tight')
    return bleu_params


if __name__ == "__main__":
    try: os.mkdir('plots_out')
    except FileExistsError: pass

    csv = read_file(sys.argv[1]) if len(sys.argv) > 1 else ducttape_summary('sacrebleu')

    table = pd.read_csv(StringIO(csv), sep="\s+")

    table['bleu_dev'] = pd.to_numeric(table['bleu_dev'], errors='coerce')
    table['ent_dev'] = pd.to_numeric(table['ent_dev'], errors='coerce')
    table['data_bytes'] = pd.to_numeric(table['data_bytes'], errors='coerce')
    table['params'] = 2 * table['Layers'] * (4 * table['ModelSize']**2 + 2 * table['ModelSize'] * table['FeedForward'])

    # Need this hack because sacrebleu evals happen on a different branch, which means bleu and xent each get their own row
    # (but every other column is the same). This works for summaries/sacrebleu.0121.summary
    bleus = table[table['bleu_dev'].isnull() != True].drop(['ent_dev', 'data_bytes'], axis=1)
    xents = table[table['ent_dev'].isnull() != True].drop(['bleu_dev'], axis=1)
    table = pd.merge(bleus, xents, how='left', on=['DataPercent', 'FeedForward', 'Lang', 'Layers', 'ModelSize', 'params'])

    print(f'Missing values: \n {table[table.isnull().any(axis=1)]}')
    table = table.dropna()

    plot_detok(table)
