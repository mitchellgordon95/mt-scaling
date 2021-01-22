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
from main import format_num, format_lang, exp_decay, read_file, ducttape_summary


def modeling_fn(data, a_D, log_D_C):
    return (math.exp(log_D_C) / data)**a_D


def plot_data_scaling(table):
    fig, axes = plt.subplots(1, 3, figsize=(8,2.5))
    axes[0].set_ylabel("Dev Cross Entropy")

    for ax in axes:
        ax.set_yscale('log')
        ax.yaxis.set_minor_formatter(mticker.StrMethodFormatter('{x:.0f}'))

    min_params, max_params = table['params'].min(), table['params'].max()
    cmap = mpl.cm.viridis
    norm = mpl.colors.LogNorm(vmin=min_params, vmax=max_params)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm)
    cbar.set_label("# Non-Embedding Parameters")

    col = -1
    scale_params = {}
    for lang in table['Lang'].unique():
        col += 1

        print(f'Lang: {lang}')
        rel = table[(table['Lang'] == lang)]

        (a_D, log_D_C), _ = scipy.optimize.curve_fit(modeling_fn, rel['data_bytes'], rel['ent_dev'], maxfev=5000, p0=[0.3, math.log(5e6)])

        print(f'Fit: {(a_D, log_D_C)}')
        scale_params[lang] = (a_D, log_D_C)

        # Data vs. Cross-Entropy
        axes[col].set_xlabel("Data Size (Bytes)")
        axes[col].set_title(f'{format_lang(lang)}')
        axes[col].set_xscale('log')

        for params in sorted(table['params'].unique()):
            params_rel = rel[rel['params'] == params].sort_values('data_bytes')
            scatter = axes[col].scatter(params_rel['data_bytes'], params_rel['ent_dev'], s=15, color=cmap(norm(params)))

        predicted_line = modeling_fn(params_rel['data_bytes'], a_D, log_D_C)
        axes[col].plot(params_rel['data_bytes'], predicted_line, linestyle=':', color="red")

    fig.savefig(f'plots_out/lr_data_scaling.png', bbox_inches='tight')
    return scale_params


def plot_bleu(table):
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

    fig.savefig(f'plots_out/lr_bleu_vs_xent.png', bbox_inches='tight')
    return bleu_params


def plot_dollars_bleu(table, scale_params, bleu_params):
    fig, axes = plt.subplots()
    axes.set_xlabel("Approximate Dollars to Label Extra Data")
    axes.set_ylabel("Max Dev BLEU Possible")
    axes.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: '$'+format_num(x)))

    for lang in table['Lang'].unique():
        rel = table[(table['Lang'] == lang)]
        (C, k) = bleu_params[lang]
        (a_D, log_D_C) = scale_params[lang]

        dollars = np.linspace(1, 100000)
        max_data = rel['data_bytes'].max()
        axes.plot(dollars, exp_decay(modeling_fn(dollars * 100 + max_data, a_D, log_D_C), C, k), label=lang)

    axes.legend()
    fig.savefig(f'plots_out/lr_dollars_to_bleu.png', bbox_inches='tight')


if __name__ == "__main__":
    try: os.mkdir('plots_out')
    except FileExistsError: pass

    csv = read_file(sys.argv[1]) if len(sys.argv) > 1 else ducttape_summary('low_resource')

    table = pd.read_csv(StringIO(csv), sep="\s+")
    table['bleu_dev'] = pd.to_numeric(table['bleu_dev'], errors='coerce')
    table['ent_dev'] = pd.to_numeric(table['ent_dev'], errors='coerce')
    table['data_bytes'] = pd.to_numeric(table['data_bytes'], errors='coerce')
    table['params'] = 2 * table['Layers'] * (4 * 512**2 + 2 * 512 * 2048)

    print(f'Missing values: \n {table[table.isnull().any(axis=1)]}')
    table = table.dropna()

    scale_params = plot_data_scaling(table)
    bleu_params = plot_bleu(table)
    plot_dollars_bleu(table, scale_params, bleu_params)
