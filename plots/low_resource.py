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
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
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

        (a_D, log_D_C), covar = scipy.optimize.curve_fit(modeling_fn, rel['data_bytes'], rel['ent_dev'], maxfev=5000, p0=[0.3, math.log(5e6)])
        a_D_err, log_D_C_err = np.sqrt(np.diagonal(covar))

        print(f'Fit: {(a_D, log_D_C)}')
        scale_params[lang] = (a_D, log_D_C, a_D_err, log_D_C_err)

        # Data vs. Cross-Entropy
        axes[col].set_xlabel("Data Size (Bytes)")
        axes[col].set_title(f'{format_lang(lang)}')
        axes[col].set_xscale('log')

        for params in sorted(table['params'].unique()):
            params_rel = rel[rel['params'] == params].sort_values('data_bytes')
            axes[col].scatter(params_rel['data_bytes'], params_rel['ent_dev'], s=15, color=cmap(norm(params)))

        predicted_line = modeling_fn(params_rel['data_bytes'], a_D, log_D_C)
        axes[col].plot(params_rel['data_bytes'], predicted_line, linestyle=':', color="red")
        # err_min = modeling_fn(params_rel['data_bytes'], a_D - a_D_err, log_D_C - log_D_C_err)
        # err_max = modeling_fn(params_rel['data_bytes'], a_D + a_D_err, log_D_C + log_D_C_err)
        # axes[col].fill_between(params_rel['data_bytes'], err_min, err_max, alpha=0.2, color="red")

    fig.savefig(f'plots_out/lr_data_scaling.png', bbox_inches='tight')
    return scale_params


def plot_bleu(table):
    fig, axes = plt.subplots()
    axes.set_xlabel("Dev Cross Entropy")
    axes.set_ylabel("Dev BLEU")
    axes.set_xlim((table['ent_dev'].max() + 0.5, 1))

    bleu_params = {}
    for lang in table['Lang'].unique():

        print(f'Lang: {lang}')
        rel = table[(table['Lang'] == lang)]

        scatter = axes.scatter(rel['ent_dev'], rel['bleu_dev'], s=15, label=f'{lang[0:2]+"-"+lang[2:]}')

        (C, k), covar = scipy.optimize.curve_fit(exp_decay, rel['ent_dev'], rel['bleu_dev'], maxfev=5000, p0=[1, -0.5])
        C_err, k_err = np.sqrt(np.diagonal(covar))
        print(f"C: {C}, k: {k}")
        bleu_params[lang] = (C, k, C_err, k_err)
        lb = math.log(50 / C) / k # Plot lines to max BLEU of 50
        lin = np.linspace(lb, 6)
        axes.plot(lin, exp_decay(lin, C, k), linestyle=":", color=scatter.get_facecolor()[0])
        axes.fill_between(lin, exp_decay(lin, C-C_err, k-k_err), exp_decay(lin, C+C_err, k+k_err), alpha=0.2, color=scatter.get_facecolor()[0])

    axes.legend()

    fig.savefig(f'plots_out/lr_bleu_vs_xent.png', bbox_inches='tight')
    return bleu_params


def plot_dollars_bleu(table, scale_params, bleu_params):
    fig, axes = plt.subplots()
    axes.set_xlabel("Approximate Dollars to Label Extra Data")
    axes.set_ylabel("Dev BLEU")
    axes.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: '$'+format_num(x)))

    for lang in table['Lang'].unique():
        if lang == "soen":
            continue
        rel = table[(table['Lang'] == lang)].sort_values('data_bytes')
        (C, k, _, _) = bleu_params[lang]
        (a_D, log_D_C, _, _) = scale_params[lang]

        def data_to_bleu(data, C, K, a_D):
            return C * np.exp(K / (data**a_D))

        (C, K, a_D), covar = scipy.optimize.curve_fit(data_to_bleu, rel['data_bytes'], rel['bleu_dev'], maxfev=5000, p0=[C, k * (np.exp(log_D_C)**a_D), a_D])
        C_err, K_err, a_D_err = np.sqrt(np.diagonal(covar))

        max_data = rel['data_bytes'].max()
        rel['dollars'] = (rel['data_bytes'] - max_data) / 100
        scatter = axes.scatter(rel['dollars'], rel['bleu_dev'], label=lang)
        axes.plot(rel['dollars'], data_to_bleu(rel['data_bytes'], C, K, a_D), color=scatter.get_facecolor()[0])
        # min_err = data_to_bleu(rel['data_bytes'], C-C_err, K-K_err, a_D-a_D_err)
        # max_err = data_to_bleu(rel['data_bytes'], C+C_err, K+K_err, a_D+a_D_err)
        # axes.fill_between(rel['data_bytes'], min_err, max_err, alpha=0.2, color=scatter.get_facecolor()[0])

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
    table['params'] = 2 * 6 * (4 * 512**2 + 2 * 512 * 2048)

    print(f'Missing values: \n {table[table.isnull().any(axis=1)]}')
    table = table.dropna()

    scale_params = plot_data_scaling(table)
    bleu_params = plot_bleu(table)
    plot_dollars_bleu(table, scale_params, bleu_params)
