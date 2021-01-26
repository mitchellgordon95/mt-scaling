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


def format_num(num):
    sign = "-" if num < 0 else ""
    num = abs(num)
    if num > 1000 and num < 1000000:
        return f'{sign}{int(num / 1000)}K'
    if num >= 1000000:
        return f'{sign}{int(num / 1000000)}M'
    return str(num)


def format_lang(lang):
    return {
        "deen": "German-English",
        "ruen": "Russian-English",
        "zhen": "Chinese-English",
        "tlen": "Tagalog-English",
        "soen": "Somali-English",
        "swen": "Swahili-English"
    }.get(lang)


# Joint param/data loss modeling
def joint_modeling_fn(lines_params, a_N, log_N_C, a_D, log_D_C):
    lines, params = lines_params
    return ((math.exp(log_N_C) / params)**(a_N / a_D) + math.exp(log_D_C) / lines)**a_D


# Exp decay fit
def exp_decay(ent, C, k):
    return C * np.exp(k * ent)


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
        rel = table[(table['Lang'] == lang) & (table['data_bytes'] > 5242880)]

        (a_N, log_N_C, a_D, log_D_C), _ = scipy.optimize.curve_fit(joint_modeling_fn, (rel['data_bytes'], rel['params']), rel['ent_dev'], maxfev=5000, p0=[0.071, math.log(8.8e10), 0.3, math.log(5e6)])

        print(f'Fit: {(a_N, log_N_C, a_D, log_D_C)}')
        scale_params[lang] = (a_N, log_N_C, a_D, log_D_C)

        # Data vs. Cross-Entropy
        axes[col].set_xlabel("Data Size (Bytes)")
        axes[col].set_title(f'{format_lang(lang)}')
        axes[col].set_xscale('log')

        for params in sorted(table['params'].unique()):
            params_rel = rel[rel['params'] == params].sort_values('data_bytes')
            scatter = axes[col].scatter(params_rel['data_bytes'], params_rel['ent_dev'], s=15, color=cmap(norm(params)))
            predicted_line = joint_modeling_fn((params_rel['data_bytes'], params_rel['params']), a_N, log_N_C, a_D, log_D_C)
            axes[col].plot(params_rel['data_bytes'], predicted_line, linestyle=':', color=scatter.get_facecolor()[0])

        # Infinite params line
        inf_params = np.array([float('inf')] * params_rel['data_bytes'].size)
        inf_params_line = joint_modeling_fn((params_rel['data_bytes'], inf_params), a_N, log_N_C, a_D, log_D_C)
        print(inf_params_line)
        # axes[col].plot(params_rel['data_bytes'], inf_params_line, color='purple', label='Inf Params')

    fig.savefig(f'plots_out/data_scaling.png', bbox_inches='tight')
    return scale_params


def plot_bleu(table):
    table = table[table['data_bytes'] > 5242880]
    fig, axes = plt.subplots(1, 2, figsize=(6,3))
    axes[0].set_xlabel("Dev Cross Entropy")
    axes[0].set_ylabel("Dev BLEU")
    axes[0].set_xlim((table['ent_dev'].max() + 0.5, 0))

    bleu_params = {}
    for lang in table['Lang'].unique():

        print(f'Lang: {lang}')
        rel = table[(table['Lang'] == lang)]

        scatter = axes[0].scatter(rel['ent_dev'], rel['bleu_dev'], s=15, label=f'{lang[0:2]+"-"+lang[2:]}')

        (C, k), _ = scipy.optimize.curve_fit(exp_decay, rel['ent_dev'], rel['bleu_dev'], maxfev=5000, p0=[1, -0.5])
        print(f"C: {C}, k: {k}")
        bleu_params[lang] = (C, k)
        lb = math.log(50 / C) / k # Plot lines to max BLEU of 50
        lin = np.linspace(lb, 6)
        axes[0].plot(lin, exp_decay(lin, C, k), linestyle=":", color=scatter.get_facecolor()[0])

    axes[0].legend()

    # HPO Experiments
    evals = pd.read_csv(f'hpo_nmt/datasets/ja-en.evals', sep="\t", names=["dev_bleu", "dev_gpu_time", "dev_ppl", "num_updates", "gpu_memory", "num_param"])
    hyps = pd.read_csv(f'hpo_nmt/datasets/ja-en.hyps', sep="\t", names=["bpe_symbols", "num_layers", "num_embed", "transformer_feed_forward_num_hidden", "transformer_attention_heads", "initial_learning_rate"])
    data = pd.concat([evals, hyps], axis=1)
    data['cross_ent'] = data['dev_ppl'].map(math.log)
    data = data[data['dev_bleu'] > 11]

    bpe_vals = data['bpe_symbols'].unique()

    axes[1].set_xlabel("Dev Cross-Entropy")
    axes[1].set_xlim((data['cross_ent'].max()+0.2, 1.5))

    for bpe in sorted(bpe_vals):
        print("ja-en BPE {bpe}")
        rel = data[data['bpe_symbols'] == bpe]
        scatter = axes[1].scatter(rel['cross_ent'], rel['dev_bleu'], s=15, label=f'ja-en BPE {format_num(bpe)}')

        (C, k), _ = scipy.optimize.curve_fit(exp_decay, rel['cross_ent'], rel['dev_bleu'], maxfev=5000, p0=[1, -0.5])
        print(f"C: {C}, k: {k}")
        lb = math.log(20 / C) / k # Plot lines to max BLEU of 20
        lin = np.linspace(lb, 6)
        axes[1].plot(lin, exp_decay(lin, C, k), linestyle=":", color=scatter.get_facecolor()[0])

    axes[1].legend()
    fig.savefig(f'plots_out/bleu_vs_xent.png', bbox_inches='tight')
    return bleu_params


def plot_dollars_bleu(table, scale_params, bleu_params):
    fig, axes = plt.subplots()
    axes.set_xlabel("Approximate Dollars to Label Data")
    axes.set_ylabel("Max Dev BLEU Possible")
    axes.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: '$'+format_num(x)))

    for lang in table['Lang'].unique():
        rel = table[(table['Lang'] == lang)]

        (C, k) = bleu_params[lang]
        (a_N, log_N_C, a_D, log_D_C) = scale_params[lang]

        def modeling_fn(data, a_D, log_D_C):
            return (math.exp(log_D_C) / data)**a_D

        axes.plot(rel['data_bytes'].sort_values() / 100, exp_decay(modeling_fn(rel['data_bytes'].sort_values(), a_D, log_D_C), C, k), label=lang)

    axes.legend()
    fig.savefig(f'plots_out/dollars_to_bleu.png', bbox_inches='tight')


def ducttape_summary(plan):
    ducttape = subprocess.Popen(["ducttape", "main.tape", "-C", "main.tconf", "-p", plan, "summary"], stdout=subprocess.PIPE)
    tabular = subprocess.Popen(['tabular'], stdin=ducttape.stdout, stdout=subprocess.PIPE)
    return subprocess.check_output(["grep", "-o", "^[^#]*"], stdin=tabular.stdout).decode('ascii')


def read_file(fname):
    with open(fname, "r")  as fd:
        csv = ""
        for line in fd.readlines():
            if line[0] != "#":
                csv += line
    return csv


if __name__ == "__main__":
    try: os.mkdir('plots_out')
    except FileExistsError: pass

    csv = read_file(sys.argv[1]) if len(sys.argv) > 1 else ducttape_summary('data_scaling')

    table = pd.read_csv(StringIO(csv), sep="\s+")
    table['bleu_dev'] = pd.to_numeric(table['bleu_dev'], errors='coerce')
    table['ent_dev'] = pd.to_numeric(table['ent_dev'], errors='coerce')
    table['data_bytes'] = pd.to_numeric(table['data_bytes'], errors='coerce')
    table['params'] = 2 * table['Layers'] * (4 * table['ModelSize']**2 + 2 * table['ModelSize'] * table['FeedForward'])

    print(f'Missing values: \n {table[table.isnull().any(axis=1)]}')
    table = table.dropna()

    scale_params = plot_data_scaling(table)
    bleu_params = plot_bleu(table)
    plot_dollars_bleu(table, scale_params, bleu_params)
