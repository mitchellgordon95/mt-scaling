import matplotlib.pyplot as plt
import subprocess
import pandas as pd
import numpy as np
from io import StringIO
import os
import scipy.optimize
import scipy.stats
import math

try: os.mkdir('plots_out')
except FileExistsError: pass

ducttape = subprocess.Popen(["ducttape", "main.tape", "-C", "main.tconf", "-p", 'sim_low_resource', "summary"], stdout=subprocess.PIPE)
tabular = subprocess.Popen(['tabular'], stdin=ducttape.stdout, stdout=subprocess.PIPE)
csv = subprocess.check_output(["grep", "-o", "^[^#]*"], stdin=tabular.stdout).decode('ascii')

table = pd.read_csv(StringIO(csv), sep="\s+")
table['bleu_dev'] = pd.to_numeric(table['bleu_dev'], errors='coerce')
table['ent_dev'] = pd.to_numeric(table['ent_dev'], errors='coerce')

table['params'] = 2 * 3 * (4 * 512**2 + 2 * 512 * 4 * 512)

def format_num(num):
    if num > 1000 and num < 1000000:
        return f'{int(num / 1000)}K'
    if num > 1000000:
        return f'{int(num / 1000000)}M'

for lang in table['Lang'].unique():

    print(f'Lang: {lang}')
    rel = table[table['Lang'] == lang]

    # Joint param/data loss modeling
    def modeling_fn(data, a_D, log_D_C):
        return (math.exp(log_D_C) / data)**a_D

    (a_D, log_D_C), _ = scipy.optimize.curve_fit(modeling_fn, rel['data_bytes'], rel['ent_dev'], maxfev=5000, p0=[0.3, math.log(5e6)])

    print(f'Fit: {(a_D, log_D_C)}')

    # Data vs. Cross-Entropy
    fig, axes = plt.subplots()
    axes.set_xlabel("Data Size (Bytes)")
    axes.set_ylabel("Dev Cross Entropy")
    axes.set_title(f'Simulated Low-Resource Data Scaling {lang}')
    axes.set_xscale('log')
    axes.set_yscale('log')
    # axes.set_xlim((rel['data_bytes'].min() / 2, rel['data_bytes'].max() * 2))
    # axes.set_ylim((rel['ent_dev'].min() / 1.1, rel['ent_dev'].max() * 1.1))

    for params in sorted(table['params'].unique()):
        params_rel = rel[rel['params'] == params].sort_values('data_bytes')
        scatter = axes.scatter(params_rel['data_bytes'], params_rel['ent_dev'])
        predicted_line = modeling_fn(params_rel['data_bytes'], a_D, log_D_C)
        axes.plot(params_rel['data_bytes'], predicted_line, color=scatter.get_facecolor()[0], label="Predicted")

    actual_a_D, actual_log_DC = {'deen': (0.35, 13.43), 'ruen': (0.38, 13.81), 'zhen': (0.43, 12.73)}.get(lang)
    axes.plot(params_rel['data_bytes'], modeling_fn(params_rel['data_bytes'], actual_a_D, actual_log_DC), color='purple', label='Actual')

    axes.legend()
    fig.savefig(f'plots_out/{lang}_sim_lr_data_scaling.png')

    # Cross-Entropy vs. BLEU
    fig, axes = plt.subplots()
    axes.set_xlabel("Dev Cross Entropy")
    axes.set_ylabel("Dev BLEU")
    axes.set_xlim((rel['ent_dev'].max() + 1, 0))
    axes.set_title(f'Data Scaling Cross Entropy vs. BLEU')

    axes.scatter(rel['ent_dev'], rel['bleu_dev'])

    # Linear best fit line
    best_fit = scipy.stats.linregress(rel['ent_dev'], rel['bleu_dev'])
    predicted_bleu = rel['ent_dev'] * best_fit.slope + best_fit.intercept
    axes.plot(rel['ent_dev'], predicted_bleu, label=f'y = {best_fit.slope:.2f}x + {best_fit.intercept:.2f}')
    axes.legend()

    fig.savefig(f'plots_out/{lang}_sim_lr_data_scaling_bleu.png')
