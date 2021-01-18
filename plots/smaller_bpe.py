import matplotlib.pyplot as plt
import subprocess
import pandas as pd
import numpy as np
from io import StringIO
import os
import scipy.optimize
import scipy.stats
import math
import sys
from main import format_num

try: os.mkdir('plots_out')
except FileExistsError: pass

if len(sys.argv) <= 1:
    ducttape = subprocess.Popen(["ducttape", "main.tape", "-C", "main.tconf", "-p", 'smaller_bpe', "summary"], stdout=subprocess.PIPE)
    tabular = subprocess.Popen(['tabular'], stdin=ducttape.stdout, stdout=subprocess.PIPE)
    csv = subprocess.check_output(["grep", "-o", "^[^#]*"], stdin=tabular.stdout).decode('ascii')
else:
    with open(sys.argv[1], "r")  as fd:
        csv = ""
        for line in fd.readlines():
            if line[0] != "#":
                csv += line

table = pd.read_csv(StringIO(csv), sep="\s+")
table['bleu_dev'] = pd.to_numeric(table['bleu_dev'], errors='coerce')
table['ent_dev'] = pd.to_numeric(table['ent_dev'], errors='coerce')

table['params'] = 2 * table['Layers'] * (4 * table['ModelSize']**2 + 2 * table['ModelSize'] * table['FeedForward'])

print(f'Missing values: \n {table[table.isnull().any(axis=1)]}')
table = table.dropna()

for lang in ['deen']: #table['Lang'].unique():

    # print(f'Lang: {lang}')
    # rel = table[table['Lang'] == lang]
    rel = table[table['data_bytes'] > 5242880]

    # Joint param/data loss modeling
    def joint_modeling_fn(lines_params, a_N, log_N_C, a_D, log_D_C):
        lines, params = lines_params
        return ((math.exp(log_N_C) / params)**(a_N / a_D) + math.exp(log_D_C) / lines)**a_D

    (a_N, log_N_C, a_D, log_D_C), _ = scipy.optimize.curve_fit(joint_modeling_fn, (rel['data_bytes'], rel['params']), rel['ent_dev'], maxfev=5000, p0=[0.071, math.log(8.8e10), 0.3, math.log(5e6)])

    print(f'Fit: {(a_N, log_N_C, a_D, log_D_C)}')
    # rel = table[table['Lang'] == lang]
    rel = table

    # Data vs. Cross-Entropy
    fig, axes = plt.subplots()
    axes.set_xlabel("Data Size (Bytes)")
    axes.set_ylabel("Dev Cross Entropy")
    axes.set_title(f'Data Scaling {lang}')
    axes.set_xscale('log')
    axes.set_yscale('log')
    # axes.set_xlim((rel['data_bytes'].min() / 2, rel['data_bytes'].max() * 2))
    # axes.set_ylim((rel['ent_dev'].min() / 1.1, rel['ent_dev'].max() * 1.1))

    for params in sorted(table['params'].unique()):
        params_rel = rel[rel['params'] == params].sort_values('data_bytes')
        scatter = axes.scatter(params_rel['data_bytes'], params_rel['ent_dev'], label=f'{format_num(params)} Params')
        predicted_line = joint_modeling_fn((params_rel['data_bytes'], params_rel['params']), a_N, log_N_C, a_D, log_D_C)
        axes.plot(params_rel['data_bytes'], predicted_line, color=scatter.get_facecolor()[0])

    # Predicted results for 500M params
    # predicted_line = joint_modeling_fn((params_rel['data_bytes'], np.array([500000000.0] * params_rel['data_bytes'].size)), a_N, log_N_C, a_D, log_D_C)
    # axes.plot(params_rel['data_bytes'], predicted_line, label="500M Params")

    # Infinite params line
    inf_params = np.array([float('inf')] * params_rel['data_bytes'].size)
    inf_params_line = joint_modeling_fn((params_rel['data_bytes'], inf_params), a_N, log_N_C, a_D, log_D_C)
    print(inf_params_line)
    # axes.plot(params_rel['data_bytes'], inf_params_line, color='purple', label='Inf Params')
    axes.axvline(5242880, linestyle='--', label='5 MB')

    axes.legend()
    fig.savefig(f'plots_out/{lang}_bpe.2000_data_scaling.png')
    fig, axes = plt.subplots()
    axes.set_xlabel("Dev Cross Entropy")
    axes.set_ylabel("Dev BLEU")
    axes.set_xlim((rel['ent_dev'].max() + 1, 0))
    axes.set_title(f'Data Scaling Cross Entropy vs. BLEU')

    axes.scatter(rel['ent_dev'], rel['bleu_dev'])

    # Exp decay fit
    def exp_decay(ent, C, k):
        return C * np.exp(k * ent)
    (C, k), _ = scipy.optimize.curve_fit(exp_decay, rel['ent_dev'], rel['bleu_dev'], maxfev=5000, p0=[1, -0.5])
    axes.plot(rel['ent_dev'].sort_values(), exp_decay(rel['ent_dev'].sort_values(), C, k), label=f'y = {C:.2f} e ^ ({k:.2f} x)')

    axes.legend()
    fig.savefig(f'plots_out/{lang}_bpe.2000_data_scaling_bleu.png')
