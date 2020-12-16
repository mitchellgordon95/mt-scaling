import matplotlib.pyplot as plt
import subprocess
import pandas as pd
from io import StringIO
import os

try: os.mkdir('plots_out')
except FileExistsError: pass

ducttape = subprocess.Popen(["ducttape", "main.tape", "-C", "main.tconf", "-p", "data_scaling", "summary"], stdout=subprocess.PIPE)
tabular = subprocess.Popen(['tabular'], stdin=ducttape.stdout, stdout=subprocess.PIPE)
csv = subprocess.check_output(["grep", "-o", "^[^#]*"], stdin=tabular.stdout).decode('ascii')

table = pd.read_csv(StringIO(csv), sep="\s+")
table['bleu_dev'] = pd.to_numeric(table['bleu_dev'], errors='coerce')
table['ent_dev'] = pd.to_numeric(table['ent_dev'], errors='coerce')

table['line_count'] = 28300000 * table['DataPercent'] / 100

fig, axes = plt.subplots()
axes.set_xlabel("Data Size (Lines)")
axes.set_ylabel("Dev Cross Entropy")
axes.set_title(f'Data Scaling')
axes.set_xscale('log')
axes.set_yscale('log')
axes.set_xlim((table['line_count'].min() / 2, table['line_count'].max() * 2))
axes.set_ylim((table['ent_dev'].min() / 2, table['ent_dev'].max() * 2))

axes.scatter(table['line_count'], table['ent_dev'])

fig.savefig(f'plots_out/data_scaling.png')

fig, axes = plt.subplots()
axes.set_xlabel("Dev Cross Entropy")
axes.set_ylabel("Dev BLEU")
axes.set_xlim((table['ent_dev'].max() + 1, 0))
axes.set_title(f'Data Scaling Cross Entropy vs. BLEU')

axes.scatter(table['ent_dev'], table['bleu_dev'])

fig.savefig(f'plots_out/data_scaling_bleu.png')
