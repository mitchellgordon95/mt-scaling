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

axes.scatter(table['line_count'], table['ent_dev'])

fig.savefig(f'plots_out/data_scaling.png')
