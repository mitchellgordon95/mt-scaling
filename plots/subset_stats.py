import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import os
import sys
from main import read_file, ducttape_summary

try: os.mkdir('plots_out')
except FileExistsError: pass

csv = read_file(sys.argv[1]) if len(sys.argv) > 1 else ducttape_summary('subset_stats')
table = pd.read_csv(StringIO(csv), sep="\s+")
table['line_count'] = pd.to_numeric(table['line_count'], errors='coerce')
table['data_bytes'] = pd.to_numeric(table['data_bytes'], errors='coerce')

fig, axes = plt.subplots()
axes.set_xlabel("Lines")
axes.set_ylabel("Bytes")
axes.set_title(f'Lines vs. Bytes')

axes.scatter(table['line_count'], table['data_bytes'])

fig.savefig(f'plots_out/lines_vs_bytes.png')
