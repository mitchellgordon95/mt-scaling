import pandas as pd
from io import StringIO
import os
import scipy.optimize
import scipy.stats
import math
import sys
from main import read_file, ducttape_summary, joint_modeling_fn


def data_scaling_stability(table):
    table = table[table['data_bytes'] > 5242880]
    for lang in table['Lang'].unique():

        print(f'Lang: {lang}')
        rel = table[table['Lang'] == lang]

        (a_N, log_N_C, a_D, log_D_C), _ = scipy.optimize.curve_fit(joint_modeling_fn, (rel['data_bytes'], rel['params']), rel['ent_dev'], maxfev=5000, p0=[0.071, math.log(8.8e10), 0.3, math.log(5e6)])

        print(f'Fit: {(a_N, log_N_C, a_D, log_D_C)}')
        for max_data_percent in reversed(sorted(rel['DataPercent'].unique())):
            if max_data_percent < 3.125:
                break
            rel = rel[rel['DataPercent'] <= max_data_percent]
            (a_Np, log_N_Cp, a_Dp, log_D_Cp), _ = scipy.optimize.curve_fit(joint_modeling_fn, (rel['data_bytes'], rel['params']), rel['ent_dev'], maxfev=5000, p0=[0.071, math.log(8.8e10), 0.3, math.log(5e6)])
            print(f'& {max_data_percent} & {abs(a_N - a_Np):.3f} & {abs(a_D - a_Dp):.3f} \\\\')

if __name__ == "__main__":
    try: os.mkdir('plots_out')
    except FileExistsError: pass

    csv = read_file(sys.argv[1]) if len(sys.argv) > 1 else ducttape_summary('data_scaling')

    table = pd.read_csv(StringIO(csv), sep="\s+")
    table['bleu_dev'] = pd.to_numeric(table['bleu_dev'], errors='coerce')
    table['ent_dev'] = pd.to_numeric(table['ent_dev'], errors='coerce')
    table['params'] = 2 * table['Layers'] * (4 * table['ModelSize']**2 + 2 * table['ModelSize'] * table['FeedForward'])

    print(f'Missing values: \n {table[table.isnull().any(axis=1)]}')
    table = table.dropna()

    data_scaling_stability(table)
