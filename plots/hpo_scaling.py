import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import matplotlib.ticker as mticker
import scipy.optimize
import numpy as np


def modeling_fn(data, a_N, log_N_C):
    return (math.exp(log_N_C) / data)**a_N


try: os.makedirs(f'plots_out')
except FileExistsError: pass

lang = 'ja-en'

evals = pd.read_csv(f'hpo_nmt/datasets/{lang}.evals', sep="\t", names=["dev_bleu", "dev_gpu_time", "dev_ppl", "num_updates", "gpu_memory", "num_param"])
hyps = pd.read_csv(f'hpo_nmt/datasets/{lang}.hyps', sep="\t", names=["bpe_symbols", "num_layers", "num_embed", "transformer_feed_forward_num_hidden", "transformer_attention_heads", "initial_learning_rate"])
data = pd.concat([evals, hyps], axis=1)
data['cross_ent'] = data['dev_ppl'].map(math.log)
data['non_embed_params'] = 2 * data['num_layers'] * (4 * data['num_embed']**2 + 2 * data['num_embed'] * data['transformer_feed_forward_num_hidden'])

fig, axes = plt.subplots()
axes.set_xlabel("# Non-Embedding Parameters")
axes.set_ylabel("Dev Cross-Entropy")
axes.set_xscale('log')
axes.set_yscale('log')
axes.set_ylim((2, 3.5))
axes.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))

data = data[data['initial_learning_rate'] != 0.001]
data = data[data['bpe_symbols'] != 30000]
for bpe in data['bpe_symbols'].unique():
    for layers in data['num_layers'].unique():
        rel = data[(data['bpe_symbols'] == bpe) & (data['num_layers'] == layers)].sort_values('non_embed_params')
        (a_N, log_N_C), _ = scipy.optimize.curve_fit(modeling_fn, rel['non_embed_params'], rel['cross_ent'], maxfev=5000, p0=[0.3, math.log(5e6)])
        scatter = axes.scatter(rel['non_embed_params'], rel['cross_ent'], label=f'{int(layers * 2)} Layers, {int(bpe / 1000)}k BPE')
        params = np.linspace(data['non_embed_params'].min(), data['non_embed_params'].max())
        axes.plot(params, modeling_fn(params, a_N, log_N_C), linestyle=":", color=scatter.get_facecolor()[0])

axes.legend()
fig.savefig(f'plots_out/hpo_scaling.png', bbox_inches='tight')
