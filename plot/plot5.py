import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

import plot_settings
import plot_utils
from func import textread

# sns.set(style="ticks", context="talk")
# plt.style.use("dark_background")

fig_dir = 'data/figure/figure5'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

csv_name = 'data/results/summary5.csv'
if not os.path.exists(csv_name) or os.path.exists(csv_name):
    dir_path = 'data/results/raw'
    rows = []

    # file name
    # ori_ndim = 200
    # ndim = 200
    # method = 'ours'
    # flag = 'Qsm41_separate35_ap_weight1_0.5_network_mixup5_1.0_gamma0.5'
    # file_name = f'{}/gemini_{org_net}_{ori_ndim}_NN_{flag}_{ndim}'

    methods = ['mashup', 'mean', 'variance', 'skewness', 'kurtosis']
    len_methods = len(methods)
    orgs = [org for org in ['yeast', 'mouse', 'human']
            for i in range(len_methods)]
    nets = [['GeneMANIA']*len_methods]*3
    nets = [i for g in nets for i in g]
    fakenets = [['GeneMANIA']*len_methods]*3
    fakenets = [i for g in fakenets for i in g]
    org_nets = [f'{org}_{net}' for org, net in zip(orgs, fakenets)]

    methods = methods*3

    file_names = []
    for org, ori_ndim, ndim in [['yeast', 800, 200],
                                ['mouse', 800, 400], ['human', 800, 400]]:
        file_names.extend([
            f'gemini_{org}_GeneMANIA_{ori_ndim}_NN_{ndim}',

            f'gemini_{org}_GeneMANIA_{ndim}_NN_Qm11_separate35_ap_' +
            f'weight1_0.5_network_mixup1_1.0_gamma0.5_seed1_{ndim}',

            f'gemini_{org}_GeneMANIA_{ndim}_NN_Qm21_separate35_ap_' +
            f'weight1_0.5_network_mixup1_1.0_gamma0.5_seed1_{ndim}',

            f'gemini_{org}_GeneMANIA_{ndim}_NN_Qsm31_separate35_ap_' +
            f'weight1_0.5_network_mixup1_1.0_gamma0.5_seed1_{ndim}',

            f'gemini_{org}_GeneMANIA_{ndim}_NN_Qsm41_separate35_ap_' +
            f'weight1_0.5_network_mixup1_1.0_gamma0.5_seed1_{ndim}',
        ])
    file_names = [f'{dir_path}/{file_name}' for file_name in file_names]

    rows = []
    for i in tqdm(range(len(orgs))):
        org = orgs[i]
        net = nets[i]
        file_name = file_names[i]
        method = methods[i]
        org_net = org_nets[i]

        mi_auprcs = textread(file_name+'_None_auprcs.txt')
        mi_auprcs = [float(i) for i in mi_auprcs]
        ma_auprcs = textread(file_name+'_None_roc.txt')
        ma_auprcs = [float(i) for i in ma_auprcs]
        max_f1s = textread(file_name+'_None_f1.txt')
        max_f1s = [float(i) for i in max_f1s]

        for test in range(5):
            max_f1 = max_f1s[test]
            rows.append(
                [org, net, method, 'f1', max_f1, test])

            mi_auprc = mi_auprcs[test]
            rows.append(
                [org, net, method, 'micro_auprc', mi_auprc, test])

            ma_auprc = ma_auprcs[test]
            rows.append(
                [org, net, method, 'macro_auprc', ma_auprc, test])
    # load all resutls
    col_name = ['org', 'net', 'method',
                'metric', 'value', 'test']
    df = pd.DataFrame(rows)
    df.columns = col_name
    df.to_csv(csv_name)
else:
    df = pd.read_csv(csv_name, index_col=0)

metric2display = {'f1': 'maximum' + r' F$_1$',
                  'micro_auprc': 'micro-AUPRC',
                  'macro_auprc': 'macro-AUPRC'}

df['method_net'] = df.method + ' ' + df.net
# figure 2
net = 'GeneMANIA'

# plot_settings.get_fig(FIG_WIDTH=10, FIG_HEIGHT=4)

for idx, metric in enumerate(['macro_auprc', 'f1', 'micro_auprc']):
    means = []
    stderrs = []
    ax = plot_settings.get_wider_axis(FIG_WIDTH=4, FIG_HEIGHT=4)

    filt = (df.metric == metric)
    dfs = df[filt]

    colors = \
        ['#8c510a',
         '#dfc27d',
         '#f5f5f5',
         '#80cdc1',
         '#018571']
    # colors = ['#a6611a',
    #           '#dfc27d',
    #           '#80cdc1',
    #           '#018571', ]
    # colors = ['#dfc27d',
    #           '#f5f5f5',
    #           '#80cdc1',
    #           '#018571', ]
    # labels = ['Mashup\n(STRING)', 'Mashup\n(STRING+BioGRID)',
    #           'Gemini\n(STRING)', 'Gemini\n(STRING+BioGRID)']
    labels = [method.capitalize() for method in methods[:len_methods]]

    orgs = ['mouse', 'human', 'yeast']
    orgs_display = ['Mouse', 'Human', 'Yeast']
    means = []
    stderrs = []
    for org in orgs:
        stderrs_ = []
        means_ = []
        for method in methods[:len_methods]:
            filt = (dfs.method == method) & (dfs.org == org)
            df_ = dfs[filt]
            means_.append(df_.value.mean())
            stderrs_.append(df_.value.std()/np.sqrt(len(df_)))
        means.append(means_)
        stderrs.append(stderrs_)

    # if metric == 'f1':
    #     min_val = 0.3
    # elif 'micro' in metric:
    #     min_val = 0.2
    # else:
    #     min_val = 0.0
    min_val = plot_utils.out_min_val(means, 0.25)
    plot_utils.grouped_barplot(
        ax, means,
        orgs_display,
        xlabel='', ylabel=metric2display[metric],
        color_legend=labels,
        nested_color=colors, nested_errs=stderrs, tickloc_top=False,
        rotangle=0, anchorpoint='right',
        legend_loc='upper left',
        min_val=min_val,
        fontsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    yticks = ax.get_yticks()
    if len(yticks) > 6:
        ll = [i for i in range(0, len(yticks), 2)]
        # if org == 'yeast' and metric == 'macro_auprc':
        #     ll = [1, 2, 3, 4, 5]
        net_yticks = yticks[ll]
        ax.set_yticks(net_yticks)
    if metric == 'micro_auprc':
        ax.set_yticks(np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))

    ax.tick_params(axis='both', which='major', labelsize=14)
    plot_utils.format_ax(ax)
    # plot_utils.format_legend(ax, *ax.get_legend_handles_labels())
    # # plot_utils.format_legend(ax, *ax.get_legend_handles_labels(),
    # #                          loc='upper left',
    # #                          ncols=2)
    # plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/figure5_{metric}.pdf')
    plt.close()
