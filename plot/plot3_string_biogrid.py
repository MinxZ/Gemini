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

fig_dir = 'data/figure/figure3'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

csv_name = 'data/results/summary3.csv'
if not os.path.exists(csv_name):
    dir_path = 'data/results/raw'
    rows = []
    orgs = [org for org in ['yeast', 'mouse', 'human'] for i in range(6)]
    orgs[-6:] = [fn.replace('human', 'human_match')
                 for fn in orgs[-6:]]

    nets = [['GOA'] * 2 + ['GeneMANIA']*2 + ['GOA+GeneMANIA']*2]*3
    nets = [i for g in nets for i in g]
    fakenets = [['mashup_ex']
                * 2 + ['GeneMANIA_ex']*2 + ['mashup_GeneMANIA_ex']*2]*3
    fakenets = [i for g in fakenets for i in g]
    org_nets = [f'{org}_{net}' for org, net in zip(orgs, fakenets)]

    methods = ['mashup', 'ours'] * 3
    methods = methods*3

    file_names = []
    for org, ori_ndim, ndim in [['yeast', 800, 200],
                                ['mouse', 800, 400], ['human', 800, 400]]:
        file_names.extend([
            f'gemini_{org}_mashup_ex_{ori_ndim}_NN_{ndim}',

            f'gemini_{org}_mashup_ex_{ndim}_NN_Qsm41_separate35_ap_' +
            f'weight1_0.5_network_mixup1_1.0_gamma0.5_seed1_{ndim}',

            f'gemini_{org}_GeneMANIA_ex_{ori_ndim}_NN_{ndim}',

            f'gemini_{org}_GeneMANIA_ex_{ndim}_NN_Qsm41_separate35_ap_' +
            f'weight1_0.5_network_mixup1_1.0_gamma0.5_seed1_{ndim}',

            f'gemini_{org}_mashup_GeneMANIA_ex_{ori_ndim}_NN_{ndim}',

            f'gemini_{org}_mashup_GeneMANIA_ex_{ndim}_NN_Qsm41_separate35_ap_' +
            f'weight1_0.5_network_mixup1_1.0_gamma0.5_seed1_{ndim}',
        ])
    file_names[-6:] = [fn.replace('human', 'human_match')
                       for fn in file_names[-6:]]
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
    df.loc[df.org == 'human_match', 'org'] = 'human'
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

for idx, metric in enumerate(['f1', 'micro_auprc', 'macro_auprc']):
    means = []
    stderrs = []
    ax = plot_settings.get_wider_axis(FIG_WIDTH=4, FIG_HEIGHT=4)
    # model_ordering = ['mashup', 'average_mashup', 'ours']
    model_ordering = ['mashup GOA',
                      'mashup GeneMANIA',
                      'mashup GOA+GeneMANIA',
                      'ours GOA',
                      'ours GeneMANIA',
                      'ours GOA+GeneMANIA'
                      ]

    filt = (df.metric == metric)
    dfs = df[filt]
    colors = \
        [
            '#8c510a',
            '#d8b365',
            '#f6e8c3',
            '#c7eae5',
            '#5ab4ac',
            '#01665e',
        ]
    labels = ['Mashup\n(STRING)', 'Mashup\n(BioGRID)',
              'Mashup\n(STRING+BioGRID)',
              'Gemini\n(STRING)', 'Gemini\n(BioGRID)',
              'Gemini\n(STRING+BioGRID)']

    orgs = ['mouse', 'human', 'yeast']
    orgs_display = ['Mouse', 'Human', 'Yeast']
    means = []
    stderrs = []
    for org in orgs:
        stderrs_ = []
        means_ = []
        for method_net in model_ordering:
            filt = (dfs.method_net == method_net) & (dfs.org == org)
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
    min_val = plot_utils.out_min_val(means, 0.28)
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
    plt.savefig(f'{fig_dir}/figure3_{metric}.pdf')
    plt.close()
