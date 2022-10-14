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

fig_dir = 'data/figure/figure4'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

ratio_lst = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]
# ratio_lst = [0.03125, 0.0625, 0.125, 0.25]
csv_name = 'data/results/mixup.csv'
if not os.path.exists(csv_name) or os.path.exists(csv_name):
    dir_path = 'data/results/raw'
    rows = []
    orgs = [org for org in ['yeast', 'mouse', 'human']
            for i in range(len(ratio_lst))]
    # orgs[-6:] = [fn.replace('human', 'human_match')
    #              for fn in orgs[-6:]]

    nets = [['GeneMANIA']*len(ratio_lst)]*3
    nets = [i for g in nets for i in g]
    fakenets = [['GeneMANIA']*len(ratio_lst)]*3
    fakenets = [i for g in fakenets for i in g]
    org_nets = [f'{org}_{net}' for org, net in zip(orgs, fakenets)]

    methods = ratio_lst * 3

    file_names = []
    for org, ori_ndim, ndim in [['yeast', 200, 200],
                                ['mouse', 400, 400], ['human', 400, 400]]:
        for ratio in ratio_lst:
            file_names.extend([
                f'gemini_{org}_GeneMANIA_{ndim}_NN_Qsm41_separate35_ap_' +
                f'weight1_0.5_network_mixup5_{ratio}_gamma0.5_seed0_{ndim}',
            ])

    file_names = [f'{dir_path}/{file_name}' for file_name in file_names]
    rows = []
    for i in tqdm(range(len(orgs))):
        org = orgs[i]
        net = nets[i]
        file_name = file_names[i]
        method = methods[i]
        org_net = org_nets[i]

        test_split = True
        try:
            for test in range(5):
                file_name = file_name.replace(f'seed{test}', f'seed{test+1}')

                if test_split:
                    mi_auprcs = textread(file_name+'_None_auprcs.txt')
                    mi_auprcs = [float(i) for i in mi_auprcs]
                    mi_auprcs = [np.array(mi_auprcs).mean()]*5
                    ma_auprcs = textread(file_name+'_None_roc.txt')
                    ma_auprcs = [float(i) for i in ma_auprcs]
                    ma_auprcs = [np.array(ma_auprcs).mean()]*5
                    max_f1s = textread(file_name+'_None_f1.txt')
                    max_f1s = [float(i) for i in max_f1s]
                    max_f1s = [np.array(max_f1s).mean()]*5
                    max_f1 = max_f1s[test]
                    rows.append(
                        [org, net, method, 'f1', max_f1, test])
                    mi_auprc = mi_auprcs[test]
                    rows.append(
                        [org, net, method, 'micro_auprc', mi_auprc, test])
                    ma_auprc = ma_auprcs[test]
                    rows.append(
                        [org, net, method, 'macro_auprc', ma_auprc, test])
                else:
                    # load pred
                    pred = np.load(file_name+'_pred.npy')

                    # load labels
                    labels = np.load(file_name+'_labels.npy')

                    # load ids
                    ids = np.load(file_name+'_testids.npy')

                    pred_ = pred
                    label_ = labels

                    # mi_aurpc = roc_auc_score(label_, pred_, average='micro')
                    # Use AUC function to calculate the area under
                    # the curve of precision recall curve
                    from sklearn.metrics import auc, precision_recall_curve
                    precision, recall, thresholds = precision_recall_curve(
                        label_.flatten(), pred_.flatten())
                    f1_scores = 2*recall*precision/(recall+precision+1e-6)
                    max_f1 = np.max(f1_scores)
                    rows.append(
                        [org, net, method, 'f1', max_f1, test])

                    mi_auprc = auc(recall, precision)
                    rows.append(
                        [org, net, method, 'micro_auprc', mi_auprc, test])

        except FileNotFoundError:
            pass

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

df['select'] = df.method
net = 'GeneMANIA'

net_num = {'mouse': 403,
           'yeast': 505,
           'human': 895}
colors = \
    [
        # '#8c510a',
        '#d8b365',
        # '#f6e8c3',
        '#f5f5f5',
        # '#c7eae5',
        '#5ab4ac',
        # '#01665e',
    ][::-1]
orgs = ['mouse', 'human', 'yeast']
orgs_display = ['Mouse', 'Human', 'Yeast']

for idx, metric in enumerate(['micro_auprc']):
    ax = plot_settings.get_wider_axis(FIG_WIDTH=4, FIG_HEIGHT=4)
    select_models = ratio_lst
    labels = [
        f'Yeast {int(505*ratio)}, Mouse {int(403*ratio)}' for ratio
        in ratio_lst]

    filt = (df.metric == metric)
    dfs = df[filt]

    idx_ = 0
    for org in orgs:
        stderrs_ = []
        means_ = []
        for select in select_models:
            filt = (dfs['select'] == select) & (dfs.org == org)
            df_ = dfs[filt]
            means_.append(df_.value.mean())
            stderrs_.append(df_.value.std()/np.sqrt(len(df_)))
        means = np.array(means_)
        stderrs = np.array(stderrs_)
        min_val = plot_utils.out_min_val(means, 0.28)
        plot_utils.line_plot(ax, means, xlabel='',
                             ylabel=metric2display[metric],
                             xdata=[net_num[org]*ra for ra in ratio_lst],
                             max_time=None, invert_axes=False,
                             color=colors[idx_],
                             linestyle='solid', label_marker=None, linewidth=1,
                             alpha=0.3, std=stderrs)

        org2ndim = {'yeast': 200, 'mouse': 400, 'human': 400}
        metric_dic = {}
        file_name = f"gemini_{org}_GeneMANIA_800_NN_{org2ndim[org]}"
        file_name = f'{dir_path}/{file_name}'

        mi_auprcs = textread(file_name+'_None_auprcs.txt')
        mi_auprcs = [float(i) for i in mi_auprcs]
        metric_dic['micro_auprc'] = np.array(mi_auprcs).mean()

        ma_auprcs = textread(file_name+'_None_roc.txt')
        ma_auprcs = [float(i) for i in ma_auprcs]
        metric_dic['macro_auprc'] = np.array(ma_auprcs).mean()

        max_f1s = textread(file_name+'_None_f1.txt')
        max_f1s = [float(i) for i in max_f1s]
        metric_dic['f1'] = np.array(max_f1s).mean()

        ax.plot(net_num[org], metric_dic[metric], c=colors[idx_], marker='*')
        idx_ += 1

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    yticks = ax.get_yticks()
    if len(yticks) > 6:
        ll = [i for i in range(0, len(yticks), 2)]
        # if org == 'yeast' and metric == 'macro_auprc':
        #     ll = [1, 2, 3, 4, 5]
        net_yticks = yticks[ll]
        ax.set_yticks(net_yticks)
    # ax.set_xscale('log')
    plot_utils.format_ax(ax)
    # plot_utils.format_legend(ax, *ax.get_legend_handles_labels())
    # plot_utils.format_legend(ax, *ax.get_legend_handles_labels(),
    #                          loc='upper left',
    #                          ncols=2)
    plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/figure4_{metric}.pdf')
    # ax.set_xscale('log')
    # plt.savefig(f'{fig_dir}/figure4_{metric}_log.pdf')
    plt.close()
