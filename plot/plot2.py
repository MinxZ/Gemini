import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import KFold
from tqdm import tqdm

import plot_settings
import plot_utils
from func import textread

# sns.set(style="ticks", context="talk")
# plt.style.use("dark_background")

fig_dir = 'data/figure/fig2'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

csv_name = 'data/results/summary2.csv'
if not os.path.exists(csv_name):
    def textread_2(filename):
        output1 = []
        output2 = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                output1.append(line.strip().split()[0])
                output2.append(line.strip().split()[1])
        return output1, output2

    # load map_go_id_ont
    naming = {'molecular_function': 'MF',
              'biological_process': 'BP',
              'cellular_component': 'CC'}
    goa, go_map = textread_2('data/map_go_id_ont.txt')
    go_map = [naming[cat] for cat in go_map]

    # mapping go to ont
    mapping = {'MF': [],
               'BP': [],
               'CC': []}
    for go, map in zip(goa, go_map):
        mapping[map].append(go)

    dir_path = 'data/results/raw'
    rows = []

    # file name
    # ori_ndim = 200
    # ndim = 200
    # method = 'ours'
    # flag = 'Qsm41_separate35_ap_weight1_0.5_network_mixup5_1.0_gamma0.5'
    # file_name = f'{}/gemini_{org_net}_{ori_ndim}_NN_{flag}_{ndim}'

    methods = ['mashup', 'average_mashup',
               'average_avd', 'average_pca', 'ours']
    orgs = [org for org in ['yeast', 'mouse', 'human']
            for i in range(len(methods))]

    nets = [['GeneMANIA']*len(methods)]*3
    nets = [i for g in nets for i in g]
    fakenets = [['GeneMANIA']*len(methods)]*3
    fakenets = [i for g in fakenets for i in g]
    org_nets = [f'{org}_{net}' for org, net in zip(orgs, fakenets)]

    methods = methods*3

    file_names = []
    for org, ori_ndim, ndim in [['yeast', 800, 200],
                                ['mouse', 800, 400], ['human', 800, 400]]:
        file_names.extend([
            f'gemini_{org}_GeneMANIA_{ori_ndim}_NN_{ndim}',

            f'gemini_{org}_GeneMANIA_{ori_ndim}_NN_average_{ndim}',

            f'gemini_{org}_GeneMANIA_{ori_ndim}_NN_svd_{ndim}',
            f'gemini_{org}_GeneMANIA_{ori_ndim}_NN_pca_{ndim}',

            f'gemini_{org}_GeneMANIA_{ndim}_NN_Qsm41_separate35_ap_' +
            f'weight1_0.5_network_mixup1_1.0_gamma0.5_seed1_{ndim}',

            # f'gemini_{org}_mashup_ex_{ori_ndim}_NN_{ndim}',

            # f'gemini_{org}_mashup_ex_{ndim}_NN_Qsm41_separate35_ap_' +
            # f'weight1_0.5_network_mixup1_1.0_gamma0.5_seed1_{ndim}',

            # f'gemini_{org}_mashup_GeneMANIA_ex_{ori_ndim}_NN_{ndim}',

            # f'gemini_{org}_mashup_GeneMANIA_ex_{ndim}_NN_Qsm41_separate35_ap_' +
            # f'weight1_0.5_network_mixup1_1.0_gamma0.5_seed1_{ndim}',
        ])
    # file_names[-4:] = [fn.replace('human', 'human_match')
    #                    for fn in file_names[-4:]]

    file_names = [f'{dir_path}/{file_name}' for file_name in file_names]

    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    rows = []
    for i in tqdm(range(len(orgs))):
        org = orgs[i]
        net = nets[i]
        file_name = file_names[i]
        method = methods[i]
        org_net = org_nets[i]

        # load pred
        pred = np.load(file_name+'_pred.npy')

        # load labels
        labels = np.load(file_name+'_labels.npy')

        # load ids
        ids = np.load(file_name+'_testids.npy')

        # load go terms
        go_terms_path = f'data/annotations/{org}/{org}_go_terms.txt'
        go_terms = textread(go_terms_path)

        train_test_ids = kf.split(range(len(pred)))
        test_num = []
        for tr, te in train_test_ids:
            test_num.append(len(te))
        test_div = np.cumsum(test_num)
        test_div = [0] + list(test_div)
        for cat in ['MF', 'BP', 'CC']:
            gos = mapping[cat]
            # print(len(gos))
            # print(len(set(go_terms).intersection(set(gos))))
            filt = np.isin(go_terms, gos)
            for test in range(5):
                pred_ = pred[test_div[test]:test_div[test+1], filt]
                label_ = labels[test_div[test]:test_div[test+1], filt]

                # mi_aurpc = roc_auc_score(label_, pred_, average='micro')
                # Use AUC function to calculate the area under
                # the curve of precision recall curve

                precision, recall, thresholds = precision_recall_curve(
                    label_.flatten(), pred_.flatten())
                f1_scores = 2*recall*precision/(recall+precision+1e-6)
                max_f1 = np.max(f1_scores)
                rows.append(
                    [org, net, method, cat, 'f1', max_f1, test])

                mi_auprc = auc(recall, precision)
                rows.append(
                    [org, net, method, cat, 'micro_auprc', mi_auprc, test])

                auprc = 0
                class_not_zero = 0
                for i in range(pred_.shape[1]):
                    if label_[:, i].sum() != 0:
                        precision, recall, thresholds = precision_recall_curve(
                            label_[:, i], pred_[:, i])
                        auprc += auc(recall, precision)
                        class_not_zero += 1
                auprc /= class_not_zero
                ma_auprc = auprc
                rows.append(
                    [org, net, method, cat, 'macro_auprc', ma_auprc, test])
    # load all resutls
    col_name = ['org', 'net', 'method', 'group',
                'metric', 'value', 'test']
    df = pd.DataFrame(rows)
    df.columns = col_name
    df.loc[df.org == 'human_match', 'org'] = 'human'
    df.to_csv(csv_name)
else:
    df = pd.read_csv(csv_name, index_col=0)

    # sns.set(style="ticks", context="talk")
    # plt.style.use("dark_background")

    # figure 2
    # net = 'GeneMANIA'
    # for org in ['yeast', 'mouse', 'human']:
    #     # for org in ['yeast']:
    #     for metric in ['f1', 'micro_auprc', 'macro_auprc']:
    #         filt = (df.net == net) & (df.org == org) & (df.metric == metric)
    #         dfs = df[filt]
    #         ax = sns.barplot(x='group', y='value',
    #                          hue='method',
    #                          data=dfs, palette="crest")
    #         fig = ax.get_figure()
    #         fig.savefig(f'data/send5/{org}_{metric}.png')
    #         plt.close()


def get_model_colors(mod):
    color = \
        [
            '#d8b365',
            '#f5f5f5',
            '#5ab4ac'
        ]
    # return {
    #     'mashup': '#e0f3db',
    #     'average_mashup': '#a8ddb5',
    #     'ours': '#43a2ca',
    # }[mod]
    return color


metric2display = {'f1': 'maximum' + r' F$_1$',
                  'micro_auprc': 'micro-AUPRC',
                  'macro_auprc': 'macro-AUPRC'}
model_ordering2display = {'mashup': 'Mahsup',
                          'average_mashup': 'Average Mashup',
                          'average_avd': 'Average SVD',
                          'average_pca': 'Average PCA',
                          'ours': 'Gemini'}
# figure 2
net = 'GeneMANIA'
for org in ['human', 'yeast', 'mouse']:
    for metric in ['f1', 'micro_auprc', 'macro_auprc']:
        means = []
        stderrs = []
        ax = plot_settings.get_wider_axis()
        model_ordering = ['average_avd', 'average_pca',  'average_mashup',
                          'mashup', 'ours']
        # model_ordering = ['average_pca', 'average_avd', 'mashup',  'ours']
        # model_ordering = ['mashup', 'ours']

        filt = (df.net == net) & (df.org == org) & (df.metric == metric)
        dfs = df[filt]

        # colors = [get_model_colors(
        #     mod) for mod in model_ordering]
        colors = \
            [
                '#a6611a',
                '#dfc27d',
                '#f5f5f5',
                '#80cdc1',
                '#018571']
        labels = [model_ordering2display[m] for m in model_ordering]

        # groups = ['MF', 'CC', 'BP']
        groups = ['BP', 'MF', 'CC']
        means = []
        stderrs = []
        for group in groups:
            stderrs_ = []
            means_ = []
            for method in model_ordering:
                filt = (dfs.method == method) & (dfs.group == group)
                df_ = dfs[filt]
                means_.append(df_.value.mean())
                stderrs_.append(df_.value.std()/np.sqrt(len(df_)))
            means.append(means_)
            stderrs.append(stderrs_)

        min_val = plot_utils.out_min_val(means, 1/4)

        plot_utils.grouped_barplot(
            ax, means,
            groups,
            xlabel='',
            ylabel=metric2display[metric],
            color_legend=labels,
            nested_color=colors, nested_errs=stderrs, tickloc_top=False,
            rotangle=0, anchorpoint='right',
            legend_loc='upper left',
            min_val=min_val,
            fontsize=20)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        yticks = ax.get_yticks()
        if len(yticks) > 6:
            ll = [i for i in range(0, len(yticks), 2)]
            if org == 'yeast' and metric == 'macro_auprc':
                ll = [0, 1, 2, 3, 4, 5]
            net_yticks = yticks[ll]
            ax.set_yticks(net_yticks)
        if metric == 'f1' and org == 'human':
            ax.set_yticks([0.35 + 0.05*i for i in range(6)])

        ax.tick_params(axis='both', which='major', labelsize=15)
        plot_utils.format_ax(ax)
        # plot_utils.format_legend(ax, *ax.get_legend_handles_labels())
        # # plot_utils.format_legend(ax, *ax.get_legend_handles_labels(),
        # #                          loc='upper left',
        # #                          ncols=2)
        # plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/figure2_{org}_{metric}.pdf')
        plt.close()
