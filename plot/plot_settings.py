import matplotlib as mpl
import matplotlib.pyplot as plt

# set basic parameters
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams.update({"ytick.color": "w",
                     "xtick.color": "w",
                     "axes.labelcolor": "w",
                     "axes.edgecolor": "w"})

MEDIUM_SIZE = 14
SMALLER_SIZE = 12
plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)	 # fontsize of the axes title
plt.rc('xtick', labelsize=SMALLER_SIZE)	 # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)	 # fontsize of the tick labels
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('font', family='Helvetica')
# , xtick.color='w', axes.labelcolor='w', axes.edge_color='w'
FIG_HEIGHT = 4
FIG_WIDTH = 4
plt.style.use('dark_background')


EVO_DEVO_MODELS = ['Sagittarius', 'linear', 'mean', 'mTAN', 'cvae', 'cpa',
                   'seq_by_seq_RNN', 'seq_by_seq_neuralODE']


def get_square_axis():
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax = plt.subplot(1, 1, 1)
    return ax


# def get_wider_axis(double=False):
#     plt.figure(figsize=(int(FIG_WIDTH * (3/2 if not double
#                         else 5/2)), FIG_HEIGHT))
#     ax = plt.subplot(1, 1, 1)
#     return ax


def get_wider_axis(idx=1,
                   horizon_figure_num=1, vertical_figure_num=1,
                   FIG_WIDTH=4, FIG_HEIGHT=4):
    get_fig(FIG_WIDTH, FIG_HEIGHT)
    ax = get_ax(idx, horizon_figure_num, vertical_figure_num)
    return ax


def get_fig(FIG_WIDTH=4, FIG_HEIGHT=4):
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))


def get_ax(idx=1, horizon_figure_num=1, vertical_figure_num=1):
    ax = plt.subplot(vertical_figure_num, horizon_figure_num, idx)
    return ax


# def get_wider_axis7():
#     plt.figure(figsize=(int(FIG_WIDTH * (7/4)), FIG_HEIGHT))
#     ax = plt.subplot(1, 1, 1)
#     return ax


def get_double_square_axis():
    plt.figure(figsize=(2*FIG_WIDTH, 2*FIG_HEIGHT))
    ax = plt.subplot(1, 1, 1)
    return ax


def get_model_ordering(actual_models):
    desired_ordering = ['Sagittarius', 'linear', 'mean',
                        'mTAN', 'cvae', 'cpa', 'seq_by_seq_RNN',
                        'seq_by_seq_neuralODE']
    return sorted(actual_models, key=lambda m: desired_ordering.index(m))


def get_metric_name(name, setting):
    return {
        'RMSE': 'RMSE',
        'spearman_rank_by_genes': 'Spearman correlation' if
        setting != 'EvoDevo'
                else 'Spearman (rank genes)',
        'spearman_rank_by_times': 'T-Spearman correlation' if
        setting != 'EvoDevo'
                else 'Spearman (rank time points)',
        'MAPRC': 'MAPRC',
    }[name]


def get_model_colors(mod):
    return {
        'cvae': '#543005',
        'cpa': '#8c510a',
        'seq_by_seq_RNN': '#bf812d',
        'seq_by_seq_neuralODE': '#dfc27d',
        'mTAN': '#003c30',
        'mean': '#01665e',
        'linear': '#35978f',
        'Sagittarius': '#c7eae5'
    }[mod]


def get_sag_vs_baseline_colors(mod):
    if mod in {'Sagittarius'}:
        return '#c7eae5'
    else:
        return '#dfc27d'


def get_model_name_conventions(mname):
    if 'cvae' in mname:
        return 'cVAE'
    if 'cpa' in mname:
        return 'CPA'
    if 'seq_by_seq_RNN' in mname:
        return 'RNN'
    if 'seq_by_seq_neuralODE' in mname:
        return 'Neural ODE'
    if 'mTAN' in mname:
        return 'mTAN'
    if 'linear' in mname:
        return 'Linear'
    if 'mean' in mname:
        return 'Mean'
    if 'Sagittarius' in mname:
        return 'Sagittarius'

    print("Unrecognized:", mname)
    assert False


def get_LINCS_task_names(task, add_return=True):
    if task == 'continuousCombinatorialGeneration':
        return 'Complete{}Generation'.format('\n' if add_return
                                             else ' ')
    elif task == 'comboAndDosage':
        return 'Combination{}& Dosage'.format('\n' if add_return
                                              else ' ')
    elif task == 'comboAndTime':
        return 'Combination{}& Treatment Time'.format('\n' if add_return else
                                                      ' ')
    print('Unrecognized', task)
    assert False


def get_species_axis_tick(species):
    if species == 'RhesusMacaque':
        return 'Rhesus\nMacaque'  # make this two lines
    return species


def get_organ_color_palette():
    return ['#01665e', '#5ab4ac', '#c7eae5', '#f6e8c3', '#dfc27d',
            '#bf812d', '#8c510a']


def get_TM_color_palette():
    return {
        'Sagittarius': '#80cdc1',
        'baseline': '#018571',
        'Heart': '#f5f5f5',
        'Kidney': '#dfc27d',
        'Liver': '#a6611a'}


def get_base_color():
    return '#dfc27d'


def get_line_style(mod):
    if mod in {'Sagittarius'}:
        return 'solid'
    elif mod == 'mean':
        return (0, (5, 1))
    elif mod == 'linear':
        return 'dotted'
    raise ValueError("Unknown model: {}".format(mod))
