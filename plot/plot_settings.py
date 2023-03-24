import matplotlib as mpl
import matplotlib.pyplot as plt

# set basic parameters
mpl.rcParams['pdf.fonttype'] = 42

LARGER_SIZE = 16
MEDIUM_SIZE = 14
SMALLER_SIZE = 12
plt.rc('font', size=LARGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALLER_SIZE)	 # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)	 # fontsize of the tick labels
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('font', family='Helvetica')
FIG_HEIGHT = 4
FIG_WIDTH = 4


MAIN_METHODS = ['SVD', 'PCA', 'Average_Mashup', 'Bionic', 'Mashup', 'Gemini']


def get_square_axis(double=False):
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT) if not double else (2*FIG_WIDTH, 2*FIG_HEIGHT))
    ax = plt.subplot(1, 1, 1)
    return ax


def get_wider_axis(double=False, scale=None):
    if double:
        scale = 2
    elif scale is None:
        scale = 3/2
    plt.figure(figsize=(int(scale*FIG_WIDTH), FIG_HEIGHT))
    ax = plt.subplot(1, 1, 1)
    return ax


def get_model_ordering(actual_models):
    desired_ordering = ['average_mashup', 'svd', 'pca', 'bionic', 'mashup', 'gemini']
    return sorted(actual_models, key=lambda m: desired_ordering.index(m.lower()))


def get_network_ordering():
    return ['String', 'BioGrid', 'Combo']


def get_network_naming_convention(net):
    naming = {
        'String': 'STRING',
        'STRING': 'STRING',
        'BioGrid': 'BioGRID',
        'BIOGRID': 'BioGRID',
        'Combo': 'STRING+BioGRID'
    }
    assert net in naming
    return naming[net]


def get_metric_name(name):
    return {
        'max f1': r'maximum F$_1$',
        'micro AUPRC': 'micro-AUPRC',
        'macro AUPRC': 'macro-AUPRC',
    }[name]


def get_model_colors(mod):
    return {
        'average_mashup': '#8c510a',
        'svd': '#d8b365',
        'pca': '#f6e8c3',
        'bionic': '#c7eae5',
        'mashup': '#5ab4ac',
        'gemini': '#01665e',
    }[mod.lower()]


def get_model_colors_by_network(mod, network):
    coloring = {
        'bionic': {
            'String': '#e7d4e8',
            'BioGrid': '#af8dc3',
            'Combo': '#762a83'},
        'mashup': {
            'String': '#f6e8c3',
            'BioGrid': '#d8b365',
            'Combo': '#8c510a'},
        'gemini': {
            'String': '#c7eae5',
            'BioGrid': '#5ab4ac',
            'Combo': '#01665e'},
    }
    return coloring[mod.lower()][network]


def get_species_color(sp):
    return {
        'mouse': '#8c510a',
        'human': '#01665e',
        'yeast': '#762a83',
    }[sp.lower()]


def get_network_linestyle(net):
    return {
        'STRING': 'dashed',
        'BIOGRID': 'solid',
    }[net.upper()]


def get_model_name_conventions(mname):
    naming = {
        'mashup': 'Mashup',
        'svd': 'SVD',
        'pca': 'PCA',
        'bionic': 'BIONIC',
        'gemini': 'Gemini',
        'average_mashup': 'Average Mashup'
    }
    assert mname.lower() in naming, 'Unknown model {}'.format(mname)
    return naming[mname.lower()]
