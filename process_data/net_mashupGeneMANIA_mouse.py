import pandas as pd
from func import textread
from tqdm import tqdm

org2binomial = {'yeast': 'Saccharomyces_cerevisiae',
                'human': 'Homo_sapiens',
                'mouse': 'Mus_musculus'}

org = 'mouse'
net = 'mashup'

# write mashup nets
file_name = 'data/raw/mashup_networks' + \
    '/10090.protein.links.detailed.v11.5.txt'
mashup_df = pd.read_csv(file_name, sep=' ')
gm = sorted(list(set(mashup_df.protein1).union(set(mashup_df.protein2))))
gm2idx = {g: idx for idx, g in enumerate(gm)}
string_nets = ['neighborhood', 'fusion', 'cooccurence',
               'coexpression',  'experimental', 'database']
string_nets = sorted(string_nets)
for i in tqdm(range(len(string_nets))):
    s_net = string_nets[i]
    filt = mashup_df[s_net] > 0
    s_df = mashup_df[filt]
    seq1 = list(s_df.protein1)
    seq2 = list(s_df.protein2)
    seq3 = list(s_df[s_net])

    seq1 = [gm2idx[gene] for gene in seq1]
    seq2 = [gm2idx[gene] for gene in seq2]
    path = f'data/networks/{org}/{org}_' +\
        f'string_{s_net}_adjacency.txt'
    with open(path, 'w') as f:
        for idx, pair in enumerate(zip(seq1, seq2)):
            f.write(f'{pair[0]} {pair[1]} {seq3[idx]}\n')
with open(f'data/networks/{org}/{org}_mashup_genes.txt', 'w') as f:
    f.writelines('\n'.join(gm))

for net in ['mashup', 'GeneMANIA']:
    genem = f'data/networks/{org}/{org}_mashup_genes.txt'
    gm = textread(genem)

    geneg = f"data/networks/{org}/" + \
        f"{org}_GeneMANIA_genes.txt"
    gg = textread(geneg)

    idx2gg = {idx: g for idx, g in enumerate(gg)}
    idx2gm = {idx: g for idx, g in enumerate(gm)}

    genes_union = sorted(list(set(gm + gg)))
    gene2idx = {gene: idx for idx, gene in enumerate(genes_union)}
    with open(f'data/networks/{org}/{org}_GeneMANIA_ex_genes.txt',
              'w') as f:
        for gene in genes_union:
            f.write(f'{gene}\n')
    with open(f'data/networks/{org}/{org}_mashup_GeneMANIA_ex_genes.txt',
              'w') as f:
        for gene in genes_union:
            f.write(f'{gene}\n')
    with open(f'data/networks/{org}/{org}_mashup_ex_genes.txt', 'w') as f:
        for gene in genes_union:
            f.write(f'{gene}\n')

    if net == 'mashup':
        string_nets = ['neighborhood', 'fusion', 'cooccurence',
                       'coexpression',  'experimental', 'database']
        string_nets = sorted(string_nets)

    elif net == 'GeneMANIA':
        org2binomial = {'yeast': 'Saccharomyces_cerevisiae',
                        'human': 'Homo_sapiens',
                        'mouse': 'Mus_musculus'}

        binomial = org2binomial[org]
        string_nets = []
        lst = pd.read_csv(f"data/raw/{binomial}/networks.txt",
                          sep='\t').File_Name
        string_nets = []
        for network in lst:
            net_path = f"data/raw/{binomial}/{network}"
            net_name = net_path.split('/')[-1].replace('.txt', '')
            string_nets.append(net_name)
        string_nets = sorted(string_nets)

    network_files = []
    for i in tqdm(range(len(string_nets))):
        network_file = f'data/networks/{org}/{org}_' +\
            f'string_{string_nets[i]}_adjacency.txt'
        seq1 = []
        seq2 = []
        seq3 = []
        with open(network_file, 'r') as f:
            for line in f.readlines():
                x, y, n = line.split()
                seq1.append(int(x))
                seq2.append(int(y))
                seq3.append(abs(float(n)))
        if net == 'GeneMANIA':
            seq1 = [gene2idx[idx2gg[gene]] for gene in seq1]
            seq2 = [gene2idx[idx2gg[gene]] for gene in seq2]
        else:
            seq1 = [gene2idx[idx2gm[gene]] for gene in seq1]
            seq2 = [gene2idx[idx2gm[gene]] for gene in seq2]
        path = f'data/networks/{org}/{org}_' +\
            f'string_{string_nets[i]}_gm_adjacency.txt'
        with open(path, 'w') as f:
            for idx, pair in enumerate(zip(seq1, seq2)):
                f.write(f'{pair[0]} {pair[1]} {seq3[idx]}\n')
