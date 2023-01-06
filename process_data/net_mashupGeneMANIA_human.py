import pandas as pd
from func import textread
from tqdm import tqdm

print('read mapping for human genes')
filename = 'data/raw/aliase/9606.protein.aliases.v11.5.txt'
alias2string = {}
with open(filename, 'r') as f:
    for line in tqdm(f.readlines()):
        string, alias = line.strip().split('\t')[:2]
        alias2string[alias] = string

org2binomial = {'yeast': 'Saccharomyces_cerevisiae',
                'human': 'Homo_sapiens',
                'mouse': 'Mus_musculus'}

org = 'human'
net = 'GeneMANIA'

# for org in ['human_match']:
# for net in ['mashup']:
for net in ['mashup', 'GeneMANIA']:
    # for net in ['GeneMANIA']:
    org = 'human_match' if net == 'mashup' else 'human'
    genem = f'data/networks/{org}/{org}_mashup_genes.txt'
    gm = textread(genem)

    geneg = f"data/networks/{org.replace('_match', '')}/" + \
        f"{org.replace('_match', '')}_GeneMANIA_genes.txt"
    gg = textread(geneg)

    idx2gm = {
        idx: alias2string[g] if g in alias2string else g for idx, g in
        enumerate(gm)}
    idx2gg = {
        idx: alias2string[g] if g in alias2string else g for idx, g in
        enumerate(gg)}

    gg = [alias2string[g] if g in alias2string else g for g in gg]
    gm = [alias2string[g] if g in alias2string else g for g in gm]

    # gs = gm if net == 'mashup' else gg
    genes_union = sorted(list(set(gm + gg)))
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
    gene2idx = {gene: idx for idx, gene in enumerate(genes_union)}
    # idx2g = {idx: g for idx, g in enumerate(gs)}

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
        org = 'human_match'
        path = f'data/networks/{org}/{org}_' +\
            f'string_{string_nets[i]}_gm_adjacency.txt'
        with open(path, 'w') as f:
            for idx, pair in enumerate(zip(seq1, seq2)):
                f.write(f'{pair[0]} {pair[1]} {seq3[idx]}\n')
        org = 'human_match' if net == 'mashup' else 'human'
