'''
For figure 3
merge STRING and BioGRID networks together: find the union genes and
map the gene index from original to the new one
'''

import pandas as pd
from func import textread
from tqdm import tqdm

for org in ['yeast']:
    for net in ['mashup', 'GeneMANIA']:
        genem = f'data/networks/{org}/{org}_mashup_genes.txt'
        gm = textread(genem)

        geneg = f"data/networks/{org.replace('_match', '')}/" + \
            f"{org.replace('_match', '')}_GeneMANIA_genes.txt"
        gg = textread(geneg)

        gs = gm if net == 'mashup' else gg
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
        idx2g = {idx: g for idx, g in enumerate(gs)}

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
            seq1 = [gene2idx[idx2g[gene]] for gene in seq1]
            seq2 = [gene2idx[idx2g[gene]] for gene in seq2]
            path = f'data/networks/{org}/{org}_' +\
                f'string_{string_nets[i]}_gm_adjacency.txt'
            with open(path, 'w') as f:
                for idx, pair in enumerate(zip(seq1, seq2)):
                    f.write(f'{pair[0]} {pair[1]} {seq3[idx]}\n')
