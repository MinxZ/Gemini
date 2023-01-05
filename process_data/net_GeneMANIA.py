# from functools import partial
# from multiprocessing import Pool, cpu_count

import os

import pandas as pd
from tqdm import tqdm

if not os.path.exists('data/networks/'):
    os.mkdir('data/networks/')

print('read mapping for mouse genes')
filename = 'data/raw/aliase/10090.protein.aliases.v11.5.txt'
alias2string = {}
with open(filename, 'r') as f:
    for line in tqdm(f.readlines()):
        string, alias = line.strip().split('\t')[:2]
        alias2string[alias] = string


org2binomial = {'yeast': 'Saccharomyces_cerevisiae',
                'human': 'Homo_sapiens',
                'mouse': 'Mus_musculus'
                }

for org in org2binomial.keys():
    print(org)
    if not os.path.exists(f'data/networks/{org}/'):
        os.mkdir(f'data/networks/{org}/')
    binomial = org2binomial[org]

    lst = pd.read_csv(f"data/raw/{binomial}/networks.txt", sep='\t')
    lst = lst.File_Name
    genes = []
    for net in tqdm(lst):
        net_path = f"data/raw/{binomial}/{net}"
        data = pd.read_csv(net_path, sep='\t')
        seq1 = list(data['Gene_A'])
        seq2 = list(data['Gene_B'])
        genes.extend(list(set(seq1+seq2)))
    genes = sorted(list(set(genes)))
    if org == 'mouse':
        genes_str = [alias2string[gene] if gene in alias2string else
                     gene for gene in genes]
        genes_str = sorted(list(set(genes_str)))
        gene_str2idx = {gene: idx for idx, gene in enumerate(genes_str)}
        gene2idx = {gene: gene_str2idx[alias2string[gene]]
                    if gene in alias2string
                    else gene_str2idx[gene] for gene in genes}
        genes = genes_str
    else:
        gene2idx = {gene: idx for idx, gene in enumerate(genes)}
    with open(f'data/networks/{org}/{org}_GeneMANIA_genes.txt', 'w') as f:
        f.writelines('\n'.join(genes))

    print('process networks')
    for net in tqdm(lst):
        net_path = f"data/raw/{binomial}/{net}"
        data = pd.read_csv(net_path, sep='\t')
        net_name = net_path.split('/')[-1].replace('.txt', '')
        seq1 = list(data['Gene_A'])
        seq2 = list(data['Gene_B'])
        seq3 = list(data['Weight'])
        seq1 = [gene2idx[gene] for gene in seq1]
        seq2 = [gene2idx[gene] for gene in seq2]
        path = f'data/networks/{org}/{org}_string_{net_name}_adjacency.txt'
        with open(path, 'w') as f:
            for idx, pair in enumerate(zip(seq1, seq2)):
                f.write(f'{pair[0]} {pair[1]} {seq3[idx]}\n')
