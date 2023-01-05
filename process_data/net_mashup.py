# from functools import partial
# from multiprocessing import Pool, cpu_count

import os
import shutil

from tqdm import tqdm

if not os.path.exists('data/networks/'):
    os.mkdir('data/networks/')

print('read mapping for human genes')
filename = 'data/raw/aliase/9606.protein.aliases.v11.5.txt'
alias2string = {}
with open(filename, 'r') as f:
    for line in tqdm(f.readlines()):
        string, alias = line.strip().split('\t')[:2]
        alias2string[alias] = string


orgs = ['human_match', 'human', 'yeast']
data_path = 'data/raw/mashup_networks'
for org in orgs:
    if not os.path.exists(f'data/networks/{org}/'):
        os.mkdir(f'data/networks/{org}/')
    genes = []
    if org == 'human_match':
        src = 'data/raw/mashup_networks/human/human_string_genes.txt'
        dst = f'data/networks/{org}/{org}_mashup_genes.txt'
        with open(src, 'r') as f:
            for line in f.readlines():
                gene = line.strip()
                genes.append(gene)
        genes_str = [alias2string[gene] if gene in alias2string else
                     gene for gene in genes]
        genes_str = sorted(list(set(genes_str)))
        gene_str2idx = {gene: idx for idx, gene in enumerate(genes_str)}
        geneidx2idx = {i: gene_str2idx[alias2string[gene]]
                       if gene in alias2string
                       else gene_str2idx[gene] for i, gene in enumerate(genes)}
        genes = genes_str
        with open(dst, 'w') as f:
            f.writelines('\n'.join(genes))
    else:
        src = f'data/raw/mashup_networks/{org}/{org}_string_genes.txt'
        dst = f'data/networks/{org}/{org}_mashup_genes.txt'
        shutil.copyfile(src, dst)

    lst = {'neighborhood', 'fusion', 'cooccurence',
           'coexpression',  'experimental', 'database'}
    for net_name in tqdm(lst):
        if org == 'human_match':
            net_path = \
                f"{data_path}/human/human_string_{net_name}_adjacency.txt"
        else:
            net_path = \
                f"{data_path}/{org}/{org}_string_{net_name}_adjacency.txt"
        seq1 = []
        seq2 = []
        seq3 = []
        with open(net_path, 'r') as f:
            for line in f.readlines():
                seq1.append(int(line.strip().split()[0])-1)
                seq2.append(int(line.strip().split()[1])-1)
                seq3.append(float(line.strip().split()[2]))
            if org == 'human_match':
                seq1 = [geneidx2idx[gene] for gene in seq1]
                seq2 = [geneidx2idx[gene] for gene in seq2]
        path = f'data/networks/{org}/{org}_string_{net_name}_adjacency.txt'
        with open(path, 'w') as f:
            for idx, pair in enumerate(zip(seq1, seq2)):
                f.write(f'{pair[0]} {pair[1]} {seq3[idx]}\n')
