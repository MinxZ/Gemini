import json
import os

import pandas as pd
from tqdm import tqdm

if not os.path.exists('data/log/'):
    os.mkdir('data/log/')
if not os.path.exists('data/results/'):
    os.mkdir('data/results/')


def textread(filename):
    output = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            output.append(line.strip())
    return output


print('read mapping for mouse genes')
filename = 'data/raw/aliase/10090.protein.aliases.v11.5.txt'
alias2string_mouse = {}
with open(filename, 'r') as f:
    for line in tqdm(f.readlines()):
        string, alias = line.strip().split('\t')[:2]
        alias2string_mouse[alias] = string

print('read mapping for human genes')
filename = 'data/raw/aliase/9606.protein.aliases.v11.5.txt'
alias2string_human = {}
with open(filename, 'r') as f:
    for line in tqdm(f.readlines()):
        string, alias = line.strip().split('\t')[:2]
        alias2string_human[alias] = string

for org in ['yeast', 'mouse', 'human', 'human_match']:
    anno_path = f'data/annotations/{org}'
    if not os.path.exists(anno_path):
        os.mkdir(anno_path)

    if org == 'human_match':
        goa = pd.read_csv('data/raw/goa/GOA_human.csv')
    else:
        goa = pd.read_csv(f'data/raw/goa/GOA_{org}.csv')
    # GO_annotations  protein id string id

    gene_name = 'protein id' if org != 'yeast' else 'string id'
    gene_anno = {}
    for i in range(len(goa)):
        item = goa.iloc[i]
        gene = item[gene_name]
        if org == 'mouse':
            gene = alias2string_mouse[gene] if gene \
                in alias2string_mouse else gene
        elif org == 'human_match':
            gene = alias2string_human[gene] if gene \
                in alias2string_human else gene
        anno = item['GO_annotations'].replace("'", "\"")
        gene_anno[gene] = json.loads(anno)

    annos = [i for key, anno in gene_anno.items() for i in anno]
    anno_set = set(annos)
    anno_lst = sorted(list(anno_set))
    anno2idx = {anno: idx for idx, anno in enumerate(anno_lst)}
    # 5912

    genes_goa = set(gene_anno.keys())
    genes_goa_lst = sorted(list(genes_goa))
    genes_goa2idx = {anno: idx for idx, anno in enumerate(genes_goa_lst)}

    if org != 'human_match':
        gene_file = f'data/networks/{org}/{org}_GeneMANIA_genes.txt'
        genes_mania = set(textread(gene_file))
        genes_mania_lst = sorted(list(genes_mania))
        # 6400

        print(org, 'GeneMANIA')
        print(f'genes_mania: {len(genes_mania)}')
        print(f'genes_goa: {len(genes_goa)}')
        print(f'common genes: {len(genes_mania.intersection(genes_goa))}')
        print()

    if org == 'yeast' or org == 'human_match':
        gene_file = f'data/networks/{org}/{org}_mashup_genes.txt'
        genes_mania = set(textread(gene_file))
        genes_mania_lst = sorted(list(genes_mania))
        # 6400

        print(org, 'mashup')
        print(f'genes_mania: {len(genes_mania)}')
        print(f'genes_goa: {len(genes_goa)}')
        print(f'common genes: {len(genes_mania.intersection(genes_goa))}')
        print()

    with open(f'{anno_path}/{org}_go_terms.txt', 'w') as f:
        for anno in anno_lst:
            f.write(f'{anno}\n')

    with open(f'{anno_path}/{org}_go_genes.txt', 'w') as f:
        for gene in genes_goa_lst:
            f.write(f'{gene}\n')

    with open(f'{anno_path}/{org}_go_adjacency.txt', 'w') as f:
        for gene, annos in gene_anno.items():
            for anno in annos:
                gene_idx = genes_goa2idx[gene]
                anno_idx = anno2idx[anno]
                f.write(f'{gene_idx} {anno_idx}\n')
