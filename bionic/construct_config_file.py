import os
import sys
import pandas as pd
import json
import argparse

sys.path.append(os.path.join(sys.path[0], '../'))
from config import GEMINI_DIR


parser = argparse.ArgumentParser()
parser.add_argument('--species', type=str, help='yeast, human, or mouse')
parser.add_argument('--network', type=str, help='BioGrid, String, or Combo')
parser.add_argument('--config-name', type=str, help='name to use for this configuration')
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--batch-size', type=int, default=2048)
parser.add_argument('--sample-size', type=int, default=0)
parser.add_argument('--gat-dim', type=int, default=64)
parser.add_argument('--gat-heads', type=int, default=10)
parser.add_argument('--gat-layers', type=int, default=2)
parser.add_argument('--embedding-dim', type=int, default=512)
parser.add_argument('--parallel', action='store_true')
args = parser.parse_args()


if not os.path.exists(GEMINI_DIR + 'bionic/config_files/'):
    os.makedirs(GEMINI_DIR + 'bionic/config_files/')

cfile = GEMINI_DIR + 'bionic/config_files/{}_{}_{}.json'.format(args.network, args.species, args.config_name)
fdir = GEMINI_DIR + 'data/networks/'
fdir += 'human_match' if args.species.lower() == 'human' else args.species.lower()

def filt(net):
    if args.network == 'Combo':
        return '_gm_adjacency.txt' in net
    is_string = any([net == '{}_string_{}_gm_adjacency.txt'.format(
        args.species if args.species != 'human' else 'human_match', evidence)
                for evidence in ['neighborhood', 'fusion', 'cooccurence', 'coexpression', 'experimental', 'database']])
    if args.network == 'String':
        return is_string
    # otherwise, must be biogrid
    assert args.network == 'BioGrid'
    return not is_string and '_gm_adjacency.txt' in net

names = os.listdir(fdir)
net_names = ['{}/{}'.format(fdir, fname) for fname in names if filt(fname)]
print('...produced {} networks.'.format(len(net_names)))

config = {
        'outname': 'bionic/output/{}/{}/{}/'.format(args.network.upper(), args.species, args.config_name),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'sample_size': args.sample_size,
        'learning_rate': 0.0005,
        'gat_shapes': {
            'dimension': args.gat_dim,
            'n_heads': args.gat_heads,
            'n_layers': args.gat_layers,
        },
        'embedding_size': args.embedding_dim,
        'save_model': True,
        'model_parallel': args.parallel,
        'net_names': net_names
}

with open(cfile, 'w') as f:
    json.dump(config, f)
    
