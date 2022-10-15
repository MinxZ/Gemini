# Gemini
Memory-efficient integration of hundreds of heterogeneous gene networks through high-order pooling

## Introductoin
Gemini is a novel network integration method that uses memory-efficient high-order pooling to represent and weight each network according to its uniquenes. Gemini mitigates the uneven distribution through mixing up existing networks to create many new networks and integrates them into one representation for further tasks. 

## Repository structure
```
| Gemini/
|
| --- data/: store all networks files and representation files.
|
| --- plot/: jupyter notebooks to recreate the figures in the paper. 
|
| --- process_data/: process raw txt network files and merge multiple source data
|
| --- gemini/: experimental code for the STRING and BioGRID dataset, including:
|       | --- main_gemini.py: run RWR(random walk with restart) on all networks and get rwr matrix for each network and one integrated embedding 
|       | --- main_gemin_cluster.py: run clustering methods on kurtosis embedding of all networks and calculate the weight for each network according to the cluster size
|       | --- main_gemini.py: run cross-validation on all network embeddings 
|
```


## Installation Tutorial

### System Requirements
Gemini is implemented using Python 3.9 on LINUX. Gemini expects torch==1.9.1+cu11.1, scipy, numpy, pandas, sklearn, matplotlib, seaborn, and so on. For best performance, Gemini can be run with a GPU and a CPU. Multi thread is recommended. However, all experiments can also be run on a CPU.

## How to use our code
1. To process all data, run `sh process_data/preprocess.sh`. 
2. To reproduce all experiment, run `sh gemin/main.sh` (will add notebooks later.)
3. To reproduce figures from the paper, run `sh figures/plot.sh` (will add notebook later.)
```