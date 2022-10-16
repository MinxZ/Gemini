# Gemini
Memory-efficient integration of hundreds of heterogeneous gene networks through high-order pooling

## Introductoin
Gemini is a novel network integration method that uses memory-efficient high-order pooling to represent and weight each network according to its uniquenes. Gemini mitigates the uneven distribution through mixing up existing networks to create many new networks and integrates them into one representation for further tasks. 

## Repository structure
```
| Gemini/
|
| --- data/: store all networks files and representation files.
|       | --- raw/:
|           | --- Saccharomyces_cerevisiae: yeast networks from BioGRID
|           | --- Home_sapiens: human networks from BioGRID
|           | --- Mus_musculus: mouse networks from BioGRID
|           | --- GOA_human.csv: GOA annotation for human
|           | --- GOA_mouse.csv: GOA annotation for mouse
|           | --- GOA_yeast.csv: GOA annotation for yeast
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
Gemini is implemented using Python 3.9 on LINUX. Gemini expects torch==1.9.1+cu11.1, scipy, numpy, pandas, sklearn, matplotlib, seaborn, and so on. For best performance, Gemini can be run with a GPU and a CPU. However, all experiments can also be run on a CPU. Multi thread is recommended, setting the `--nnum_thread`

## How to use our code
1. To process all data, run `sh process_data/preprocess.sh`. 
2. To mimic the papers's results, use `sh gemini/main_gemini.sh`.
3. To run Gemini on BioGRID or STRING dataset, run `python gemini/main_gemini.py --mixup 1 --mixup2 1.0 --embed_type kurtosis --net <GeneMANIA or mashup> --org <org name>`. Set `mixup` to 1 to run Gemini, set `mixup` to 0 to run Mahsup. Set `embed_type` to mean, variance, or skewness to change embedding used for clustering. To change the number of mixup pairs used in the integration, set `mixup2` to the ratio of the wanted number to the number of networks in the dataset. To view all possible commandline arguments, add the `--help` flag. To set cpu multithread use, use `--num_thread <number of thread>`. (will add notebook later.)
4. To run Mashup on BioGRID or STRING dataset, run `python gemini/main_gemini.py --mixup 0 --net <GeneMANIA or mashup> --org <org name>`. Set `mixup` to 0 to run Mahsup. To view all possible commandline arguments, add the `--help` flag. To set cpu multithread use, use `--num_thread <number of thread>`. (will add notebook later.)
5. To reproduce figures from the paper, run `sh figures/plot.sh` (will add notebook later.)
6. To run Gemini on a new dataset, open the `gemini/gemini_new_networks.ipynb` and run, replacing the dataset path as desired.
6. To test embedding from Gemini on BioGRID dataset on new annotations, open the `gemini/gemini_new_annotations.ipynb` and run, replacing the annotation path as desired. (will add notebook later.)