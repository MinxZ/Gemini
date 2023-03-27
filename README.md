# Gemini
Memory-efficient integration of hundreds of heterogeneous gene networks through high-order pooling

## Introduction
Gemini is a novel network integration method that uses memory-efficient high-order pooling to represent and weight each network according to its uniqueness. Gemini mitigates the uneven distribution of biological signal by through up existing networks to create many new networks and then produces integrated gene embeddings for downstream tasks. 

## Repository structure
```
| Gemini/
|
| --- gemini/: experimental code for the STRING and BioGRID dataset, including:
|       | --- main_gemini.py: run RWR(random walk with restart) on all networks and get rwr matrix for each network and one integrated embedding 
|       | --- main_gemin_cluster.py: run clustering methods on kurtosis embedding of all networks and calculate the weight for each network according to the cluster size
|       | --- main_classifier.py: run cross-validation on all network embeddings 
|
| --- plot/: jupyter notebooks to recreate the figures in the paper. 
|
| --- reproduce_experiments/: bash scripts to produce the results presented in the Gemini paper, including:
|       | --- run_BioGRID_Gemini.sh: learn Gemini embeddings on the BioGRID network collection and evaluate for downstream protein function prediction.
|
| --- results/: saved result files for experiments.
|
| --- time_and_memory_study/: code for runtime and CPU/GPU ablation in the Gemini paper. --- data/: store all networks files and representation files.
|       | --- raw/:
|           | --- mashup_networks
|               | --- human: STRING networks for human
|               | --- yeast: STRING networks for yeast
|               | --- 10090.protein.links.detailed.v11.5.txt: STRING networks for mouse
|           | --- goa
|               | --- GOA_human.csv: GOA annotation for human
|               | --- GOA_mouse.csv: GOA annotation for mouse
|               | --- GOA_yeast.csv: GOA annotation for yeast
|           | --- Saccharomyces_cerevisiae: yeast networks from BioGRID
|           | --- Home_sapiens: human networks from BioGRID
|           | --- Mus_musculus: mouse networks from BioGRID
|
| --- process_data/: process raw txt network files and merge multiple source data
|
| --- bionic/: configuration files and processing code for the BIONIC comparison model.
```


## Installation Tutorial

### System Requirements
Gemini is implemented using Python 3.9 on LINUX. Gemini expects torch==1.9.1+cu11.1, scipy, numpy, pandas, sklearn, matplotlib, seaborn, and so on. For best performance, Gemini can be run with a GPU and a CPU. However, all experiments can also be run on a CPU. Multi thread is recommended, setting the `--nnum_thread`

To run the BIONIC comparison model, install the BIONIC package from https://github.com/bowang-lab/BIONIC using Python 3.8.

## How to download data
1. Use links following to download data to data/raw

BioGRID human:
https://drive.google.com/file/d/1l4Daft3yQHW-StKuWzPdN257EaWQm9zJ/view?usp=share_link

BioGRID mouse: 
https://drive.google.com/file/d/1z3rGGIrKpa8YmnY2IBaYqbOzTGAlcpK8/view?usp=share_link

BioGRID yeast: 
https://drive.google.com/file/d/1Im3J6gj7jY3_eGSG52Rvw4Oj0i8HoMhf/view?usp=share_link

GO Annotation:
https://drive.google.com/file/d/1UZW5ZIrmzGkv0_iOU8_CDNJ2PMzEQZ6g/view?usp=share_link


2. Use `sh get_data.sh` to download other data


## How to use our code
1. To process all data, first run `sh process_data/preprocess.sh`. 
2. Update `config.py` with the absolute path to the Gemini directory in your computing environment.
3. To mimic the papers's results, use the relevant files in `reproduce_experiments/`.
4. To reproduce paper figures, use the Jupyter Notebooks or python scripts in `plot/`. These can be run using our stored result files, or can be run after the `reproduce_experiments/` commands in (2) to use updated result files.
5. To run Gemini on a new dataset, open the `gemini/gemini_new_networks.ipynb` and run, replacing the dataset path as desired.
6. To test embedding from Gemini on BioGRID dataset on new annotations, open the `gemini/gemini_new_annotations.ipynb` and run, replacing the annotation path as desired.

## Issues
Please open an issue for any difficulties installing or running Gemini, and we will be happy to assist.
