export CUDA_VISIBLE_DEVICES=0  # change as appropriate

num_thread=5
torch_thread=1
echo "...evaluating BIONIC embeddings for STRING+BioGRID"

org=yeast
ndim=200
echo Running $org
python bionic/construct_config_file.py --species yeast --network Combo --config-name restricted --epochs 1000 --batch-size 256 --sample-size 10 --gat-dim 16 --gat-heads 10 --gat-layers 1 --embedding-dim 200
bionic bionic/config_files/Combo_yeast_restricted.json
python bionic/downstream_evaluation.py --org yeast --network Combo --config-name restricted

echo "...moving the results"
cp data/results/raw/Combo_yeast_restricted_labels.npy results/Combo/BIONIC/yeast_labels.npy
cp data/results/raw/Combo_yeast_restricted_pred.npy results/Combo/BIONIC/yeast_labels.npy
cp data/results/raw/Combo_yeast_restricted_testids.npy results/Combo/BIONIC/yeast_labels.npy
cp data/results/Combo_yeast_restricted_None_result.txt results/Combo/BIONIC/yeast.txt

echo "...processing the results by sub-ontology"
python gemini/process_subontology_results.py --method Bionic --network Combo
