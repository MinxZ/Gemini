export CUDA_VISIBLE_DEVICES=0  # change as appropriate

num_thread=5
torch_thread=1
echo "...evaluating BIONIC embeddings for STRING"

echo Running yeast
python bionic/construct_config_file.py --species yeast --network String --config-name defaults --epochs 3000 --batch-size 2048 --sample-size 0 --gat-dim 64 --gat-heads 10 --gat-layers 2 --embedding-dim 512
bionic bionic/config_files/String_yeast_defaults.json
python bionic/downstream_evaluation.py --org yeast --network String --config-name defaults

echo Running mouse
python bionic/construct_config_file.py --species mouse --network String --config-name restricted --epochs 3000 --batch-size 2048 --sample-size 0 --gat-dim 64 --gat-heads 10 --gat-layers 1 --embedding-dim 512 --parallel
bionic bionic/config_files/String_mouse_restricted.json
python bionic/downstream_evaluation.py --org mouse --network String --config-name restricted

echo Running human
python bionic/construct_config_file.py --species human --network String --config-name defaults --epochs 3000 --batch-size 2048 --sample-size 0 --gat-dim 64 --gat-heads 10 --gat-layers 2 --embedding-dim 512
bionic bionic/config_files/String_human_defaults.json
python bionic/downstream_evaluation.py --org human --network String --config-name defaults

echo "...moving the results"
cp data/results/raw/String_yeast_defaults_labels.npy results/String/BIONIC/yeast_labels.npy
cp data/results/raw/String_yeast_defaults_pred.npy results/String/BIONIC/yeast_labels.npy
cp data/results/raw/String_yeast_defaults_testids.npy results/String/BIONIC/yeast_labels.npy
cp data/results/String_yeast_defaults_None_result.txt results/String/BIONIC/yeast.txt

cp data/results/raw/String_mouse_restricted_labels.npy results/String/BIONIC/mouse_labels.npy
cp data/results/raw/String_mouse_restricted_pred.npy results/String/BIONIC/mouse_labels.npy
cp data/results/raw/String_mouse_restricted_testids.npy results/String/BIONIC/mouse_labels.npy
cp data/results/String_mouse_restricted_None_result.txt results/String/BIONIC/mouse.txt

cp data/results/raw/String_human_defaults_labels.npy results/String/BIONIC/human_labels.npy
cp data/results/raw/String_human_defaults_pred.npy results/String/BIONIC/human_labels.npy
cp data/results/raw/String_human_defaults_testids.npy results/String/BIONIC/human_labels.npy
cp data/results/String_human_defaults_None_result.txt results/String/BIONIC/human.txt

echo "...processing the results by sub-ontology"
python gemini/process_subontology_results.py --method Bionic --network String
