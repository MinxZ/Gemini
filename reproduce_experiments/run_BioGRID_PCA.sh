set -e

export CUDA_VISIBLE_DEVICES=0  # change as appropriate

num_thread=5
torch_thread=1
echo "...evaluating PCA embeddings for BioGRID"

for org in yeast mouse human_match
do
    if [ $org = yeast ]
    then
        ndim=200
    else
        ndim=400
    fi
    
    echo Running $org
    OPENBLAS_NUM_THREADS=1 python gemini/main_gemini.py --rwr pca --mixup -2 --num_thread $num_thread --torch_thread $torch_thread --method gemini --org $org --net GeneMANIA_ex --ndim 800
    python gemini/main_classifier.py --ndim $ndim --embed_name gemini_${org}_GeneMANIA_ex_800_pca --org $org --net GeneMANIA_ex --experiment_name gemini_${org}_GeneMANIA_ex_800_NN_pca
done

echo "...moving the results"
for org in yeast mouse human_match
do
    if [ $org = yeast ]
    then
        ndim=200
    else
        ndim=400
    fi
    
    if [ $org = human_match ]
    then
        short_org=human
    else
        short_org=$org
    fi
    
    cp data/results/raw/gemini_${org}_GeneMANIA_ex_800_NN_pca_${ndim}_labels.npy results/BioGrid/PCA/${short_org}_labels.npy
    cp data/results/raw/gemini_${org}_GeneMANIA_ex_800_NN_pca_${ndim}_pred.npy results/BioGrid/PCA/${short_org}_pred.npy
    cp data/results/raw/gemini_${org}_GeneMANIA_ex_800_NN_pca_${ndim}_testids.npy results/BioGrid/PCA/${short_org}_testids.npy
    cp data/results/gemini_${org}_GeneMANIA_ex_800_NN_pca_${ndim}_None_result.txt results/BioGrid/PCA/${short_org}.txt
done

echo "...processing the results by sub-ontology"
python gemini/process_subontology_results.py --method PCA --network BioGrid
