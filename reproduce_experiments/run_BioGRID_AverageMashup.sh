set -e

export CUDA_VISIBLE_DEVICES=0  # change as appropriate

num_thread=5
torch_thread=1
echo "...evaluating Average Mashup embeddings for BioGRID"

for org in yeast mouse human_match
do
    if [ $org = yeast ]
    then
        ndim=200
    else
        ndim=400
    fi
    
    echo Running $org
    OPENBLAS_NUM_THREADS=1 python gemini/main_gemini.py --mixup -1 --num_thread $num_thread --torch_thread $torch_thread --method DCA --org $org --net GeneMANIA_ex --ndim 800
    python gemini/main_classifier.py --ndim $ndim --embed_name DCA_${org}_GeneMANIA_ex_800_average --org $org --net GeneMANIA_ex --experiment_name DCA_${org}_GeneMANIA_ex_800_NN_average
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
    
    cp data/results/raw/DCA_${org}_GeneMANIA_ex_800_NN_average_${ndim}_labels.npy results/BioGrid/AVERAGE_MASHUP/${shrot_org}_labels.npy
    cp data/results/raw/DCA_${org}_GeneMANIA_ex_800_NN_average_${ndim}_pred.npy results/BioGrid/AVERAGE_MASHUP/${short_org}_pred.npy
    cp data/results/raw/DCA_${org}_GeneMANIA_ex_800_NN_average_${ndim}_testids.npy results/BioGrid/AVERAGE_MASHUP/${short_org}_testids.npy
    cp data/results/DCA_${org}_GeneMANIA_ex_800_NN_average_${ndim}_None_result.txt results/BioGrid/AVERAGE_MASHUP/${short_org}.txt
done

echo "...processing the results by sub-ontology"
python gemini/process_subontology_results.py --method Average_Mashup --network BioGrid
