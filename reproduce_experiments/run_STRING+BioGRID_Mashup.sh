export CUDA_VISIBLE_DEVICES=0  # change as appropriate

num_thread=5
torch_thread=1
echo "...evaluating Mashup embeddings for STRING+BioGRID"

for org in yeast mouse human_match
do
    if [ $org = yeast ]
    then
        ndim=200
    else
        ndim=400
    fi
    
    echo Running $org
    OPENBLAS_NUM_THREADS=1 python gemini/main_gemini.py --num_thread $num_thread --torch_thread $torch_thread --method DCA --org $org --net mashup_GeneMANIA_ex --ndim 800
    python gemini/main_classifier.py --ndim $ndim --embed_name DCA_${org}_mashup_GeneMANIA_ex_800 --org $org --net mashup_GeneMANIA_ex --experiment_name DCA_${org}_mashup_GeneMANIA_ex_800_NN
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
    
    cp data/results/raw/DCA_${org}_mashup_GeneMANIA_ex_800_NN_${ndim}_labels.npy results/Combo/MASHUP/${short_org}_labels.npy
    cp data/results/raw/DCA_${org}_mashup_GeneMANIA_ex_800_NN_${ndim}_pred.npy results/Combo/MASHUP/${short_org}_pred.npy
    cp data/results/raw/DCA_${org}_mashup_GeneMANIA_ex_800_NN_${ndim}_testids.npy results/Combo/MASHUP/${short_org}_testids.npy
    cp data/results/DCA_${org}_mashup_GeneMANIA_ex_800_NN_${ndim}_None_result.txt results/Combo/MASHUP/${short_org}.txt
done

echo "...processing the results by sub-ontology"
python gemini/process_subontology_results.py --method Mashup --network Combo
