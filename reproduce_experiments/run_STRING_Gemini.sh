export CUDA_VISIBLE_DEVICES=0  # change as appropriate

num_thread=5
torch_thread=1
echo "...evaluating Gemini embeddings for STRING"

for org in yeast mouse human_match
do
    if [ $org = yeast ]
    then
        ndim=200
    else
        ndim=400
    fi
    
    echo Running $org
    OPENBLAS_NUM_THREADS=5 python gemini/main_gemini_cluster.py --embed_type Qsm4 --axis 1 --level network --cluster_method ap --separate 35 --run_mashup 0 --num_thread $num_thread --torch_thread $torch_thread --method gemini --net mashup_ex --ndim $ndim --org $org
    OPENBLAS_NUM_THREADS=1 python gemini/main_gemini.py --num_thread $num_thread --torch_thread $torch_thread --method gemini --org $org --net mashup_ex --ndim 800 --embed_type Qsm4 --axis 1 --level network --cluster_method ap --separate 35 --weight 1 --ori_weight 0.5 --mixup 1 --mixup2 1 --gamma 0.5
    python gemini/main_classifier.py --ndim $ndim --embed_name gemini_${org}_mashup_ex_800_Qsm41_separate35_ap_weight1_0.5_network_mixup1_1.0_gamma0.5 --org $org --net mashup_ex --experiment_name gemini_${org}_mashup_ex_800_Qsm41_separate35_ap_weight1_0.5_network_mixup1_1.0_gamma0.5
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
    
    cp data/results/raw/gemini_${org}_mashup_ex_800_Qsm41_separate35_ap_weight1_0.5_network_mixup1_1.0_gamma0.5_${ndim}_labels.npy results/String/GEMINI/${short_org}_labels.npy
    cp data/results/raw/gemini_${org}_mashup_ex_800_Qsm41_separate35_ap_weight1_0.5_network_mixup1_1.0_gamma0.5_${ndim}_pred.npy results/String/GEMINI/${short_org}_pred.npy
    cp data/results/raw/gemini_${org}_mashup_ex_800_Qsm41_separate35_ap_weight1_0.5_network_mixup1_1.0_gamma0.5_${ndim}_testids.npy results/String/GEMINI/${org}_testids.npy
    cp data/results/gemini_${org}_mashup_ex_800_Qsm41_separate35_ap_weight1_0.5_network_mixup1_1.0_gamma0.5_${ndim}_None_result.txt results/String/GEMINI/${short_org}.txt
done

echo "...processing the results by sub-ontology"
python gemini/process_subontology_results.py --method Gemini --network String
