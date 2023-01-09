# fig 3 (3 mashup
export CUDA_VISIBLE_DEVICES=0
OPENBLAS_NUM_THREADS=5
num_thread=4
torch_thread=5
method=gemini
model_type=NN
# for org in yeast mouse human_match; do
for org in human_match yeast mouse; do
    for net in mashup_GeneMANIA_ex GeneMANIA_ex mashup_ex; do
    # for net in mashup_ex mashup_GeneMANIA_ex GeneMANIA_ex; do
        ndim=800
        echo $experiment_name
        experiment_name=${method}_${org}_${net}_${ndim}_${model_type}
        embed_name=${method}_${org}_${net}_${ndim}
        OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS python gemini/main_gemini.py --num_thread $num_thread --torch_thread $torch_thread --method $method --org $org --net $net --ndim $ndim
        if [ $org = yeast ]; then
            ndim=200
        else
            ndim=400
        fi
        python gemini/main_classifier.py --ndim $ndim --embed_name $embed_name --org $org --net $net --experiment_name $experiment_name
    done
done

# fig 2 (4
export CUDA_VISIBLE_DEVICES=0
OPENBLAS_NUM_THREADS=1
num_thread=20
torch_thread=1
method=gemini
model_type=NN
net=mashup
for org in yeast mouse human; do
    ndim=800
    echo $experiment_name
    experiment_name=${method}_${org}_${net}_${ndim}_${model_type}
    embed_name=${method}_${org}_${net}_${ndim}
    OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS python gemini/main_gemini.py --num_thread $num_thread --torch_thread $torch_thread --method $method --org $org --net $net --ndim $ndim
    if [ $org = yeast ]; then
        ndim=200
    else
        ndim=400
    fi
    python gemini/main_classifier.py --ndim $ndim --embed_name $embed_name --org $org --net $net --experiment_name $experiment_name

    ndim=800
    experiment_name=${method}_${org}_${net}_${ndim}_${model_type}_average
    embed_name=${method}_${org}_${net}_${ndim}_average
    OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS python gemini/main_gemini.py --mixup -1 --num_thread $num_thread --torch_thread $torch_thread --method $method --org $org --net $net --ndim $ndim
    if [ $org = yeast ]; then
        ndim=200
    else
        ndim=400
    fi
    python gemini/main_classifier.py --ndim $ndim --embed_name $embed_name --org $org --net $net --experiment_name $experiment_name

    for rwr in pca svd; do
        ndim=800
        experiment_name=${method}_${org}_${net}_${ndim}_${model_type}_${rwr}
        embed_name=${method}_${org}_${net}_${ndim}_${rwr}
        OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS python gemini/main_gemini.py --rwr $rwr --mixup -2 --num_thread $num_thread --torch_thread $torch_thread --method $method --org $org --net $net --ndim $ndim
        if [ $org = yeast ]; then
            ndim=200
        else
            ndim=400
        fi
        python gemini/main_classifier.py --ndim $ndim --embed_name $embed_name --org $org --net $net --experiment_name $experiment_name
    done
done

# fig3 (3
export CUDA_VISIBLE_DEVICES=0
OPENBLAS_NUM_THREADS=5
num_thread=4
torch_thread=5
method=gemini
model_type=NN
ori_seed=1
echo $ori_seed
gamma=0.5
cluster_method=ap
separate=35
od=4
embed_type_=Qsm
embed_type=${embed_type_}${od}
mixup=1
weight=1
ori_weight=0.5
level=network
mixup2=1.0
for net in GeneMANIA_ex mashup_GeneMANIA_ex mashup_ex; do
# for net in mashup_ex GeneMANIA_ex mashup_GeneMANIA_ex; do
    for org in yeast mouse human_match; do
        if [ $org = yeast ]; then
            ndim=200
        else
            ndim=400
        fi
        embed_type=Qsm4
        axis=1
        OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS python gemini/main_gemini_cluster.py --embed_type $embed_type --axis $axis --level network --cluster_method $cluster_method --separate $separate --run_mashup 0 --num_thread $num_thread --torch_thread $torch_thread --method $method --org $org --net $net --ndim $ndim
        OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS python gemini/main_gemini.py --ori_seed $ori_seed --mixup2 $mixup2 --gamma $gamma --embed_type $embed_type --mixup $mixup --axis $axis --level $level --cluster_method $cluster_method --ori_weight $ori_weight --weight $weight --separate $separate --num_thread $num_thread --torch_thread $torch_thread --method $method --org $org --net $net --ndim $ndim
        embed_name=${method}_${org}_${net}_${ndim}_${embed_type}${axis}_separate${separate}_${cluster_method}_weight${weight}_${ori_weight}_${level}_mixup${mixup}_${mixup2}_gamma${gamma}
        experiment_name=${method}_${org}_${net}_${ndim}_${model_type}_${embed_type}${axis}_separate${separate}_${cluster_method}_weight${weight}_${ori_weight}_${level}_mixup${mixup}_${mixup2}_gamma${gamma}
        echo $experiment_name
        # python gemini/main_classifier.py --mixup $mixup --ndim $ndim --embed_name $embed_name --org $org --net $net --experiment_name $experiment_name
        # for seed in 1 2 3 4; do
        for seed in 1; do
            experiment_name=${method}_${org}_${net}_${ndim}_${model_type}_${embed_type}${axis}_separate${separate}_${cluster_method}_weight${weight}_${ori_weight}_${level}_mixup${mixup}_${mixup2}_gamma${gamma}_seed${seed}
            echo $experiment_name
            python gemini/main_classifier.py --seed $seed --mixup $mixup --ndim $ndim --embed_name $embed_name --org $org --net $net --experiment_name $experiment_name
        done
    done
done

# Fig 2 4 mean variance skewness kurtosis (4
for net in GeneMANIA; do
    for org in yeast mouse human; do
        if [ $org = yeast ]; then
            ndim=200
        else
            ndim=400
        fi
        for embed_type in Qm1 Qm2 Qsm3 Qsm4; do
            axis=1
            OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS python gemini/main_gemini_cluster.py --embed_type $embed_type --axis $axis --level network --cluster_method $cluster_method --separate $separate --run_mashup 0 --num_thread $num_thread --torch_thread $torch_thread --method $method --org $org --net $net --ndim $ndim
            OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS python gemini/main_gemini.py --ori_seed $ori_seed --mixup2 $mixup2 --gamma $gamma --embed_type $embed_type --mixup $mixup --axis $axis --level $level --cluster_method $cluster_method --ori_weight $ori_weight --weight $weight --separate $separate --num_thread $num_thread --torch_thread $torch_thread --method $method --org $org --net $net --ndim $ndim
            embed_name=${method}_${org}_${net}_${ndim}_${embed_type}${axis}_separate${separate}_${cluster_method}_weight${weight}_${ori_weight}_${level}_mixup${mixup}_${mixup2}_gamma${gamma}
            experiment_name=${method}_${org}_${net}_${ndim}_${model_type}_${embed_type}${axis}_separate${separate}_${cluster_method}_weight${weight}_${ori_weight}_${level}_mixup${mixup}_${mixup2}_gamma${gamma}
            echo $experiment_name
            # python gemini/main_classifier.py --mixup $mixup --ndim $ndim --embed_name $embed_name --org $org --net $net --experiment_name $experiment_name
            # for seed in 1 2 3 4; do
            for seed in 1; do
                experiment_name=${method}_${org}_${net}_${ndim}_${model_type}_${embed_type}${axis}_separate${separate}_${cluster_method}_weight${weight}_${ori_weight}_${level}_mixup${mixup}_${mixup2}_gamma${gamma}_seed${seed}
                echo $experiment_name
                python gemini/main_classifier.py --seed $seed --mixup $mixup --ndim $ndim --embed_name $embed_name --org $org --net $net --experiment_name $experiment_name
            done
        done
    done
done

# fig 5 number of mixup (30)
export CUDA_VISIBLE_DEVICES=0
OPENBLAS_NUM_THREADS=5
num_thread=4
torch_thread=5
method=gemini
model_type=NN
ori_seed=1
echo $ori_seed
gamma=0.5
cluster_method=ap
separate=35
od=4
embed_type_=Qsm
embed_type=${embed_type_}${od}
mixup=5
weight=1
ori_weight=0.5
level=network
for mixup2 in 0.03125 0.0625 0.125 0.25 0.5 1.0 2.0; do
    for net in GeneMANIA; do
        for org in yeast mouse human; do
            if [ $org = yeast ]; then
                ndim=200
            else
                ndim=400
            fi
            embed_type=Qsm4
            axis=1
            OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS python gemini/main_gemini_cluster.py --embed_type $embed_type --axis $axis --level network --cluster_method $cluster_method --separate $separate --run_mashup 0 --num_thread $num_thread --torch_thread $torch_thread --method $method --org $org --net $net --ndim $ndim
            OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS python gemini/main_gemini.py --ori_seed $ori_seed --mixup2 $mixup2 --gamma $gamma --embed_type $embed_type --mixup $mixup --axis $axis --level $level --cluster_method $cluster_method --ori_weight $ori_weight --weight $weight --separate $separate --num_thread $num_thread --torch_thread $torch_thread --method $method --org $org --net $net --ndim $ndim
            embed_name=${method}_${org}_${net}_${ndim}_${embed_type}${axis}_separate${separate}_${cluster_method}_weight${weight}_${ori_weight}_${level}_mixup${mixup}_${mixup2}_gamma${gamma}
            experiment_name=${method}_${org}_${net}_${ndim}_${model_type}_${embed_type}${axis}_separate${separate}_${cluster_method}_weight${weight}_${ori_weight}_${level}_mixup${mixup}_${mixup2}_gamma${gamma}
            echo $experiment_name
            for seed in 1 2 3 4 5; do
                experiment_name=${method}_${org}_${net}_${ndim}_${model_type}_${embed_type}${axis}_separate${separate}_${cluster_method}_weight${weight}_${ori_weight}_${level}_mixup${mixup}_${mixup2}_gamma${gamma}_seed${seed}
                echo $experiment_name
                python gemini/main_classifier.py --seed $seed --mixup $mixup --ndim $ndim --embed_name $embed_name --org $org --net $net --experiment_name $experiment_name
            done
        done
    done
done
