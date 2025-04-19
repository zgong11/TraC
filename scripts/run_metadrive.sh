env=$1
seg_ratio=$2
device_id=$3
config=./configs/trac_metadrive.yaml
path=./exp_metadrive
alpha=0.2
gamma=1.0
eta=0.25
kappa=0.7
safe_top_perc=0.25
safe_bottom_perc=0.0

metadrive=(MetaDrive-easydense MetaDrive-easymean MetaDrive-easysparse
            MetaDrive-mediumdense MetaDrive-mediummean MetaDrive-mediumsparse
            MetaDrive-harddense MetaDrive-hardmean MetaDrive-hardsparse)

# for env in $bulletgym; do
eval_env=OfflineMetadrive-$env-v0
dataset_path=~/.dsrl/datasets/SafeMetaDrive-$env-v0*
echo $eval_env
echo $dataset_path
for dir in $dataset_path; do
    for target_cost in 10 20 40; do
        for seed in 0 10 20; do
            echo 'alpha='$alpha 'seed='$seed
            CUDA_VISIBLE_DEVICES=$device_id python train.py --config=$config --path=$path --seed=$seed --eval_env=$eval_env --dataset_path=$dir --target_cost=$target_cost \
            --alpha=$alpha --gamma=$gamma --eta=$eta --seg_ratio=$seg_ratio --kappa=$kappa --safe_top_perc=$safe_top_perc --safe_bottom_perc=$safe_bottom_perc
        done
    done
done
# done