#!/bin/bash
#SBATCH --job-name=hwang
#SBATCH --partition=zen3_0512_a100x2  
#SBATCH --qos=zen3_0512_a100x2  
#SBATCH --account=p72515
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4    # Adjust CPU allocation
#SBATCH --output=/gpfs/data/fs72515/nadja_g/MMCL-ECG-CMR/logs/output_%j.out   # Standard output file (with unique job ID)
#SBATCH --error=/gpfs/data/fs72515/nadja_g/MMCL-ECG-CMR/logs/error_%j.err  
module load cuda/11.8.0-gcc-12.2.0-bplw5nu
source /home/fs72515/nadja_g/.bashrc
conda activate mae
nvidia-smi

# Basic parameters seed = [0, 101, 202, 303, 404]
seed=(0)
batch_size=(8)
accum_iter=(1)

epochs="400"
warmup_epochs="5"

# Callback parameters
patience="25"
max_delta="0.25" # for AUROC

# Model parameters
input_channels="1"
input_electrodes="12"
time_steps="2500"
model_size="tiny"
model="vit_"$model_size"_patchX"

patch_height="1"
patch_width=(100)

# Augmentation parameters
masking_blockwise="False"
mask_ratio="0.00"
mask_c_ratio="0.00"
mask_t_ratio="0.00"

jitter_sigma="0.2"
rescaling_sigma="0.5"
ft_surr_phase_noise="0.075"

drop_path=(0.2)
layer_decay=(0.5)

# Optimizer parameters
blr=(3e-6) # 3e-5 if from scratch
min_lr="0.0"
weight_decay=(0.2)

# Criterion parameters
smoothing=(0.1)

from_scratch="False"

# Data path
path="tower"
if [ "$path" = "tower" ]; then
    data_base="/gpfs/data/fs72515/nadja_g/ECGMRIProjekt/ECG_data/"
    checkpoint_base="/gpfs/data/fs72515/nadja_g/ECGMRIProjekt/checkpoints/"
else
    data_base="/vol/aimspace/projects/ukbb/cardiac/cardiac_segmentations/projects"
    checkpoint_base="/vol/aimspace/users/tuo"
fi

# Dataset parameters

data_path=$data_base"train/tensor.pt"
labels_path=$data_base"train/labels.pt"
downstream_task="classification"
nb_classes="2"

# LV
lower_bnd="0"
upper_bnd="1"
nb_classes="2"
# # RV
# lower_bnd="6"
# upper_bnd="10"
# nb_classes="4"
# # WT
# lower_bnd="24"
# upper_bnd="41"
# nb_classes="17"
# # Ecc
# lower_bnd="41"
# upper_bnd="58"
# nb_classes="17"
# # Err
# lower_bnd="58"
# upper_bnd="75"
# nb_classes="17"

# Validation unbalanced
# val_data_path=$data_base"/ecg/ecgs_val_ecg_imaging_noBase_gn.pt"
# val_labels_path=$data_base"/ecg/labelsOneHot/labels_val_flutter_all.pt"
# pos_label="1"
val_data_path=$data_base"val/tensor.pt"
val_labels_path=$data_base"val/labels.pt"
pos_label="1"
# val_data_path=$data_base"/ecg/ecgs_val_ecg_imaging_noBase_gn.pt"
# val_labels_path=$data_base"/ecg/labelsOneHot/labels_val_CAD_all.pt"
# pos_label="1"


global_pool=(True)
attention_pool=(True)
num_workers="24"

# Log specifications
save_output="True"
wandb="True"
wandb_project="MAE_ECG_Fin_Tiny_LV"
wandb_id=""

plot_attention_map="False"
plot_embeddings="False"
save_embeddings="False"
save_predictions="True"

# Pretraining specifications
pre_batch_size=(128)
pre_blr=(1e-5)

# EVALUATE
eval="False"
# As filename: State the checkpoint for the inference of a specific model
# or state the (final) epoch for the inference of all models up to this epoch
#resume=$checkpoint_base"/sprai/mae_he/mae/output/fin/"$folder"/id/"$subfolder"/fin_b"$(($batch_size*$accum_iter))"_blr"$blr"_"$pre_data"/checkpoint-89.pth"

for sd in "${seed[@]}"
do

    for bs in "${batch_size[@]}"
    do
        for ld in "${layer_decay[@]}"
        do
            for lr in "${blr[@]}"
            do

                for dp in "${drop_path[@]}"
                do 
                    for wd in "${weight_decay[@]}"
                    do
                        for smth in "${smoothing[@]}"
                        do

                            folder="ecg/LV/MM/attnPool"
                            subfolder=("seed$sd/"$model_size"/t"$time_steps"/p"$patch_height"x"$patch_width"/ld"$ld"/dp"$dp"/smth"$smth"/wd"$weight_decay"/m0.8")

                            pre_data="b"$pre_batch_size"_blr"$pre_blr
                            # finetune=$checkpoint_base"/sprai/mae_he/mae/output/pre/"$folder"/"$subfolder"/pre_"$pre_data"/checkpoint-399.pth"
                            
                            # # MAE + MMCL
                            finetune="/gpfs/data/fs72515/nadja_g/ECGMRIProjekt/EKGMRIProjekt/model_weights/signal_encoder_mmcl_wo_mdm.pth"
                            
                            # # MMCL only
                            # finetune=$checkpoint_base"/ECGMultimodalContrastiveLearning/oezguen/checkpoints/mm_v283_mae_checkpoint.pth"

                            # # MAE only
                            # finetune=$checkpoint_base"/ECGMultimodalContrastiveLearning/pretrained_checkpoints/tiny/v1/checkpoint-399.pth"
                            
                            # # BYOL 
                            # finetune=$checkpoint_base"/ECGMultimodalContrastiveLearning/oezguen/checkpoints/ecg/ecg_v204_um_checkpoint.pth"

                            # # BarlowTwins
                            # finetune=$checkpoint_base"/ECGMultimodalContrastiveLearning/oezguen/checkpoints/ecg/ecg_v305_um_checkpoint.pth"

                            # # SimCLR
                            # finetune=$checkpoint_base"/ECGMultimodalContrastiveLearning/oezguen/checkpoints/ecg/ecg_v404_um_checkpoint.pth"

                            # # CLOCS
                            # finetune=$checkpoint_base"/ECGMultimodalContrastiveLearning/oezguen/checkpoints/ecg/ecg_v150_mae_checkpoint.pth"

                            output_dir=$checkpoint_base"/weights/"
                            log_dir=$checkpoint_base

                            # resume=$checkpoint_base"/"$folder"/"$subfolder"/fin_b"$bs"_blr"$lr"_"$pre_data"/checkpoint-4-pcc-0.54.pth"

                            if [ "$downstream_task" = "regression" ]; then
                                cmd="python3 /gpfs/data/fs72515/nadja_g/MMCL-ECG-CMR/mae_ng/main_linprobe.py --lower_bnd $lower_bnd --upper_bnd $upper_bnd --seed $sd --downstream_task $downstream_task --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --patience $patience --max_delta $max_delta --accum_iter $accum_iter --drop_path $dp --weight_decay $wd --layer_decay $ld --min_lr $min_lr --blr $lr --warmup_epoch $warmup_epochs --smoothing $smth --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --nb_classes $nb_classes --log_dir $log_dir --num_workers $num_workers"
                            else
                                cmd="python3 /gpfs/data/fs72515/nadja_g/MMCL-ECG-CMR/mae_ng/main_linprobe.py --seed $sd --downstream_task $downstream_task --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --patience $patience --max_delta $max_delta --accum_iter $accum_iter --drop_path $dp --weight_decay $wd --layer_decay $ld --min_lr $min_lr --blr $lr --warmup_epoch $warmup_epochs --smoothing $smth --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --nb_classes $nb_classes --log_dir $log_dir --num_workers $num_workers"
                            fi

                            # cmd="python3 main_finetune.py --lower_bnd $lower_bnd --upper_bnd $upper_bnd --seed $sd --downstream_task $downstream_task --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --patience $patience --max_delta $max_delta --accum_iter $accum_iter --drop_path $dp --weight_decay $wd --layer_decay $ld --min_lr $min_lr --blr $lr --warmup_epoch $warmup_epochs --smoothing $smth --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --nb_classes $nb_classes --log_dir $log_dir --num_workers $num_workers"
                            # cmd="python3 main_finetune.py --seed $sd --downstream_task $downstream_task --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --patience $patience --max_delta $max_delta --accum_iter $accum_iter --drop_path $dp --weight_decay $wd --layer_decay $ld --min_lr $min_lr --blr $lr --warmup_epoch $warmup_epochs --smoothing $smth --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --nb_classes $nb_classes --log_dir $log_dir --num_workers $num_workers"

                            if [ "$masking_blockwise" = "True" ]; then
                                cmd=$cmd" --masking_blockwise --mask_c_ratio $mask_c_ratio --mask_t_ratio $mask_t_ratio"
                            fi
                            
                            if [ "$from_scratch" = "False" ]; then
                                cmd=$cmd" --finetune $finetune"
                            fi

                            if [ ! -z "$pos_label" ]; then
                                cmd=$cmd" --pos_label $pos_label"
                            fi

                            if [ ! -z "$labels_mask_path" ]; then
                                cmd=$cmd" --labels_mask_path $labels_mask_path"
                            fi

                            if [ ! -z "$val_labels_mask_path" ]; then
                                cmd=$cmd" --val_labels_mask_path $val_labels_mask_path"
                            fi

                            if [ "$global_pool" = "True" ]; then
                                cmd=$cmd" --global_pool"
                            fi

                            if [ "$attention_pool" = "True" ]; then
                                cmd=$cmd" --attention_pool"
                            fi

                            if [ "$wandb" = "True" ]; then
                                cmd=$cmd" --wandb --wandb_project $wandb_project"
                                if [ ! -z "$wandb_id" ]; then
                                    cmd=$cmd" --wandb_id $wandb_id"
                                fi
                            fi

                            if [ "$save_output" = "True" ]; then
                                cmd=$cmd" --output_dir $output_dir"
                            fi

                            if [ "$plot_attention_map" = "True" ]; then
                                cmd=$cmd" --plot_attention_map"
                            fi

                            if [ "$plot_embeddings" = "True" ]; then
                                cmd=$cmd" --plot_embeddings"
                            fi

                            if [ "$save_embeddings" = "True" ]; then
                                cmd=$cmd" --embeddings_dir $output_dir"
                            fi


                            if [ "$eval" = "True" ]; then
                                cmd=$cmd" --eval --resume $resume"
                            fi

                            if [ ! -z "$resume" ]; then
                                cmd=$cmd" --resume $resume"
                            fi
                            
                            echo $cmd && $cmd

                        done
                    done
                done

            done
        done
    done

done
