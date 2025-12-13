export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_PATH="/root/autodl-tmp/models/llama3.2_1b_instruct/origin_model"

#SAVE_DIR=/mnt/shared-storage-user/weixilin/MLLM/coconut/codi/outputs
SAVE_DIR="/root/autodl-tmp/simcot_ckpts/CODI/ckpts/llama1b_codi_SIM-CoT"

mkdir -p "$SAVE_DIR"

# cp scripts/train_28.20_ce_llama1b_dynamic-teacher_factor-exp_lat6.sh "$SAVE_DIR"

echo "Running SIM-CoT training on GPUs: $CUDA_VISIBLE_DEVICES"

# python train.py \
# 	--output_dir "$SAVE_DIR" \
#   	--expt_name gsm8k_llama1b_latent_baseline-decoder-debug \
# 	--logging_dir "$SAVE_DIR/logs"\
# 	--logging_steps 10 \
# 	--model_name_or_path /data2/huangxutao/projects/SIM-CoT/models/llama3.2_1b_instruct/origin_model \
# 	--data_name icot \
# 	--seed 11 \
# 	--model_max_length 512 \
# 	--per_device_train_batch_size 64 \
#   	--gradient_accumulation_steps 2 \
# 	--bf16 \
# 	--num_train_epochs 10 \
# 	--learning_rate 8e-4 \
# 	--max_grad_norm 2.0 \
# 	--use_lora True \
# 	--lora_r 128 --lora_alpha 32 --lora_init \
# 	--save_strategy "no" \
# 	--save_total_limit 1 \
# 	  --save_safetensors False \
# 	--weight_decay 0.1 \
# 	--warmup_ratio 0.03 \
# 	--lr_scheduler_type "cosine" \
# 	--do_train \
# 	--report_to none \
#    --num_latent 6 \
#    --logging_strategy "steps" \
# 	--use_prj True \
# 	--prj_dim 2048 \
# 	--prj_dropout 0.0 \
# 	--distill_loss_div_std True \
# 	--exp_mode False \
# 	--exp_data_num 200 \
# 	--remove_eos True \
# 	--distill_loss_factor 20 \
# 	--print_ref_model_stats True \
# 	--max_token_num 200 \
# 	--use_decoder True

# 使用 torchrun 进行分布式并行训练 (DDP)，nproc_per_node=5 对应 5 张卡
torchrun --nproc_per_node=8 --master_port=29500 train.py \
    --output_dir "$SAVE_DIR" \
    --expt_name gsm8k_llama1b_SIM-CoT \
    --logging_dir "$SAVE_DIR/logs" \
    --logging_steps 1 \
    --model_name_or_path "$MODEL_PATH" \
    --data_name icot \
    --seed 11 \
    --model_max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --bf16 \
    --num_train_epochs 10 \
    --learning_rate 8e-4 \
    --max_grad_norm 1.0 \
    --use_lora True \
    --lora_r 64 --lora_alpha 32 --lora_init \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --save_safetensors False \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --do_train \
    --report_to tensorboard \
    --num_latent 3 \
    --logging_strategy "steps" \
    --use_prj True \
    --prj_dim 2048 \
    --prj_dropout 0.0 \
    --distill_loss_div_std False \
    --exp_mode False \
    --remove_eos True \
    --distill_loss_factor 1.0 \
    --print_ref_model_stats True \
    --max_token_num 1024 \
    --use_decoder True \
   	--explain_loss_factor 1.0