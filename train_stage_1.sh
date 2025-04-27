accelerate launch --config_file='./dataloader/accelerate_config.yaml' --mixed_precision="fp16" ./train_SDXL_stage_1.py --pretrained_model_name_or_path='./checkpoint/Real_4_SDXL' --pretrained_vae_model_name_or_path='./checkpoint/VAE_FP16' --vae_precision='fp16'  --resolution=512 --random_flip --train_batch_size=16 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=6000 --learning_rate=5e-5 --max_grad_norm=1.0 --lr_scheduler="cosine" --lr_warmup_steps=300 --output_dir="train_FaithDiff_stage_1_offline" --seed=42 --allow_tf32 --noise_offset=0.02 --use_ema --validation_steps=500 --checkpointing_steps=2000 --ema_update_interval=100



