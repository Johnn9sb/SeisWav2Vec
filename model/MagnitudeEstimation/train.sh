CUDA_VISIBLE_DEVICES=0 \
exec -a Johnn9_Finetune \
python train.py \
--save_path '/mnt/nas3/johnn9/mag_checkpoint/test' \
--model_opt 'w2v' \
--decoder_type 'cnn' \
--batch_size 64 \
--epochs 200 \
--lr 0.00005 \
--workers 4 \
--level 4 \
--w2v_path '/mnt/nas3/johnn9/pretrain/wav2vec2/11-54-34/checkpoints/checkpoint_46_50000.pt' \
--without_noise 'False' \

# --w2v_path '/mnt/nas3/johnn9/pretrain/11-54-34/checkpoints/checkpoint_46_50000.pt' \

# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
# --save_path "/mnt/nas3/johnn9/mag_checkpoint/test" \
# --model_opt 'magnet' \
# --batch_size 64 \
# --epochs 200 \
# --lr 0.0001 \
# --workers 4 \
# --level 4 \
# --without_noise 'True' \