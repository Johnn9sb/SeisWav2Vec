# CUDA_VISIBLE_DEVICES=2,3 \
# exec -a Johnn9_Finetune \
# python threshold.py \
# --model_name 'phasenet' \
# --train_model 'phasenet' \
# --batch_size 64 \
# --num_workers 4 \
# --parl 'y' \
# --task 'pick' \
# --dataset 'ins' \
# --test_mode 'true' \

CUDA_VISIBLE_DEVICES=0,1,2,3 \
exec -a Johnn9_Finetune \
python threshold.py \
--model_name 'data2vecpick_tune' \
--train_model 'wav2vec2' \
--batch_size 64 \
--num_workers 4 \
--decoder_type 'cnn' \
--parl 'y' \
--task 'pick' \
--weighted_sum 'n' \
--dataset 'ins' \
# --test_mode 'true' \

