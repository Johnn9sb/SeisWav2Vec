# CUDA_VISIBLE_DEVICES=2,3 \
# exec -a Johnn9_Finetune \
# python eval.py \
# --model_name 'phasenet' \
# --train_model 'phasenet' \
# --batch_size 64 \
# --num_workers 4 \
# --parl 'y' \
# --task 'pick' \
# --threshold 0.2 \
# --dataset 'ins' \
# --test_mode 'true' \

CUDA_VISIBLE_DEVICES=1 \
exec -a Johnn9_Finetune \
python eval.py \
--model_name 'lempick_frez_cnn' \
--train_model 'wav2vec2' \
--batch_size 1 \
--num_workers 4 \
--decoder_type 'cnn' \
--parl 'y' \
--task 'pick' \
--threshold 0.5 \
--weighted_sum 'n' \
--dataset '250up' \
# --noise_need 'false' \
# --test_mode 'true' \

