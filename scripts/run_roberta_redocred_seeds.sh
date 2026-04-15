SEEDS=(66)
for SEED in "${SEEDS[@]}"
do
  epochs=20
  lr=2e-5
  bert_name=roberta
  P_WEIGHT=16.5
  python train.py --data_dir ./dataset/redocred \
  --transformer_type roberta \
  --model_name_or_path roberta-large \
  --train_file train_revised.json \
  --dev_file dev_revised.json \
  --test_file test_revised.json \
  --train_batch_size 4 \
  --test_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_labels 4 \
  --learning_rate ${lr} \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.06 \
  --num_train_epochs ${epochs} \
  --save_path ./checkpoint/redocred_${bert_name}_${epochs}epochs_${lr}lr_${P_WEIGHT}p_weight_${SEED}seed.pt \
  --save_name redocred_${bert_name}_${epochs}epochs_${lr}lr_${P_WEIGHT}p_weight_${SEED}seed \
  --seed ${SEED} \
  --device 0 \
  --p_weight ${P_WEIGHT} \
  --num_class 97
done
