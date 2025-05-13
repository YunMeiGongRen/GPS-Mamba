export CUDA_VISIBLE_DEVICES=2

model_name=S_Mamba

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting/PEMS080" ]; then
    mkdir ./logs/LongForecasting/PEMS080
fi

# python -u run.py \
#   --is_training 1 \
#   --root_path /home/yzh/Data/PEMS/PEMS08 \
#   --data_path PEMS08.npz \
#   --model_id PEMS08_96_12 \
#   --emb1  /home/yzh/GraphMamba/order/PEMS08/emb.npy \
#   --pro1  /home/yzh/GraphMamba/order/PEMS08/pro.npy \
#   --emb2  /home/yzh/GraphMamba/database/PEMS08/emb0.npy \
#   --pro2  /home/yzh/GraphMamba/database/PEMS08/pro0.npy \
#   --model $model_name \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --pred_len 12 \
#   --e_layers 2 \
#   --s_layers 2 \
#   --enc_in 170 \
#   --dec_in 170 \
#   --c_out 170 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --use_norm 1 >logs/LongForecasting/PEMS080/12_slayer2.log

# python -u run.py \
#   --is_training 1 \
#   --root_path /home/yzh/Data/PEMS/PEMS08 \
#   --data_path PEMS08.npz \
#   --model_id PEMS08_96_24 \
#   --emb1  /home/yzh/GraphMamba/order/PEMS08/emb.npy \
#   --pro1  /home/yzh/GraphMamba/order/PEMS08/pro.npy \
#   --emb2  /home/yzh/GraphMamba/database/PEMS08/emb0.npy \
#   --pro2  /home/yzh/GraphMamba/database/PEMS08/pro0.npy \
#   --model $model_name \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --s_layers 2 \
#   --enc_in 170 \
#   --dec_in 170 \
#   --c_out 170 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --learning_rate 0.0007 \
#   --itr 1 \
#   --use_norm 1 >logs/LongForecasting/PEMS080/24_slayer2.log

# python -u run.py \
#   --is_training 1 \
#   --root_path /home/yzh/Data/PEMS/PEMS08 \
#   --data_path PEMS08.npz \
#   --model_id PEMS08_96_48 \
#   --emb1  /home/yzh/GraphMamba/order/PEMS08/emb.npy \
#   --pro1  /home/yzh/GraphMamba/order/PEMS08/pro.npy \
#   --emb2  /home/yzh/GraphMamba/database/PEMS08/emb0.npy \
#   --pro2  /home/yzh/GraphMamba/database/PEMS08/pro0.npy \
#   --model $model_name \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --pred_len 48 \
#   --e_layers 4 \
#   --s_layers 4 \
#   --enc_in 170 \
#   --dec_in 170 \
#   --c_out 170 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 256 \
#   --batch_size 16\
#   --learning_rate 0.001 \
#   --itr 1 \
#   --use_norm 1 >logs/LongForecasting/PEMS080/48_slayer4.log

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/PEMS/PEMS08 \
  --data_path PEMS08.npz \
  --model_id PEMS08_96_96 \
  --emb1  /home/yzh/GraphMamba/order/PEMS08/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/PEMS08/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/PEMS08/emb0.npy \
  --pro2  /home/yzh/GraphMamba/database/PEMS08/pro0.npy \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --s_layers 4 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --batch_size 16\
  --learning_rate 0.001\
  --itr 1 \
  --train_epochs 10 \
  --use_norm 1 >logs/LongForecasting/PEMS080/96_slayer4.log
