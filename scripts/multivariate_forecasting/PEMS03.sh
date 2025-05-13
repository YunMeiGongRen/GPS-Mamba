export CUDA_VISIBLE_DEVICES=2

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting/PEMS030" ]; then
    mkdir ./logs/LongForecasting/PEMS030
fi

model_name=S_Mamba
# d_state = 32
python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/PEMS/PEMS03 \
  --data_path PEMS03.npz \
  --model_id PEMS03_96_12 \
  --emb1  /home/yzh/GraphMamba/order/PEMS03/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/PEMS03/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/PEMS03/emb0.npy \
  --pro2  /home/yzh/GraphMamba/database/PEMS03/pro0.npy \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 4 \
  --s_layers 4 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --itr 1 >logs/LongForecasting/PEMS030/12_slayer4.log

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/PEMS/PEMS03 \
  --data_path PEMS03.npz \
  --model_id PEMS03_96_24 \
  --emb1  /home/yzh/GraphMamba/order/PEMS03/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/PEMS03/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/PEMS03/emb0.npy \
  --pro2  /home/yzh/GraphMamba/database/PEMS03/pro0.npy \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers 4 \
  --s_layers 4 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.001 \
  --itr 1 >logs/LongForecasting/PEMS030/24_slayer4.log

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/PEMS/PEMS03 \
  --data_path PEMS03.npz \
  --model_id PEMS03_96_48 \
  --emb1  /home/yzh/GraphMamba/order/PEMS03/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/PEMS03/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/PEMS03/emb0.npy \
  --pro2  /home/yzh/GraphMamba/database/PEMS03/pro0.npy \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 4 \
  --s_layers 4 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.001 \
  --itr 1 >logs/LongForecasting/PEMS030/48_slayer4.log

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/PEMS/PEMS03 \
  --data_path PEMS03.npz \
  --model_id PEMS03_96_96 \
  --emb1  /home/yzh/GraphMamba/order/PEMS03/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/PEMS03/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/PEMS03/emb0.npy \
  --pro2  /home/yzh/GraphMamba/database/PEMS03/pro0.npy \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --s_layers 4 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0015 \
  --itr 1 >logs/LongForecasting/PEMS030/96_slayer4.log