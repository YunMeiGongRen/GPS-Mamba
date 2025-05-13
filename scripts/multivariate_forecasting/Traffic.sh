export CUDA_VISIBLE_DEVICES=2

model_name=S_Mamba

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting/Traffic0" ]; then
    mkdir ./logs/LongForecasting/Traffic0
fi

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/traffic \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --emb1  /home/yzh/GraphMamba/order/Traffic/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/Traffic/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/Traffic/emb0_100_SIM_Mamba.npy \
  --pro2  /home/yzh/GraphMamba/database/Traffic/pro0_100_SIM_Mamba.npy \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 0 \
  --s_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --itr 1 >logs/LongForecasting/Traffic0/96_slayer4.log

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/traffic \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --emb1  /home/yzh/GraphMamba/order/Traffic/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/Traffic/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/Traffic/emb0_100_SIM_Mamba.npy \
  --pro2  /home/yzh/GraphMamba/database/Traffic/pro0_100_SIM_Mamba.npy \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 0 \
  --s_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --itr 1 >logs/LongForecasting/Traffic0/192_slayer4.log

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/traffic \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --emb1  /home/yzh/GraphMamba/order/Traffic/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/Traffic/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/Traffic/emb0_100_SIM_Mamba.npy \
  --pro2  /home/yzh/GraphMamba/database/Traffic/pro0_100_SIM_Mamba.npy \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 0 \
  --s_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.002 \
  --train_epochs 10 \
  --itr 1 >logs/LongForecasting/Traffic0/336_slayer4.log

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/traffic \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --emb1  /home/yzh/GraphMamba/order/Traffic/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/Traffic/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/Traffic/emb0_100_SIM_Mamba.npy \
  --pro2  /home/yzh/GraphMamba/database/Traffic/pro0_100_SIM_Mamba.npy \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 0 \
  --s_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0008\
  --train_epochs 10 \
  --itr 1 >logs/LongForecasting/Traffic0/720_slayer4.log