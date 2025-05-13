export CUDA_VISIBLE_DEVICES=1

model_name=S_Mamba

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting/solar" ]; then
    mkdir ./logs/LongForecasting/solar
fi

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/solar \
  --data_path solar_AL.txt \
  --model_id solar_96_96 \
  --emb1  /home/yzh/GraphMamba/order/solar/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/solar/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/solar/emb0_SIM.npy \
  --pro2  /home/yzh/GraphMamba/database/solar/pro0_SIM.npy \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --s_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_state 8 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --itr 1 >logs/LongForecasting/solar/96.log

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/solar \
  --data_path solar_AL.txt \
  --model_id solar_96_192 \
  --emb1  /home/yzh/GraphMamba/order/solar/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/solar/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/solar/emb0_SIM.npy \
  --pro2  /home/yzh/GraphMamba/database/solar/pro0_SIM.npy \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --s_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_state 8 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --itr 1 >logs/LongForecasting/solar/192.log

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/solar \
  --data_path solar_AL.txt \
  --model_id solar_96_336 \
  --emb1  /home/yzh/GraphMamba/order/solar/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/solar/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/solar/emb0_SIM.npy \
  --pro2  /home/yzh/GraphMamba/database/solar/pro0_SIM.npy \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --s_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_state 8 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --itr 1 >logs/LongForecasting/solar/336.log

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/solar \
  --data_path solar_AL.txt \
  --model_id solar_96_720 \
  --emb1  /home/yzh/GraphMamba/order/solar/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/solar/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/solar/emb0_SIM.npy \
  --pro2  /home/yzh/GraphMamba/database/solar/pro0_SIM.npy \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --s_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_state 8 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --itr 1 >logs/LongForecasting/solar/720.log
