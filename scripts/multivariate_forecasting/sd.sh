export CUDA_VISIBLE_DEVICES=0

model_name=S_Mamba

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting/sd" ]; then
    mkdir ./logs/LongForecasting/sd
fi

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/SD \
  --data_path sd.npz \
  --model_id sd_96_12 \
  --emb1  /home/yzh/GraphMamba/order/sd/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/sd/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/sd/emb0_50.npy \
  --pro2  /home/yzh/GraphMamba/database/sd/pro0_50.npy \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 2 \
  --s_layers 2 \
  --enc_in 716 \
  --dec_in 716 \
  --c_out 716 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16\
  --learning_rate 0.0007 \
  --itr 1 \
  --d_state 32 >logs/LongForecasting/sd/12.log

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/SD \
  --data_path sd.npz \
  --model_id PEMS07_96_24 \
  --emb1  /home/yzh/GraphMamba/order/sd/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/sd/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/sd/emb0_50.npy \
  --pro2  /home/yzh/GraphMamba/database/sd/pro0_50.npy \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers 2 \
  --s_layers 2 \
  --enc_in 716 \
  --batch_size 16\
  --dec_in 716 \
  --c_out 716 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0007 \
  --itr 1 \
  --d_state 32 >logs/LongForecasting/sd/24.log

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/SD \
  --data_path sd.npz \
  --model_id sd_96_48 \
  --emb1  /home/yzh/GraphMamba/order/sd/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/sd/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/sd/emb0_50.npy \
  --pro2  /home/yzh/GraphMamba/database/sd/pro0_50.npy \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 4 \
  --s_layers 4 \
  --enc_in 716 \
  --dec_in 716 \
  --c_out 716 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --itr 1 \
  --d_state 32\
  --batch_size 16 >logs/LongForecasting/sd/48.log

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/SD \
  --data_path sd.npz \
  --model_id sd_96_96 \
  --emb1  /home/yzh/GraphMamba/order/sd/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/sd/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/sd/emb0_50.npy \
  --pro2  /home/yzh/GraphMamba/database/sd/pro0_50.npy \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --s_layers 4 \
  --enc_in 716 \
  --dec_in 716 \
  --c_out 716 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16\
  --learning_rate 0.0005 \
  --itr 1 \
  --d_state 32 >logs/LongForecasting/sd/96.log
