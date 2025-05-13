export CUDA_VISIBLE_DEVICES=2

model_name=S_Mamba

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting/ECL0" ]; then
    mkdir ./logs/LongForecasting/ECL0
fi

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/electricity \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --emb1  /home/yzh/GraphMamba/order/ECL/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/ECL/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/ECL/emb0.npy \
  --pro2  /home/yzh/GraphMamba/database/ECL/pro0.npy \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --s_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --d_state 16 \
  --train_epochs 10 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1 >logs/LongForecasting/ECL0/96_slayer.log
python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/electricity \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --emb1  /home/yzh/GraphMamba/order/ECL/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/ECL/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/ECL/emb0.npy \
  --pro2  /home/yzh/GraphMamba/database/ECL/pro0.npy \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --s_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --train_epochs 10 \
  --learning_rate 0.0005 \
  --itr 1 >logs/LongForecasting/ECL0/192_slayer.log
  python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/electricity \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --emb1  /home/yzh/GraphMamba/order/ECL/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/ECL/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/ECL/emb0.npy \
  --pro2  /home/yzh/GraphMamba/database/ECL/pro0.npy \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --s_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --train_epochs 10 \
  --learning_rate 0.0005 \
  --itr 1 >logs/LongForecasting/ECL0/336_slayer.log
  python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/electricity \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --emb1  /home/yzh/GraphMamba/order/ECL/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/ECL/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/ECL/emb0.npy \
  --pro2  /home/yzh/GraphMamba/database/ECL/pro0.npy \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --s_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 10 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1 >logs/LongForecasting/ECL0/720_slayer.log