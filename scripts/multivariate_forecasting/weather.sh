export CUDA_VISIBLE_DEVICES=0

model_name=S_Mamba

if [ ! -d "./logs/LongForecasting/weather" ]; then
    mkdir ./logs/LongForecasting/weather
fi

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/weather \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --emb1  /home/yzh/GraphMamba/order/weather/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/weather/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/weather/emb0.npy \
  --pro2  /home/yzh/GraphMamba/database/weather/pro0.npy \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --s_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --learning_rate 0.00005 \
  --train_epochs 10\
  --d_state 2 \
  --d_ff 512\
  --itr 1 >logs/LongForecasting/weather/96.log


python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/weather \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --emb1  /home/yzh/GraphMamba/order/weather/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/weather/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/weather/emb0.npy \
  --pro2  /home/yzh/GraphMamba/database/weather/pro0.npy \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --s_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --learning_rate 0.00005 \
  --train_epochs 10\
  --d_model 512\
  --d_state 2 \
  --d_ff 512\
  --itr 1 >logs/LongForecasting/weather/192.log


python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/weather \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --emb1  /home/yzh/GraphMamba/order/weather/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/weather/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/weather/emb0.npy \
  --pro2  /home/yzh/GraphMamba/database/weather/pro0.npy \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --s_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --learning_rate 0.00005 \
  --train_epochs 10\
  --d_model 512\
  --d_state 2 \
  --d_ff 512\
  --itr 1 >logs/LongForecasting/weather/336.log

python -u run.py \
  --is_training 1 \
  --root_path /home/yzh/Data/weather \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --emb1  /home/yzh/GraphMamba/order/weather/emb.npy \
  --pro1  /home/yzh/GraphMamba/order/weather/pro.npy \
  --emb2  /home/yzh/GraphMamba/database/weather/emb0.npy \
  --pro2  /home/yzh/GraphMamba/database/weather/pro0.npy \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --s_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --learning_rate 0.00005 \
  --train_epochs 10\
  --d_model 512\
  --d_state 2 \
  --d_ff 512\
  --itr 1 >logs/LongForecasting/weather/720.log