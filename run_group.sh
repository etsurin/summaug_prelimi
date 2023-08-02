for seed in 103 113 123 133 143
do
python roberta_main.py --n_sample 180 --epoch 70 --augfile summ_aug --gpu_id 4 --seed $seed
python roberta_main.py --n_sample 180 --epoch 70 --augfile aeda_aug --gpu_id 4 --seed $seed
python roberta_main.py --n_sample 180 --epoch 70 --gpu_id 4 --seed $seed
python roberta_main.py --n_sample 1350 --epoch 18 --augfile summ_aug --gpu_id 4 --seed $seed
python roberta_main.py --n_sample 1350 --epoch 18 --augfile aeda_aug --gpu_id 4 --seed $seed
python roberta_main.py --n_sample 1350 --epoch 18 --gpu_id 4 --seed $seed
python roberta_main.py --augfile summ_aug --gpu_id 4 --seed $seed
python roberta_main.py --augfile aeda_aug --gpu_id 4 --seed $seed
python roberta_main.py --gpu_id 4 --seed $seed

done  