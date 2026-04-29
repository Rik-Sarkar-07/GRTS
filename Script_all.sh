
## For uc_merced Dataset

python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/UCMerced_LandUse --dataset uc_merced --opt adamw --lr 8e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.2 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/ucm/test1 --weight-decay 0.05 --clip-grad 1.0 --smoothing 0.02 --dropout 0.2


python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/UCMerced_LandUse --dataset uc_merced --opt adamw --lr 8e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.2 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/ucm/test2 --weight-decay 0.05 --clip-grad 1.0 --smoothing 0.02 --dropout 0.4


python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/UCMerced_LandUse --dataset uc_merced --opt adamw --lr 8e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.2 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/ucm/test3 --weight-decay 0.05 --clip-grad 1.0 --smoothing 0.02 --dropout 0.6



python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/UCMerced_LandUse --dataset uc_merced --opt adamw --lr 4e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.2 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/ucm/test4 --weight-decay 0.05 --clip-grad 1.0 --smoothing 0.02 --dropout 0.4


python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/UCMerced_LandUse --dataset uc_merced --opt adamw --lr 2e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.2 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/ucm/test5 --weight-decay 0.05 --clip-grad 1.0 --smoothing 0.02 --dropout 0.4


python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/UCMerced_LandUse --dataset uc_merced --opt adamw --lr 4e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.2 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/ucm/test6 --weight-decay 0.05 --clip-grad 1.0 --smoothing 0.01 --dropout 0.4


python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/UCMerced_LandUse --dataset uc_merced --opt adamw --lr 4e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/ucm/test7 --weight-decay 0.05 --clip-grad 1.0 --smoothing 0.01 --dropout 0.4



python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/UCMerced_LandUse --dataset uc_merced --opt adamw --lr 4e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/ucm/test8 --weight-decay 0.01 --clip-grad 1.0 --smoothing 0.01 --dropout 0.4


##  For AID Dataset

python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/AID --dataset aid --opt adamw --lr 8e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.2 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/aid/test_1 --weight-decay 0.05 --clip-grad 1.0 --smoothing 0.02 --dropout 0.2


python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/AID --dataset aid --opt adamw --lr 8e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.2 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/aid/test_2 --weight-decay 0.05 --clip-grad 1.0 --smoothing 0.02 --dropout 0.4

python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/AID --dataset aid --opt adamw --lr 8e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.2 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/aid/test_3 --weight-decay 0.05 --clip-grad 1.0 --smoothing 0.02 --dropout 0.6


python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/AID --dataset aid --opt adamw --lr 8e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.01 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/aid/test_4 --weight-decay 0.05 --clip-grad 1.0 --smoothing 0.02 --dropout 0.6


python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/AID --dataset aid --opt adamw --lr 8e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.02 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/aid/test_5 --weight-decay 0.05 --clip-grad 1.0 --smoothing 0.02 --dropout 0.6


python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/AID --dataset aid --opt adamw --lr 8e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.04 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/aid/test_6 --weight-decay 0.05 --clip-grad 1.0 --smoothing 0.02 --dropout 0.6



python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/AID --dataset aid --opt adamw --lr 1e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.04 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/aid/test_7 --weight-decay 0.05 --clip-grad 1.0 --smoothing 0.02 --dropout 0.6

python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/AID --dataset aid --opt adamw --lr 2e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.04 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/aid/test_8 --weight-decay 0.05 --clip-grad 1.0 --smoothing 0.02 --dropout 0.6


python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/AID --dataset aid --opt adamw --lr 4e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.04 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/aid/test_9 --weight-decay 0.05 --clip-grad 1.0 --smoothing 0.02 --dropout 0.6


python -m torch.distributed.launch --nproc_per_node=1 main.py --data_dir /home/others/J20250297/Sudipta/ANET/Dataset/AID --dataset aid --opt adamw --lr 2e-4 --min-lr 1e-6 --epochs 50 --sched cosine --batch-size 20 --num_workers 4 --mixup 0.8 --cutmix 1.0 --drop-path 0.02 --warmup-epochs 5 --no-amp --input_size 224 --output_dir /home/others/J20250297/Sudipta/ANET/GRTS/output/aid/test_10 --weight-decay 0.05 --clip-grad 1.0 --smoothing 0.02 --dropout 0.4
 



