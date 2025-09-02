python -u train.py \
        --seed 42 \
        --epochs 100 \
        --batch_size 46 \
        --dataset_name UPMC_Food101 \
        --class_num 101 \
        --img_size 224 \
        > "log/train.log"