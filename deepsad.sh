export CUDA_VISIBLE_DEVICES=1

lr_list=(0.01 0.001 0.0001)
n_epochs_list=(100 150 200)
ae_lr_list=(0.01 0.001 0.0001)
ae_n_epochs_list=(100 150 200)

for((cnt=1;cnt<=1;cnt++))
do
    for n_epochs in "${n_epochs_list[@]}"
    do
        for ae_n_epochs in "${ae_n_epochs_list[@]}"
        do
            for lr in "${lr_list[@]}"
            do
                for ae_lr in "${ae_lr_list[@]}"
                do
                    # python src/main.py cifar10ood cifar10_LeNet /home/hzw/DGAD/Deep-SAD-PyTorch/results/DEBUG / \
                    python src/main.py PACS PACS_resnet results/CL_for_PACS_0123_456 data \
                    --lr ${lr} --n_epochs ${n_epochs} --ae_lr ${ae_lr} --ae_n_epochs ${ae_n_epochs} --seed 42 --batch_size 32 --ae_batch_size 32 --cnt ${cnt}
                done
            done
            wait
            echo ${epochs} ${pre_epochs}
        done
    done
done

# nohup ./deepsad.sh > deepsad.log 2>&1 &