gpus='0'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "Using gpus ${gpus}"
now=$(date +"%T")
echo "Current time : $now"

datapath='data/Static/'
savepath='./pretrain_ckpt/'
CMD="python train_img.py --data-path ${datapath} 
                         --output-dir ${savepath}"
$CMD

python ./core/convert_pretrain.py