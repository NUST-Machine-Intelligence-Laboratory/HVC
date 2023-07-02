gpus='0'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "Using gpus ${gpus}"
now=$(date +"%T")
echo "Current time : $now"

datapath='data/YTB/2018/train_all_frames/'
savepath='./pretrain_ckpt/'
CMD="python train_vid.py --data-path ${datapath} 
                         --output-dir ${savepath}"
$CMD

python ./core/convert_pretrain.py