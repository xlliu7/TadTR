dataset=$1
   
if [[ $dataset = thumos14 ]];then

    CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/thumos14_i3d2s_tadtr.yml --eval --resume data/thumos14/thumos14_i3d2s_tadtr_reference.pth
else
    echo "Unsupported dataset ${dataset}. Exit"
fi
