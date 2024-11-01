amp=True
data_root=YOUR_DATA_ROOT

# network
encoder="vit_b"
decoder="zim"

# evaluation
workers=4
image_size=1024
prompt_type="point,bbox"
model_list="zim,sam"
valset="MicroMat3K"
data_type="fine,coarse"
data_list_txt="data_list.txt"
zim_weights="results/zim_vit_b_2043"
sam_weights="results/sam_vit_b_01ec64.pth"


ngpus=$(nvidia-smi --list-gpus | wc -l)
torchrun --standalone --nnodes=1 --nproc_per_node=${ngpus} script/evaluation.py \
--amp ${amp} \
--data-root ${data_root} \
--network-encoder ${encoder} \
--network-decoder ${decoder} \
--eval-workers ${workers} \
--eval-image-size ${image_size} \
--eval-prompt-type ${prompt_type} \
--eval-model-list ${model_list} \
--eval-zim-weights ${zim_weights} \
--eval-sam-weights ${sam_weights} \
--dataset-valset ${valset} \
--dataset-data-type ${data_type} \
--dataset-data-list-txt ${data_list_txt} \
