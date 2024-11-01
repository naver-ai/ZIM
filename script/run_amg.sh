img_dir="demo/examples"
save_dir="demo/amg"
model="zim,sam"

backbone="vit_b"
zim_ckpt="results/zim_vit_b_2043.pt"
sam_ckpt="results/sam_vit_b_01ec64.pth"

points_per_batch=16
pred_iou_thresh=0.7
stability_score_thresh=0.9

python script/amg.py \
--img_dir ${img_dir} \
--save_dir ${save_dir} \
--model ${model} \
--backbone ${backbone} \
--zim_ckpt ${zim_ckpt} \
--sam_ckpt ${sam_ckpt} \
--points_per_batch ${points_per_batch} \
--pred_iou_thresh ${pred_iou_thresh} \
--stability_score_thresh ${stability_score_thresh} \
