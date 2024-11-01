"""
Copyright (c) 2024-present Naver Cloud Corp.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os, sys
sys.path.append(os.getcwd())

# Gradio demo, comparison SAM vs ZIM
import os
import torch
import gradio as gr
from gradio_image_prompter import ImagePrompter
import numpy as np
import cv2
from zim import zim_model_registry, ZimPredictor, ZimAutomaticMaskGenerator
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from zim.utils import show_mat_anns

def get_shortest_axis(image):
    h, w, _ = image.shape
    return h if h < w else w

def reset_image(image, prompts):
    if image is None:
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    else:
        image = image['image']
    zim_predictor.set_image(image)
    sam_predictor.set_image(image)
    prompts = dict()
    black = np.zeros(image.shape[:2], dtype=np.uint8)

    return (image, image, image, image, black, black, black, black, prompts)

def reset_example_image(image, prompts):
    if image is None:
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)

    zim_predictor.set_image(image)
    sam_predictor.set_image(image)
    prompts = dict()
    black = np.zeros(image.shape[:2], dtype=np.uint8)

    image_dict = {}
    image_dict['image'] = image
    image_dict['prompts'] = prompts

    return (image, image_dict, image, image, image, black, black, black, black, prompts)

def run_amg(image):
    zim_masks = zim_mask_generator.generate(image)
    zim_masks_vis = show_mat_anns(image, zim_masks)
    
    sam_masks = sam_mask_generator.generate(image)
    sam_masks_vis = show_mat_anns(image, sam_masks)
    
    return zim_masks_vis, sam_masks_vis
    
    
def run_model(image, prompts):
    if not prompts:
        raise gr.Error(f'Please input any point or BBox')
    point_coords = None
    point_labels = None
    boxes = None

    if "point" in prompts:
        point_coords, point_labels = [], []

        for type, pts in prompts["point"]:
            point_coords.append(pts)
            point_labels.append(type)
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)

    if "bbox" in prompts:
        boxes = prompts['bbox']
        boxes = np.array(boxes)

    if "scribble" in prompts:
        point_coords, point_labels = [], []

        for pts in prompts["scribble"]:
            point_coords.append(np.flip(pts))
            point_labels.append(1)
        if len(point_coords) == 0:
            raise gr.Error("Please input any scribbles.")
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)

    # run ZIM
    zim_mask, _, _ = zim_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=boxes,
        multimask_output=False,
    )
    zim_mask = np.squeeze(zim_mask, axis=0)
    zim_mask = np.uint8(zim_mask * 255)

    # run SAM
    sam_mask, _, _ = sam_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=boxes,
        multimask_output=False,
        )
    sam_mask = np.squeeze(sam_mask, axis=0)
    sam_mask = np.uint8(sam_mask * 255)
    
    return zim_mask, sam_mask

def reset_scribble(image, scribble, prompts):
    # scribble = dict()
    for k in prompts.keys():
        prompts[k] = []

    for k, v in scribble.items():
        scribble[k] = None

    black = np.zeros(image.shape[:1], dtype=np.uint8)

    return scribble, black, black

def update_scribble(image, scribble, prompts):
    if "point" in prompts:
        del prompts["point"]

    if "bbox" in prompts:
        del prompts["bbox"]
    
    prompts = dict() # reset prompt
    scribble_mask = scribble["layers"][0][..., -1] > 0

    scribble_coords = np.argwhere(scribble_mask)
    n_points = min(len(scribble_coords), 24)
    indices = np.linspace(0, len(scribble_coords)-1, n_points, dtype=int)
    scribble_sampled = scribble_coords[indices]

    prompts["scribble"] = scribble_sampled
    
    zim_mask, sam_mask = run_model(image, prompts)

    return zim_mask, sam_mask, prompts


def draw_point(img, pt, size, color):
    # draw circle with white boundary region
    cv2.circle(img, (int(pt[0]), int(pt[1])), int(size * 1.3), (255, 255, 255), -1)
    cv2.circle(img, (int(pt[0]), int(pt[1])), int(size * 0.9), color, -1)


def draw_images(image, mask, prompts):
    if len(prompts) == 0 or mask.shape[1] == 1:
        return image, image, image

    minor = get_shortest_axis(image)
    size = int(minor / 80)

    image = np.float32(image)

    def blending(image, mask):
        mask = np.float32(mask) / 255
        blended_image = np.zeros_like(image, dtype=np.float32)
        blended_image[:, :, :] = [108, 0, 192]
        blended_image = (image * 0.5) + (blended_image * 0.5)
    
        img_with_mask = mask[:, :, None] * blended_image + (1 - mask[:, :, None]) * image
        img_with_mask = np.uint8(img_with_mask)

        return img_with_mask

    img_with_mask = blending(image, mask)
    img_with_point = img_with_mask.copy()

    if "point" in prompts:
        for type, pts in prompts["point"]:
            if type == "Positive":
                color = (0, 0, 255)
                draw_point(img_with_point, pts, size, color)
            elif type == "Negative":
                color = (255, 0, 0)
                draw_point(img_with_point, pts, size, color)

    size = int(minor / 200)

    return (
        img,
        img_with_mask,
    )

def get_point_or_box_prompts(img, prompts):
    image, img_prompts = img['image'], img['points']
    point_prompts = []
    box_prompts = []
    for prompt in img_prompts:
        for p in range(len(prompt)):
            prompt[p] = int(prompt[p])
        if prompt[2] == 2 and prompt[5] == 3:  # box prompt
            if len(box_prompts) != 0:
                raise gr.Error("Please input only one BBox.", duration=5)
            box_prompts.append([prompt[0], prompt[1], prompt[3], prompt[4]])
        elif prompt[2] == 1 and prompt[5] == 4:  # Positive point prompt
            point_prompts.append((1, (prompt[0], prompt[1])))
        elif prompt[2] == 0 and prompt[5] == 4:  # Negative point prompt
            point_prompts.append((0, (prompt[0], prompt[1])))

    if "scribble" in prompts:
        del prompts["scribble"]

    if len(point_prompts) > 0:
        prompts['point'] = point_prompts
    elif 'point' in prompts:
        del prompts['point']

    if len(box_prompts) > 0:
        prompts['bbox'] = box_prompts
    elif 'bbox' in prompts:
        del prompts['bbox']

    zim_mask, sam_mask = run_model(image, prompts)

    return image, zim_mask, sam_mask, prompts

def get_examples():
    assets_dir = os.path.join(os.path.dirname(__file__), 'examples')
    images = os.listdir(assets_dir)
    return [os.path.join(assets_dir, img) for img in images]

if __name__ == "__main__":
    backbone = "vit_b"
    
    # load ZIM
    ckpt_mat = "results/zim_vit_b_2043"
    zim = zim_model_registry[backbone](checkpoint=ckpt_mat)
    if torch.cuda.is_available():
        zim.cuda()
    zim_predictor = ZimPredictor(zim)
    zim_mask_generator = ZimAutomaticMaskGenerator(
        zim, 
        pred_iou_thresh=0.7, 
        points_per_batch=8,
        stability_score_thresh=0.9, 
    )

    # load SAM
    ckpt_sam = "results/sam_vit_b_01ec64.pth"
    sam = sam_model_registry[backbone](checkpoint=ckpt_sam)
    if torch.cuda.is_available():
        sam.cuda()
    sam_predictor = SamPredictor(sam)
    sam_mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_batch=8,
    )
    
    with gr.Blocks() as demo:
        gr.Markdown("# <center> [Demo] ZIM: Zero-Shot Image Matting for Anything")

        prompts = gr.State(dict())
        img = gr.Image(visible=False)
        example_image = gr.Image(visible=False)
        
        with gr.Row():
            with gr.Column():
                # Point and Bbox prompt
                with gr.Tab(label="Point or Box"):
                    img_with_point_or_box = ImagePrompter(
                        label="query image", 
                        sources="upload"
                    )
                    interactions = "Left Click (Pos) | Middle/Right Click (Neg) | Press Move (Box)"
                    gr.Markdown("<h3 style='text-align: center'>[🖱️] 🌟 {} 🌟 </h3>".format(interactions))
                    run_bttn = gr.Button("Run")
                    amg_bttn = gr.Button("Automatic Mask Generation")

                # Scribble prompt
                with gr.Tab(label="Scribble"):
                    img_with_scribble = gr.ImageEditor(
                        label="Scribble", 
                        brush=gr.Brush(colors=["#00FF00"], default_size=40),
                        sources="upload", 
                        transforms=None, 
                        layers=False
                    )
                    interactions = "Press Move (Scribble)"
                    gr.Markdown("<h3 style='text-align: center'> Step 1. Select Draw button </h3>")
                    gr.Markdown("<h3 style='text-align: center'> Step 2. 🌟 {} 🌟 </h3>".format(interactions))
                    scribble_bttn = gr.Button("Run")
                    scribble_reset_bttn = gr.Button("Reset Scribbles")
                    amg_scribble_bttn = gr.Button("Automatic Mask Generation")
                
                # Example image
                gr.Examples(get_examples(), inputs=[example_image])

            # with gr.Row():
            with gr.Column():
                with gr.Tab(label="ZIM Image"):
                    img_with_zim_mask = gr.Image(
                        label="ZIM Image", 
                        interactive=False
                    )

                with gr.Tab(label="ZIM Mask"):
                    zim_mask = gr.Image(
                        label="ZIM Mask", 
                        image_mode="L", 
                        interactive=False
                    )
                with gr.Tab(label="ZIM Auto Mask"):
                    zim_amg = gr.Image(
                        label="ZIM Auto Mask", 
                        interactive=False
                    )
                    
            with gr.Column():
                with gr.Tab(label="SAM Image"):
                    img_with_sam_mask = gr.Image(
                        label="SAM image", 
                        interactive=False
                    )

                with gr.Tab(label="SAM Mask"):
                    sam_mask = gr.Image(
                        label="SAM Mask", 
                        image_mode="L", 
                        interactive=False
                    )
                    
                with gr.Tab(label="SAM Auto Mask"):
                    sam_amg = gr.Image(
                        label="SAM Auto Mask", 
                        interactive=False
                    )
                    
        example_image.change(
            reset_example_image,
            [example_image, prompts],
            [
                img,
                img_with_point_or_box,
                img_with_scribble,
                img_with_zim_mask,
                img_with_sam_mask,
                zim_amg,
                sam_amg,
                zim_mask,
                sam_mask,
                prompts,
            ]
        )

        img_with_point_or_box.upload(
            reset_image,
            [img_with_point_or_box, prompts],
            [
                img,
                img_with_scribble,
                img_with_zim_mask,
                img_with_sam_mask,
                zim_amg,
                sam_amg,
                zim_mask,
                sam_mask,
                prompts,
            ],
        )

        amg_bttn.click(
            run_amg,
            [img],
            [zim_amg, sam_amg]
        )
        amg_scribble_bttn.click(
            run_amg,
            [img],
            [zim_amg, sam_amg]
        )
        
        run_bttn.click(
            get_point_or_box_prompts,
            [img_with_point_or_box, prompts],
            [img, zim_mask, sam_mask, prompts]
        )

        zim_mask.change(
            draw_images,
            [img, zim_mask, prompts],
            [
                img, img_with_zim_mask, 
            ],
        )
        sam_mask.change(
            draw_images,
            [img, sam_mask, prompts],
            [
                img, img_with_sam_mask, 
            ],
        )
        
        scribble_reset_bttn.click(
            reset_scribble,
            [img, img_with_scribble, prompts],
            [img_with_scribble, zim_mask, sam_mask],
        )
        scribble_bttn.click(
            update_scribble,
            [img, img_with_scribble, prompts],
            [zim_mask, sam_mask, prompts],
        )

    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=11928,
    )