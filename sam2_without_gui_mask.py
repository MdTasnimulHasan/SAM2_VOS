#%%
import os

cwd = os.getcwd()
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = os.path.join(cwd,"checkpoints/sam2_hiera_large.pt")
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


#%%
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

#%%
# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
cwd = os.getcwd()
video_dir = os.path.join(cwd, "notebooks/videos/bedroom")
print(video_dir )
# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
print(frame_names) # ['00000.jpg', '00001.jpg', '00002.jpg', '00003.jpg',...|

#%%
# for each new directory when opened in browser
import shutil
import json



annotation_savefolder_path = '/home/mdxuser/segment-anything-2/sam2_gui_annoatation'  # Replace with your folder path
if not os.path.exists(annotation_savefolder_path):
    os.mkdir(annotation_savefolder_path)

dataset_foldername = os.path.basename(video_dir) 
annotation_dataset_foldername = os.path.join(annotation_savefolder_path, dataset_foldername )  

# if the current annotation folder for dataset folder to be annotated exists then remove and create new one
if os.path.exists(annotation_dataset_foldername):
    shutil.rmtree(annotation_dataset_foldername)
os.mkdir(annotation_dataset_foldername)

# hand mask directory
hand_mask_folderpath = os.path.join(annotation_dataset_foldername, 'hand_mask')
if not os.path.exists(hand_mask_folderpath):
    os.mkdir(hand_mask_folderpath)

# obj mask directory
obj_mask_folderpath = os.path.join(annotation_dataset_foldername, 'obj_mask')
if not os.path.exists(obj_mask_folderpath):
    os.mkdir(obj_mask_folderpath)

img_width, img_height = Image.open(os.path.join(video_dir, frame_names[0])).size
print(img_width, img_height)
initial_mask_array = np.zeros((img_height , img_width), dtype=np.uint8)
# Convert the NumPy array to a PIL Image
initial_mask_binary_img = Image.fromarray(initial_mask_array)

for init_idx in range(0, len(frame_names), 1):
    init_mask_id = frame_names[init_idx]
    init_mask_png_filename = os.path.splitext(init_mask_id)[0] + '.png'

    initial_mask_binary_img.save(os.path.join(hand_mask_folderpath, init_mask_png_filename))
    initial_mask_binary_img.save(os.path.join(obj_mask_folderpath, init_mask_png_filename))

# initiate inference state after for browser folder
inference_state = predictor.init_state(video_path=video_dir)


#%%
# call in each time new point input to be given

predictor.reset_state(inference_state)
prompts = {} 
# add the first object
ann_frame_idx = 10  # the frame index we interact with

###
# hand
###
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a 2nd negative click at (x, y) = (275, 175) to refine the first object
# sending all clicks (and their labels) to `add_new_points_or_box`
points = np.array([[250, 300], [280, 250]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 0], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
'''
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
for i, out_obj_id in enumerate(out_obj_ids):
    # show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
'''
###
# object
###

ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

# Let's now move on to the second object we want to track (giving it object id `3`)
# with a positive click at (x, y) = (400, 150)
points = np.array([[350, 200]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels

# `add_new_points_or_box` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(10, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)

# print("out_obj_ids: ", out_obj_ids) # len 2, value [1,2]
# print((out_mask_logits[0] > 0.0).cpu().numpy())
# print((out_mask_logits[0] > 0.0).cpu().numpy().shape) #  (1, 540, 960)

### !!!! save the mask file

refined_data = {}
refined_img_id = frame_names[ann_frame_idx]
# print(refined_img_id)

refined_data[refined_img_id] = {"hand": (out_mask_logits[0] > 0.0).cpu().numpy(), 
                      "obj": (out_mask_logits[1] > 0.0).cpu().numpy()
                      }

# print(type(refined_data))
# print(type(refined_data[refined_img_id]["hand"]))

refined_mask_id = os.path.splitext(refined_img_id)[0] + '.png'
hand_mask_nparray = (refined_data[refined_img_id]["hand"]* 255).astype(np.uint8)
hand_mask_binary_image = Image.fromarray(hand_mask_nparray[0])
hand_mask_binary_image.save(os.path.join(hand_mask_folderpath, refined_mask_id))


obj_mask_nparray = (refined_data[refined_img_id]["obj"]* 255).astype(np.uint8)
obj_mask_binary_image = Image.fromarray(obj_mask_nparray[0])
obj_mask_binary_image.save(os.path.join(obj_mask_folderpath, refined_mask_id))
#%%
# propagrate through the video for new inputs
# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# save the propagated masks
# print(len(video_segments)) # 190
# print(video_segments.keys()) # dict_keys([10, 11, 12, 13, 14, 15, 16
for _, key in enumerate(video_segments.keys()):

    refined_data = {}
    refined_img_id = frame_names[key] # 00010.jpg
    refined_data[refined_img_id] = {"hand": video_segments[key][1], 
                      "obj": video_segments[key][2]
                      }

    refined_mask_id = os.path.splitext(refined_img_id)[0] + '.png'

    hand_mask_nparray = (refined_data[refined_img_id]["hand"]* 255).astype(np.uint8)
    hand_mask_binary_image = Image.fromarray(hand_mask_nparray[0])
    hand_mask_binary_image.save(os.path.join(hand_mask_folderpath, refined_mask_id))


    obj_mask_nparray = (refined_data[refined_img_id]["obj"]* 255).astype(np.uint8)
    obj_mask_binary_image = Image.fromarray(obj_mask_nparray[0])
    obj_mask_binary_image.save(os.path.join(obj_mask_folderpath, refined_mask_id))
