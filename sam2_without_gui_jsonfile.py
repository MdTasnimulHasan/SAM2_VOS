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
def save_image_names_to_json(folder_path, annotation_foldername):
   
    # Iterate over all files in the folder
    for img_id in sorted(os.listdir(folder_path)):

        # Dictionary to store image data
        image_data = {}
        image_data[img_id] = {"hand": "", "obj": ""} # Initialize with empty strings or default values
        img_id_base = img_id.split('.')[0]
        img_id_json = img_id_base + '.json' 
        anntation_json_file = os.path.join(annotation_foldername, img_id_json)

        if os.path.exists(anntation_json_file ):
            os.remove(anntation_json_file)

        with open(anntation_json_file, 'w') as json_file:
            json.dump(image_data, json_file, indent=4)
    print(image_data)
    # Save the image data to a JSON file


annotation_savefolder_path = '/home/mdxuser/segment-anything-2/sam2_gui_annoatation'  # Replace with your folder path
if not os.path.exists(annotation_savefolder_path):
    os.mkdir(annotation_savefolder_path)

dataset_foldername = os.path.basename(video_dir) 
annotation_dataset_foldername = os.path.join(annotation_savefolder_path, dataset_foldername )  

# if the current annotation folder for dataset folder to be annotated exists then remove and create new one
if os.path.exists(annotation_dataset_foldername):
    shutil.rmtree(annotation_dataset_foldername)
os.mkdir(annotation_dataset_foldername)

save_image_names_to_json(video_dir, annotation_dataset_foldername)


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

### !!!! need to save the json file as it7ll be used fo showing modified image in gui
refined_data = {}
refined_img_id = frame_names[ann_frame_idx]
print(refined_img_id)

refined_data[refined_img_id] = {"hand": (out_mask_logits[0] > 0.0).cpu().numpy().tolist(), 
                      "obj": (out_mask_logits[1] > 0.0).cpu().numpy().tolist()
                      }

print(type(refined_data))
print(refined_data[refined_img_id]["hand"])

def replace_json_file(current_annotation_dataset_folder, new_segmentation_data):

    replace_json_key = list(new_segmentation_data.keys())[0]
    replace_json_filename = replace_json_key.split('.')[0] + '.json'
    # print(json_key, json_filename)
    replace_jason_filepath = os.path.join(current_annotation_dataset_folder, replace_json_filename)
    # print(replace_jason_filepath)

    if os.path.exists(replace_jason_filepath):
        os.remove(replace_jason_filepath)
    with open(replace_jason_filepath, 'w') as json_file:
        json.dump(new_segmentation_data, json_file, separators=(',', ':'), indent=None)

    
replace_json_file(annotation_dataset_foldername, refined_data)

#%%
# propagrate through the video for new inputs
# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
#%%

def get_shape(lst):
    if not isinstance(lst, list) or not lst:
        return []
    
    shape = []
    current_level = lst
    
    while isinstance(current_level, list):
        shape.append(len(current_level))
        if len(current_level) > 0:
            current_level = current_level[0]
        else:
            break
    
    return shape

print(len(video_segments)) # 190 from 10-199

converted_data = video_segments[10][1].tolist()
print(get_shape(converted_data))

#%%
# annotation_json_filepath
# New values for "hand" and "obj"
new_segmentation_json = {}
for propagate_idx in range (ann_frame_idx, len(frame_names), 1):
    propagate_img_id = frame_names[propagate_idx]
    new_segmentation_json[propagate_img_id] = {"hand": video_segments[propagate_idx][1].tolist(), "obj": video_segments[propagate_idx][2].tolist()} 
# print(new_segmentation_json.keys()) # dict_keys(['00010.jpg', '00011.jpg', '00012.jpg', '00013.jpg', '00014.jpg', '00015.jpg', '00016.jpg', '00017.jpg', '00018.jpg',...]
print(len(new_segmentation_json))

#%%
# load json data
with open(annotation_json_filepath, 'r') as file:
    loaded_json_data = json.load(file)
# Print the loaded data to verify
print(len(loaded_json_data))
print(loaded_json_data['00199.jpg'])
#%%
# replace json value
for img_id, new_data in new_segmentation_json.items():
    loaded_json_data[img_id] = new_data
    print(loaded_json_data[img_id])
    break
#%%
# save json value
with open(annotation_json_filepath, 'w') as file:
    json.dump(loaded_json_data, file, indent=4)
print("JSON file updated successfully.")
#%%
# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)


# %%
