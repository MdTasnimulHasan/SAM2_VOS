#%%
import os
cwd = os.getcwd()
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import os
import math
import shutil

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
# show RGB with mask function which returns a PIL  image
def show_RGB_mask(rgb_image, mask1, mask2):
    """
    Displays an RGB image with two masks overlaid in different transparent colors.

    Parameters:
        rgb_image (PIL.Image): The RGB image to be displayed.
        mask1 (numpy.ndarray): The first binary mask (shape: [H, W]).
        mask2 (numpy.ndarray): The second binary mask (shape: [H, W]).
        returns a PIL image with mask overlapped
    """
    # Convert RGB image to RGBA
    rgb_image_rgba = rgb_image.convert('RGBA')
    
    # Create color masks with transparency
    color_mask1 = np.zeros((*mask1.shape, 4), dtype=np.uint8)
    color_mask1[mask1] = [0, 0, 255, 100]  # Red with transparency

    color_mask2 = np.zeros((*mask2.shape, 4), dtype=np.uint8)
    color_mask2[mask2] = [255, 0, 0, 100]  # Blue with transparency

    # Convert color masks to PIL images
    color_mask1_image = Image.fromarray(color_mask1, 'RGBA')
    color_mask2_image = Image.fromarray(color_mask2, 'RGBA')

    # Overlay masks onto the RGB image
    combined_image = Image.alpha_composite(rgb_image_rgba, color_mask1_image)
    combined_image = Image.alpha_composite(combined_image, color_mask2_image)
    
    # Display the result
    return combined_image


#%%


class ImageBrowser:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Browser")
        
        # Set up initial variables
        self.image_list = []
        self.current_image_index = 0
        # added
        self.video_dir = ''
        self.frame_names = []
        self.ann_frame_idx = 0


        self.hand_select_mode = False  # Mode to select hand blue dot  pixels
        self.remove_hand_mode = False  # Mode to draw remove hand red dots

        self.obj_select_mode = False  # Mode to select hand blue dot  pixels
        self.remove_obj_mode = False  # Mode to draw remove hand red dots


        self.clicked_positions = {}  # Dictionary to store clicked positions per image
        
        # Create a frame for the image and buttons
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Create the label to show the image
        self.label = tk.Label(self.image_frame)
        self.label.pack()

        # Create a label to show the image path
        self.path_label = tk.Label(self.image_frame, text="", wraplength=400)
        self.path_label.pack()

        # Create buttons for pixel selection
        self.select_hand_button = tk.Button(self.button_frame, text="Select Hand Pixel", command=self.toggle_hand_select_mode)
        self.select_hand_button.pack(pady=5)

        self.remove_hand_button = tk.Button(self.button_frame, text="Remove Hand Pixel", command=self.toggle_hand_remove_mode)
        self.remove_hand_button.pack(pady=5)

        self.select_obj_button = tk.Button(self.button_frame, text="Select Object Pixel", command=self.toggle_obj_select_mode)
        self.select_obj_button.pack(pady=5)

        self.remove_obj_button = tk.Button(self.button_frame, text="Remove Object Pixel", command=self.toggle_obj_remove_mode)
        self.remove_obj_button.pack(pady=5)

        # Create a label to show pixel coordinates
        self.coord_label = tk.Label(self.button_frame, text="", wraplength=200)
        self.coord_label.pack(pady=5)
        
        # create button for mask generation
        self.mask_generate_button = tk.Button(self.button_frame, text="Mask Generate", command=self.sam2_mask_generate)
        self.mask_generate_button.pack(pady=5)

        # create button for sam2 mask generation propagate
        self.mask_propagate_button = tk.Button(self.button_frame, text="SAM2 Propagate ", command=self.sam2_mask_gen_propagate)
        self.mask_propagate_button.pack(pady=5)

        # Create buttons and pack them in the button frame
        self.prev_button = tk.Button(self.button_frame, text="Previous", command=self.show_previous_image)
        self.prev_button.pack(pady=5)

        self.next_button = tk.Button(self.button_frame, text="Next", command=self.show_next_image)
        self.next_button.pack(pady=5)

        self.select_folder_button = tk.Button(self.button_frame, text="Select Folder", command=self.select_folder)
        self.select_folder_button.pack(pady=5)

        # Bind the left and right arrow keys
        self.root.bind("<Left>", self.show_previous_image)
        self.root.bind("<Right>", self.show_next_image)
        self.label.bind("<Button-1>", self.on_image_click)  # Bind left mouse click to the image
        self.label.bind("<Button-3>", self.on_image_right_click)  # Bind right mouse click to the image

    def toggle_hand_select_mode(self):
        self.hand_select_mode = not self.hand_select_mode
        if self.hand_select_mode:
            self.select_hand_button.config(relief=tk.SUNKEN)
        else:
            self.select_hand_button.config(relief=tk.RAISED)

    def toggle_hand_remove_mode(self):
        self.remove_hand_mode = not self.remove_hand_mode
        if self.remove_hand_mode:
            self.remove_hand_button.config(relief=tk.SUNKEN)
        else:
            self.remove_hand_button.config(relief=tk.RAISED)

    def toggle_obj_select_mode(self):
        self.obj_select_mode = not self.obj_select_mode
        if self.obj_select_mode:
            self.select_obj_button.config(relief=tk.SUNKEN)
        else:
            self.select_obj_button.config(relief=tk.RAISED)

    def toggle_obj_remove_mode(self):
        self.remove_obj_mode = not self.remove_obj_mode
        if self.remove_obj_mode:
            self.remove_obj_button.config(relief=tk.SUNKEN)
        else:
            self.remove_obj_button.config(relief=tk.RAISED)
    
    def sam2_mask_generate(self):

        # call in each time new point input to be given

        predictor.reset_state(self.inference_state)
        self.prompts = {} 
        # add the first object
        self.ann_frame_idx = self.current_image_index  # the frame index we interact with

        img_path = os.path.join(self.video_dir, self.frame_names[self.current_image_index])
        print(self.clicked_positions[img_path])
        # print(len(self.clicked_positions[img_path])) 
        # {'hand_blue': [(415, 206), (283, 180)], 'hand_red': [(428, 180), (278, 144)], 'obj_blue': [(338, 353), (405, 340)], 'obj_red': [(348, 449), (432, 433)]}
        
        # get hand data
        hand_select_list = [list(item) for item in self.clicked_positions[img_path]['hand_blue']]
        hand_remove_list = [list(item) for item in self.clicked_positions[img_path]['hand_red']]
        self.hand_points = []
        self.hand_select_remove_labels = []
        for itr in range(0, len(hand_select_list),1):
            self.hand_points.append(hand_select_list[itr])
            self.hand_select_remove_labels.append(1)
        for itr in range(0, len(hand_remove_list),1):
            self.hand_points.append(hand_remove_list[itr])
            self.hand_select_remove_labels.append(0)
        print(self.hand_points, self.hand_select_remove_labels)


        # get obj data
        obj_select_list = [list(item) for item in self.clicked_positions[img_path]['obj_blue']]
        obj_remove_list = [list(item) for item in self.clicked_positions[img_path]['obj_red']]
        self.obj_points = []
        self.obj_select_remove_labels = []
        for itr in range(0, len(obj_select_list),1):
            self.obj_points.append(obj_select_list[itr])
            self.obj_select_remove_labels.append(1)
        for itr in range(0, len(obj_remove_list),1):
            self.obj_points.append(obj_remove_list[itr])
            self.obj_select_remove_labels.append(0)
        print(self.obj_points, self.obj_select_remove_labels)


        ###
        # hand
        ###
        if len(self.hand_points):
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
            points = np.array(self.hand_points, dtype=np.float32)
            # for labels, `1` means positive click and `0` means negative click
            labels = np.array(self.hand_select_remove_labels, np.int32)
            self.prompts[ann_obj_id] = points, labels
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=self.ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

        ###
        # obj
        ###
        if len(self.obj_points):
            ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)
            points = np.array(self.obj_points, dtype=np.float32)
            # for labels, `1` means positive click and `0` means negative click
            labels = np.array(self.obj_select_remove_labels, np.int32)
            self.prompts[ann_obj_id] = points, labels
            # `add_new_points_or_box` returns masks for all objects added so far on this interacted frame
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=self.ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

        
        
        refined_img_id = self.frame_names[self.ann_frame_idx]
        img_width, img_height = Image.open(os.path.join(self.video_dir, self.frame_names[self.ann_frame_idx])).size
        print(img_width, img_height)
        init_mask_nparray = np.zeros((img_height , img_width), dtype=np.uint8)

        # print(refined_img_id)
        refined_data = {}
        refined_data[refined_img_id] = {"hand": init_mask_nparray, 
                                        "obj": init_mask_nparray
                                        }
        if len(self.hand_points) > 0 and len(self.obj_points) > 0:
            refined_data[refined_img_id]["hand"] =  (out_mask_logits[0] > 0.0).cpu().numpy()[0]
            refined_data[refined_img_id]["obj"] =   (out_mask_logits[1] > 0.0).cpu().numpy()[0]
        elif len(self.hand_points)> 0 and len(self.obj_points) == 0:
            refined_data[refined_img_id]["hand"] =  (out_mask_logits[0] > 0.0).cpu().numpy()[0]
        elif len(self.hand_points) == 0 and len(self.obj_points) > 0:
            refined_data[refined_img_id]["obj"] =   (out_mask_logits[0] > 0.0).cpu().numpy()[0]

        refined_mask_id = os.path.splitext(refined_img_id)[0] + '.png'
        hand_mask_nparray = (refined_data[refined_img_id]["hand"]* 255).astype(np.uint8)
        hand_mask_binary_image = Image.fromarray(hand_mask_nparray)
        hand_mask_binary_image.save(os.path.join(self.hand_mask_folderpath, refined_mask_id))


        obj_mask_nparray = (refined_data[refined_img_id]["obj"]* 255).astype(np.uint8)
        obj_mask_binary_image = Image.fromarray(obj_mask_nparray)
        obj_mask_binary_image.save(os.path.join(self.obj_mask_folderpath, refined_mask_id))

        if self.frame_names:
            self.show_image()

    def sam2_mask_gen_propagate(self):

        self.ann_frame_idx = self.current_image_index  # the frame index we interact with

        img_path = os.path.join(self.video_dir, self.frame_names[self.current_image_index])
        print(self.clicked_positions[img_path])  

        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        for _, key in enumerate(video_segments.keys()):

            refined_img_id = self.frame_names[key]
            img_width, img_height = Image.open(os.path.join(self.video_dir, refined_img_id )).size
            print(img_width, img_height)
            init_mask_nparray = np.zeros((img_height , img_width), dtype=np.uint8)

            # print(refined_img_id)
            refined_data = {}
            refined_data[refined_img_id] = {"hand": init_mask_nparray, 
                                            "obj": init_mask_nparray
                                            }

            if len(self.hand_points) > 0 and len(self.obj_points) > 0:
                refined_data[refined_img_id]["hand"] =  video_segments[key][1]
                refined_data[refined_img_id]["obj"] =   video_segments[key][2]
            elif len(self.hand_points)> 0 and len(self.obj_points) == 0:
                refined_data[refined_img_id]["hand"] =  video_segments[key][1]
            elif len(self.hand_points) == 0 and len(self.obj_points) > 0:
                refined_data[refined_img_id]["obj"] =   video_segments[key][2]


            refined_mask_id = os.path.splitext(refined_img_id)[0] + '.png'

            hand_mask_nparray = (refined_data[refined_img_id]["hand"]* 255).astype(np.uint8)
            hand_mask_binary_image = Image.fromarray(hand_mask_nparray[0])
            hand_mask_binary_image.save(os.path.join(self.hand_mask_folderpath, refined_mask_id))


            obj_mask_nparray = (refined_data[refined_img_id]["obj"]* 255).astype(np.uint8)
            obj_mask_binary_image = Image.fromarray(obj_mask_nparray[0])
            obj_mask_binary_image.save(os.path.join(self.obj_mask_folderpath, refined_mask_id))

        if self.frame_names:
            self.show_image()

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        self.video_dir = folder_path 

        if folder_path:
            self.image_list = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))])
            
            # added
            # print(self.image_list)
            self.frame_names = [
                p for p in os.listdir(self.video_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
            self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            print(self.frame_names)
            print(self.image_list)

            self.current_image_index = 0
            self.ann_frame_idx = self.current_image_index

            # added
            self.annotation_savefolder_path = '/home/mdxuser/segment-anything-2/sam2_gui_annoatation'  # Replace with your folder path
            if not os.path.exists(self.annotation_savefolder_path):
                os.mkdir(self.annotation_savefolder_path)

            self.dataset_foldername = os.path.basename(self.video_dir) 
            self.annotation_dataset_foldername = os.path.join(self.annotation_savefolder_path, self.dataset_foldername)  

            # if the current annotation folder for dataset folder to be annotated exists then remove and create new one
            if os.path.exists(self.annotation_dataset_foldername):
                shutil.rmtree(self.annotation_dataset_foldername)
            os.mkdir(self.annotation_dataset_foldername)

            # hand mask directory
            self.hand_mask_folderpath = os.path.join(self.annotation_dataset_foldername, 'hand_mask')
            if not os.path.exists(self.hand_mask_folderpath):
                os.mkdir(self.hand_mask_folderpath)

            # obj mask directory
            self.obj_mask_folderpath = os.path.join(self.annotation_dataset_foldername, 'obj_mask')
            if not os.path.exists(self.obj_mask_folderpath):
                os.mkdir(self.obj_mask_folderpath)

            img_width, img_height = Image.open(os.path.join(self.video_dir, self.frame_names[0])).size
            # print(img_width, img_height)
            self.initial_mask_array = np.zeros((img_height , img_width), dtype=np.uint8)
            # Convert the NumPy array to a PIL Image
            self.initial_mask_binary_img = Image.fromarray(self.initial_mask_array)

            for init_idx in range(0, len(self.frame_names), 1):
                init_mask_id = self.frame_names[init_idx]
                init_mask_png_filename = os.path.splitext(init_mask_id)[0] + '.png'

                self.initial_mask_binary_img.save(os.path.join(self.hand_mask_folderpath, init_mask_png_filename))
                self.initial_mask_binary_img.save(os.path.join(self.obj_mask_folderpath, init_mask_png_filename))

            # initiate inference state after for browser folder
            self.inference_state = predictor.init_state(video_path=self.video_dir)

            self.clicked_positions = {}  # Reset clicked positions when a new folder is selected
            
            # if self.image_list:
            #    self.show_image()
            if self.frame_names:
                self.show_image()

    def show_image(self):
        # if self.image_list:
        if self.frame_names:

            # img_path = self.image_list[self.current_image_index]
            img_path = os.path.join(self.video_dir, self.frame_names[self.current_image_index])

            self.original_img = Image.open(img_path).convert('RGB')  # Keep the original image

            current_img_id = self.frame_names[self.current_image_index]
            current_mask_id = os.path.splitext(current_img_id)[0] + '.png'

            self.hand_mask_path = os.path.join(self.hand_mask_folderpath , current_mask_id)
            self.obj_mask_path = os.path.join(self.obj_mask_folderpath , current_mask_id)
            
            self.hand_mask_loaded = Image.open(self.hand_mask_path).convert('L')
            self.obj_mask_loaded = Image.open(self.obj_mask_path).convert('L')
            # print(type(hand_mask_loaded)) # <class 'PIL.Image.Image'>

            # Convert masks to numpy arrays
            self.hand_mask_array = np.array(self.hand_mask_loaded) > 0
            self.obj_mask_array = np.array(self.obj_mask_loaded) > 0
            # print(type(self.hand_mask_array), self.hand_mask_array.shape) # <class 'numpy.ndarray'> (540, 960)
            
            # load the RGB image with mask
            self.img_RGB_with_mask = show_RGB_mask(self.original_img, self.hand_mask_array, self.obj_mask_array)

            # self.original_img = self.original_img.resize((512, 512), Image.LANCZOS)

            self.img = self.img_RGB_with_mask.copy()
            # self.img = self.original_img.copy()  # Work with a copy of the original image
            self.tk_img = ImageTk.PhotoImage(self.img)
            self.label.config(image=self.tk_img)
            self.label.image = self.tk_img
            self.path_label.config(text=img_path)  # Update the path label with the current image path
            
            # Clear dots for new image and redraw
            self.clicked_positions[img_path] = {'hand_blue': [], 'hand_red': [], 'obj_blue': [], 'obj_red': []}
            self.redraw_clicked_positions()

    def draw_star(self, draw, center, color, size=10):
        """Draw a star on the image at the specified center position."""
        x, y = center
        points = []
        for i in range(5):
            angle = math.radians(i * 144)  # 144 degrees between points
            px = x + size * math.cos(angle)
            py = y - size * math.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=color)

    def redraw_clicked_positions(self):
        # if not self.image_list:
        #    return
        if not self.frame_names:
            return
        
        # img_path = self.image_list[self.current_image_index]
        # img_with_dots = self.original_img.copy()

        # changed
        img_path = os.path.join(self.video_dir, self.frame_names[self.current_image_index])
        img_with_dots = self.img_RGB_with_mask.copy()
        img_draw = ImageDraw.Draw(img_with_dots)

        # Draw all stored hand_blue positions
        for (x, y) in self.clicked_positions[img_path]['hand_blue']:
            # img_draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="blue")
            self.draw_star(img_draw, (x, y), color="blue")

        # Draw all stored hand_red positions
        for (x, y) in self.clicked_positions[img_path]['hand_red']:
            # img_draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red")
            self.draw_star(img_draw, (x, y), color="red")

        # Draw all stored obj_blue positions
        for (x, y) in self.clicked_positions[img_path]['obj_blue']:
            img_draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="blue")
            # self.draw_star(img_draw, (x, y), color="blue")

        # Draw all stored obj_red positions
        for (x, y) in self.clicked_positions[img_path]['obj_red']:
            img_draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red")
            # self.draw_star(img_draw, (x, y), color="red")

        # Update the image with all dots
        self.img = img_with_dots
        self.tk_img = ImageTk.PhotoImage(self.img)
        self.label.config(image=self.tk_img)
        self.label.image = self.tk_img

    def show_next_image(self, event=None):
        if self.image_list and self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.show_image()

    def show_previous_image(self, event=None):
        if self.image_list and self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image()

    def on_image_click(self, event):
        if self.image_list:
            x = event.x
            y = event.y

            # Convert coordinates to original image
            scale_x = self.original_img.width / self.label.winfo_width()
            scale_y = self.original_img.height / self.label.winfo_height()
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)

            img_path = self.image_list[self.current_image_index]
            
            # Initialize position lists if not already done
            if img_path not in self.clicked_positions:
                self.clicked_positions[img_path] = {'hand_blue': [], 'hand_red': [], 'obj_blue': [], 'obj_red': []}

            # Add the position to the list based on the current mode
            if self.remove_hand_mode:
                self.clicked_positions[img_path]['hand_red'].append((orig_x, orig_y))
            elif self.hand_select_mode:
                self.clicked_positions[img_path]['hand_blue'].append((orig_x, orig_y))

            elif self.remove_obj_mode:
                self.clicked_positions[img_path]['obj_red'].append((orig_x, orig_y))   
            elif self.obj_select_mode:
                self.clicked_positions[img_path]['obj_blue'].append((orig_x, orig_y))

            # Draw all dots
            self.redraw_clicked_positions()

            # Show the coordinates of all clicked pixels
            coords_text = "\n".join([f"hand_blue: ({px}, {py})" for px, py in self.clicked_positions[img_path]['hand_blue']] +
                                    [f"hand_red: ({px}, {py})" for px, py in self.clicked_positions[img_path]['hand_red']] +
                                    [f"obj_blue: ({px}, {py})" for px, py in self.clicked_positions[img_path]['obj_blue']] +
                                    [f"obj_red: ({px}, {py})" for px, py in self.clicked_positions[img_path]['obj_red']] 
                                    )
            self.coord_label.config(text=f"Clicked Coordinates:\n{coords_text}")

    def on_image_right_click(self, event):
        if self.image_list:
            img_path = self.image_list[self.current_image_index]

            if self.remove_hand_mode and self.clicked_positions.get(img_path, {}).get('hand_red'):
                # Remove the last red clicked position
                self.clicked_positions[img_path]['hand_red'].pop()
                self.redraw_clicked_positions()
                coords_text = "\n".join([f"hand_blue: ({px}, {py})" for px, py in self.clicked_positions[img_path]['hand_blue']] +
                                        [f"hand_red: ({px}, {py})" for px, py in self.clicked_positions[img_path]['hand_red']] +
                                        [f"obj_blue: ({px}, {py})" for px, py in self.clicked_positions[img_path]['obj_blue']] +
                                        [f"obj_red: ({px}, {py})" for px, py in self.clicked_positions[img_path]['obj_red']] 
                                        )
                self.coord_label.config(text=f"Clicked Coordinates:\n{coords_text}")

            elif self.hand_select_mode and self.clicked_positions.get(img_path, {}).get('hand_blue'):
                # Remove the last blue clicked position
                self.clicked_positions[img_path]['hand_blue'].pop()
                self.redraw_clicked_positions()
                coords_text = "\n".join([f"hand_blue: ({px}, {py})" for px, py in self.clicked_positions[img_path]['hand_blue']] +
                                        [f"hand_red: ({px}, {py})" for px, py in self.clicked_positions[img_path]['hand_red']] +
                                        [f"obj_blue: ({px}, {py})" for px, py in self.clicked_positions[img_path]['obj_blue']] +
                                        [f"obj_red: ({px}, {py})" for px, py in self.clicked_positions[img_path]['obj_red']] 
                                        )
                self.coord_label.config(text=f"Clicked Coordinates:\n{coords_text}")

            elif self.remove_obj_mode and self.clicked_positions.get(img_path, {}).get('obj_red'):
                # Remove the last red clicked position
                self.clicked_positions[img_path]['obj_red'].pop()
                self.redraw_clicked_positions()
                coords_text = "\n".join([f"hand_blue: ({px}, {py})" for px, py in self.clicked_positions[img_path]['hand_blue']] +
                                        [f"hand_red: ({px}, {py})" for px, py in self.clicked_positions[img_path]['hand_red']] +
                                        [f"obj_blue: ({px}, {py})" for px, py in self.clicked_positions[img_path]['obj_blue']] +
                                        [f"obj_red: ({px}, {py})" for px, py in self.clicked_positions[img_path]['obj_red']] 
                                        )
                self.coord_label.config(text=f"Clicked Coordinates:\n{coords_text}")
                
            elif self.obj_select_mode and self.clicked_positions.get(img_path, {}).get('obj_blue'):
                # Remove the last blue clicked position
                self.clicked_positions[img_path]['obj_blue'].pop()
                self.redraw_clicked_positions()
                coords_text = "\n".join([f"hand_blue: ({px}, {py})" for px, py in self.clicked_positions[img_path]['hand_blue']] +
                                        [f"hand_red: ({px}, {py})" for px, py in self.clicked_positions[img_path]['hand_red']] +
                                        [f"obj_blue: ({px}, {py})" for px, py in self.clicked_positions[img_path]['obj_blue']] +
                                        [f"obj_red: ({px}, {py})" for px, py in self.clicked_positions[img_path]['obj_red']] 
                                        )
                self.coord_label.config(text=f"Clicked Coordinates:\n{coords_text}")
# Set up the main application window
root = tk.Tk()
app = ImageBrowser(root)
root.geometry("1080x1080")  # Adjust size as needed
root.mainloop()


# %%

#%%