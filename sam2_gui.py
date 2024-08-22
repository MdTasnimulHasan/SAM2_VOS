import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import os
import math

class ImageBrowser:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Browser")
        
        # Set up initial variables
        self.image_list = []
        self.current_image_index = 0

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

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]
            self.current_image_index = 0
            self.clicked_positions = {}  # Reset clicked positions when a new folder is selected
            if self.image_list:
                self.show_image()

    def show_image(self):
        if self.image_list:
            img_path = self.image_list[self.current_image_index]
            self.original_img = Image.open(img_path)  # Keep the original image
            self.original_img = self.original_img.resize((512, 512), Image.ANTIALIAS)
            self.img = self.original_img.copy()  # Work with a copy of the original image
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
        if not self.image_list:
            return

        img_path = self.image_list[self.current_image_index]
        img_with_dots = self.original_img.copy()
        img_draw = ImageDraw.Draw(img_with_dots)

        # Draw all stored hand_blue positions
        for (x, y) in self.clicked_positions[img_path]['hand_blue']:
            img_draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="blue")
            # self.draw_star(img_draw, (x, y), color="blue")

        # Draw all stored hand_red positions
        for (x, y) in self.clicked_positions[img_path]['hand_red']:
            img_draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red")
            # self.draw_star(img_draw, (x, y), color="red")

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
root.geometry("800x600")  # Adjust size as needed
root.mainloop()
'''
import numpy as np 
import pandas as pd 
import os
from matplotlib import pyplot as plt
import torch
import cv2
import sys
import argparse
sys.path.append("..")
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


model_type = "vit_t"
sam_checkpoint = 'E:/Tasnim/FingerVision/MobileSAM/MobileSAM/weights/mobile_sam.pt'

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

mask_generator = SamAutomaticMaskGenerator(mobile_sam)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # print(sorted_anns)
    # print(sorted_anns[0])
    # print(type(sorted_anns))
    for i in range(len(sorted_anns)):
        print(sorted_anns[i]['area'])
    del sorted_anns[0:2]
    print('after removing rack')
    for i in range(len(sorted_anns)):
        print(sorted_anns[i]['area'])
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    
    return img, sorted_anns


masked_img_ID = 'masked_after.jpg'
    

img_1 = cv2.imread(os.path.join(os.path.split(__file__)[0], masked_img_ID), cv2.COLOR_BGR2RGB)

# img_1 = cv2.bitwise_and(img_1, img_1, mask=foreground_mask)

plt.figure(figsize=(20,20))
plt.imshow(img_1)
plt.axis('off')
plt.show()
masks_1 = mask_generator.generate(img_1)
# print(masks_1) 
print(len(masks_1))
print(masks_1[0].keys())
plt.figure(figsize=(20,20))
plt.title("sam segmentation")
plt.imshow(img_1)
sam_seg, sorted_anns = show_anns(masks_1)
plt.axis('off')
plt.show() 
'''

