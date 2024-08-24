#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def show_RGB_mask(rgb_image, mask1, mask2):
    """
    Displays an RGB image with two masks overlaid in different transparent colors.

    Parameters:
        rgb_image (PIL.Image): The RGB image to be displayed.
        mask1 (numpy.ndarray): The first binary mask (shape: [H, W]).
        mask2 (numpy.ndarray): The second binary mask (shape: [H, W]).
        ax (matplotlib.axes.Axes): The axis on which to display the image and masks.
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

# Example usage
if __name__ == "__main__":
    # Load the RGB image
    current_img_id = '00050.jpg'
    current_mask_id = os.path.splitext(current_img_id)[0] + '.png'

    rgb_image_path = os.path.join('/home/mdxuser/segment-anything-2/notebooks/videos/bedroom', current_img_id)
    rgb_image = Image.open(rgb_image_path).convert('RGB')
    
    # Load the mask images
    hand_mask_path = os.path.join('/home/mdxuser/segment-anything-2/sam2_gui_annoatation/bedroom/hand_mask', current_mask_id)
    obj_mask_path = os.path.join('/home/mdxuser/segment-anything-2/sam2_gui_annoatation/bedroom/obj_mask', current_mask_id)
    
    hand_mask_loaded = Image.open(hand_mask_path).convert('L')
    obj_mask_loaded = Image.open(obj_mask_path).convert('L')
    print(type(hand_mask_loaded)) # <class 'PIL.Image.Image'>

    # Convert masks to numpy arrays
    hand_mask_array = np.array(hand_mask_loaded) > 0
    obj_mask_array = np.array(obj_mask_loaded) > 0
    print(type(hand_mask_array), hand_mask_array.shape) # <class 'numpy.ndarray'> (540, 960)


    # Show the RGB image with the masks
    RGB_with_mask = show_RGB_mask(rgb_image, hand_mask_array, obj_mask_array)
    print(type(RGB_with_mask))
    RGB_with_mask.show()
    
#%%