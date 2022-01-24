# Create numpy tensors from RGB and thermal images
import os
import cv2
from pathlib import Path
import numpy as np

thermal_path = Path.cwd() / "data/your_path_to_thermal_images"
visible_path = Path.cwd() / "data/your_path_to_visible_images"
output_path = Path.cwd() / "data/your_path_to_newly_created_tensors"


def create_tensor(target_folder, img_path):
    thermal_array = cv2.imread(str(thermal_path) + img_path, 3)
    thermal_array = (thermal_array / np.max(thermal_array) * 255).astype('uint8')
    visible_array = cv2.imread(str(visible_path) + img_path, 3)
    tensor_array = np.concatenate((visible_array, thermal_array), axis=2)

    print(img_path)
    output_folder = target_folder / Path(img_path).name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tensor_name = target_folder / image_path[:-4] + ".npy"
    np.save(tensor_name, tensor_array)


for subfolder in sorted(visible_path.iterdir()):
    for set_folder in sorted(subfolder.iterdir()):
        for image in sorted(set_folder.iterdir()):
            image_path = "/" + "/".join(list(image.parts[-3:]))
            create_tensor(output_path, image_path)
