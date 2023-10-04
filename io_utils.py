import os, sys, shutil
from pathlib import Path
import subprocess
import requests
import zipfile
import mimetypes
from PIL import Image
import signal
import time
import numpy as np

def make_validation_img_grid(img_folder):
    """

    find all the .jpg imgs in img_folder (template = *.jpg)
    if >=4 validation imgs, create a 2x2 grid of them
    otherwise just return the first validation img

    """
    
    # Find all validation images
    validation_imgs = sorted([f for f in os.listdir(img_folder) if f.endswith(".jpg")])

    if len(validation_imgs) < 4:
        # If less than 4 validation images, return path of the first one
        return os.path.join(img_folder, validation_imgs[0])
    else:
        # If >= 4 validation images, create 2x2 grid
        imgs = [Image.open(os.path.join(img_folder, img)) for img in validation_imgs[:4]]

        # Assuming all images are the same size, get dimensions of first image
        width, height = imgs[0].size

        # Create an empty image with 2x2 grid size
        grid_img = Image.new("RGB", (2 * width, 2 * height))

        # Paste the images into the grid
        for i in range(2):
            for j in range(2):
                grid_img.paste(imgs.pop(0), (i * width, j * height))

        # Save the new image
        grid_img_path = os.path.join(img_folder, "validation_grid.jpg")
        grid_img.save(grid_img_path)

        return grid_img_path


def run_and_kill_cmd(command, pipe_output=True):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.25)

    # Get output from stdout and stderr
    stdout, stderr = p.communicate()    
    # Print the output to stdout in the main process
    if pipe_output:
        if stdout:
            print("cmd, stdout:")
            print(stdout)
        if stderr:
            print("cmd, stderr:")
            print(stderr)

    p.send_signal(signal.SIGTERM) # Sends termination signal
    p.wait()  # Waits for process to terminate

    # Get output from stdout and stderr
    stdout, stderr = p.communicate()

    # If the process hasn't ended yet
    if p.poll() is None:  
        p.kill()  # Forcefully kill the process
        p.wait()  # Wait for the process to terminate

    # Print the output to stdout in the main process
    if pipe_output:
        if stdout:
            print("cmd done, stdout:")
            print(stdout)
        if stderr:
            print("cmd done, stderr:")
            print(stderr)


from pathlib import Path
import requests
import os
import mimetypes

def download(url, folder, filepath=None):    
    """
    Robustly download a file from a given URL to the specified folder, automatically infering the file extension.
    
    Args:
        url (str):      The URL of the file to download.
        folder (str):   The folder where the downloaded file should be saved.
        filepath (str): (Optional) The path to the downloaded file. If None, the path will be inferred from the URL.
        
    Returns:
        filepath (Path): The path to the downloaded file.

    """
    try:
        folder_path = Path(folder)
        
        if filepath is None:
            # Guess file extension from URL itself
            parsed_url_path = Path(url.split('/')[-1])
            ext = parsed_url_path.suffix
            
            # If extension is not in URL, then use Content-Type
            if not ext:
                response = requests.head(url, allow_redirects=True)
                content_type = response.headers.get('Content-Type')
                ext = mimetypes.guess_extension(content_type) or ''
            
            filename = parsed_url_path.stem + ext  # Append extension only if needed
            filepath = folder_path / filename
        
        os.makedirs(folder_path, exist_ok=True)
        
        if filepath.exists():
            print(f"{filepath} already exists, skipping download..")
            return filepath
        
        print(f"Downloading {url} to {filepath}...")
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return filepath

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def is_zip_file(file_path):
    with open(file_path, 'rb') as file:
        return file.read(4) == b'\x50\x4b\x03\x04'

import tarfile
def untar_to_folder(input_zip_path, target_folder):
    with tarfile.open(input_zip_path, "r") as tar_ref:
        for tar_info in tar_ref:
            if tar_info.name[-1] == "/" or tar_info.name.startswith("__MACOSX"):
                continue

            mt = mimetypes.guess_type(tar_info.name)
            if mt and mt[0] and mt[0].startswith("image/"):
                tar_info.name = os.path.basename(tar_info.name)
                tar_ref.extract(tar_info, target_folder)

def unzip_to_folder(zip_path, target_folder, remove_zip = True):
    """
    Unzip the .zip file to the target folder.
    """

    os.makedirs(target_folder, exist_ok=True)

    if not is_zip_file(zip_path):
        untar_to_folder(input_zip_path, target_folder)
    else:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_folder)

    if remove_zip: # remove the zip file:
        os.remove(zip_path)

def load_image_with_orientation(path, mode = "RGB"):
    image = Image.open(path)

    # Try to get the Exif orientation tag (0x0112), if it exists
    try:
        exif_data = image._getexif()
        orientation = exif_data.get(0x0112)
    except (AttributeError, KeyError, IndexError):
        orientation = None

    # Apply the orientation, if it's present
    if orientation:
        if orientation == 2:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            image = image.rotate(180)
        elif orientation == 4:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            image = image.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            image = image.rotate(-90)
        elif orientation == 7:
            image = image.rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            image = image.rotate(90)

    return image.convert(mode)

def is_image_or_txt_file(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except: 
        # check if the string filepath is a txt file
        return file_path.endswith(".txt")

def flatten_dir(root_dir):
    try:
        # Recursively find all files and move them to the root directory
        for foldername, _, filenames in os.walk(root_dir):
            for filename in filenames:
                src = os.path.join(foldername, filename)
                dst = os.path.join(root_dir, filename)
                
                # Separate filename and extension
                base_name, ext = os.path.splitext(filename)

                # Avoid overwriting an existing file in the root directory
                counter = 0
                while os.path.exists(dst):
                    counter += 1
                    dst = os.path.join(root_dir, f"{base_name}_{counter}{ext}")

                shutil.move(src, dst)

        # Remove all subdirectories
        for foldername, subfolders, _ in os.walk(root_dir, topdown=False):
            for subfolder in subfolders:
                shutil.rmtree(os.path.join(foldername, subfolder))

    except Exception as e:
        print(f"An error occurred while flattening the directory: {e}")

def clean_and_prep_image(file_path, max_n_pixels = 2048*2048):
    try:
        image = load_image_with_orientation(file_path)
        if image.size[0] * image.size[1] > max_n_pixels:
            image.thumbnail((2048, 2048), Image.LANCZOS)

        # Generate the save path
        directory, basename = os.path.dirname(file_path), os.path.basename(file_path)
        base_name, ext = os.path.splitext(basename)
        save_path = os.path.join(directory, f"{base_name}.jpg")
        image.save(save_path, quality=95)

        if file_path != save_path:
            os.remove(file_path) # remove the original file

    except Exception as e:
        print(f"An error occurred while prepping the image {file_path}: {e}")

def prep_img_dir(target_folder):
    try:
        flatten_dir(target_folder)

        # Process image files and remove all other files
        n_final_imgs = 0
        for filename in os.listdir(target_folder):
            file_path = os.path.join(target_folder, filename)

            if not is_image_or_txt_file(file_path):
                os.remove(file_path)
            else:
                clean_and_prep_image(file_path)
                n_final_imgs += 1

        print(f"Succesfully prepped {n_final_imgs} .jpg images in {target_folder}!")

    except Exception as e:
        print(f"An error occurred while prepping the image directory: {e}")


def download_and_prep_training_data(lora_training_urls, data_dir):

    for lora_url in str(lora_training_urls).split('|'):
        download(lora_url, data_dir)

    # Loop over all files in the data directory:
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if is_zip_file(filepath):
            unzip_to_folder(filepath, data_dir, remove_zip=True)
    
    # Prep the image directory:
    prep_img_dir(data_dir)



if __name__ == '__main__':
    zip_url = "https://storage.googleapis.com/public-assets-xander/Random/remove/test.zip|https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp"
    download_and_prep_training_data(zip_url, "test_folder")