# Have SwinIR upsample
# Have BLIP auto caption
# Have CLIPSeg auto mask concept

import gc
import fnmatch
import mimetypes
import os
import time
import re
import shutil
import tarfile
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from zipfile import ZipFile

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
    Swin2SRForImageSuperResolution,
    Swin2SRImageProcessor,
)

MODEL_PATH = "./cache"

from io_utils import download_and_prep_training_data

def preprocess(
    working_directory,
    input_images_filetype: str,
    input_zip_path: Path,
    caption_text: str,
    mask_target_prompts: str,
    target_size: int,
    crop_based_on_salience: bool,
    use_face_detection_instead: bool,
    temp: float,
    substitution_tokens: List[str],
    left_right_flip_augmentation: bool = False,
) -> Path:

    # clear TEMP_IN_DIR first.
    TEMP_IN_DIR = os.path.join(working_directory,  "images_in")
    TEMP_OUT_DIR = os.path.join(working_directory, "images_out")

    for path in [TEMP_OUT_DIR, TEMP_IN_DIR]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    if input_images_filetype == "zip" or str(input_zip_path).endswith(".zip"):
        with ZipFile(str(input_zip_path), "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                    "__MACOSX"
                ):
                    continue
                mt = mimetypes.guess_type(zip_info.filename)
                if mt and mt[0] and mt[0].startswith("image/"):
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zip_ref.extract(zip_info, TEMP_IN_DIR)
    elif input_images_filetype == "tar" or str(input_zip_path).endswith(".tar"):
        assert str(input_zip_path).endswith(
            ".tar"
        ), "files must be a tar file if not zip"
        with tarfile.open(input_zip_path, "r") as tar_ref:
            for tar_info in tar_ref:
                if tar_info.name[-1] == "/" or tar_info.name.startswith("__MACOSX"):
                    continue

                mt = mimetypes.guess_type(tar_info.name)
                if mt and mt[0] and mt[0].startswith("image/"):
                    tar_info.name = os.path.basename(tar_info.name)
                    tar_ref.extract(tar_info, TEMP_IN_DIR)
    else:
        assert False, "input_images_filetype must be zip or tar"

    #download_and_prep_training_data(input_zip_path, TEMP_IN_DIR)

    output_dir: str = TEMP_OUT_DIR

    n_training_imgs, trigger_text, segmentation_prompt = load_and_save_masks_and_captions(
        files=TEMP_IN_DIR,
        output_dir=output_dir,
        caption_text=caption_text,
        mask_target_prompts=mask_target_prompts,
        target_size=target_size,
        crop_based_on_salience=crop_based_on_salience,
        use_face_detection_instead=use_face_detection_instead,
        temp=temp,
        substitution_tokens=substitution_tokens,
        add_lr_flips = left_right_flip_augmentation,
    )

    return Path(TEMP_OUT_DIR), n_training_imgs, trigger_text, segmentation_prompt


@torch.no_grad()
@torch.cuda.amp.autocast()
def swin_ir_sr(
    images: List[Image.Image],
    model_id: Literal[
        "caidas/swin2SR-classical-sr-x2-64",
        "caidas/swin2SR-classical-sr-x4-48",
        "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    ] = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    target_size: Optional[Tuple[int, int]] = None,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    **kwargs,
) -> List[Image.Image]:
    """
    Upscales images using SwinIR. Returns a list of PIL images.
    If the image is already larger than the target size, it will not be upscaled
    and will be returned as is.

    """

    model = Swin2SRForImageSuperResolution.from_pretrained(
        model_id, cache_dir=MODEL_PATH
    ).to(device)
    processor = Swin2SRImageProcessor()

    out_images = []

    for image in tqdm(images):
        ori_w, ori_h = image.size
        if target_size is not None:
            if ori_w >= target_size[0] and ori_h >= target_size[1]:
                out_images.append(image)
                continue

        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        output = (
            outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        )
        output = np.moveaxis(output, source=0, destination=-1)
        output = (output * 255.0).round().astype(np.uint8)
        output = Image.fromarray(output)

        out_images.append(output)

    return out_images


@torch.no_grad()
@torch.cuda.amp.autocast()
def clipseg_mask_generator(
    images: List[Image.Image],
    target_prompts: Union[List[str], str],
    model_id: Literal[
        "CIDAS/clipseg-rd64-refined", "CIDAS/clipseg-rd16"
    ] = "CIDAS/clipseg-rd64-refined",
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    bias: float = 0.01,
    temp: float = 1.0,
    **kwargs,
) -> List[Image.Image]:
    """
    Returns a greyscale mask for each image, where the mask is the probability of the target prompt being present in the image
    """

    if isinstance(target_prompts, str):
        print(
            f'Warning: only one target prompt "{target_prompts}" was given, so it will be used for all images'
        )

        target_prompts = [target_prompts] * len(images)

    processor = CLIPSegProcessor.from_pretrained(model_id, cache_dir=MODEL_PATH)
    model = CLIPSegForImageSegmentation.from_pretrained(
        model_id, cache_dir=MODEL_PATH
    ).to(device)

    masks = []

    for image, prompt in tqdm(zip(images, target_prompts)):
        original_size = image.size

        inputs = processor(
            text=[prompt, ""],
            images=[image] * 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits / temp, dim=0)[0]
        probs = (probs + bias).clamp_(0, 1)
        probs = 255 * probs / probs.max()

        # make mask greyscale
        mask = Image.fromarray(probs.cpu().numpy()).convert("L")

        # resize mask to original size
        mask = mask.resize(original_size)

        masks.append(mask)

    return masks


import re

# Define a function for case-insensitive text replacement
def case_insensitive_replace(text, target, replacement):
    pattern = re.compile(re.escape(target), re.IGNORECASE)
    return pattern.sub(replacement, text)

import openai
from dotenv import load_dotenv
load_dotenv()  # This will load variables from .env file into the environment
openai.api_key = os.getenv("OPENAI_API_KEY")

def cleanup_prompts_with_chatgpt(
    prompts, 
    max_tokens = 70,
    chatgpt_mode = "chat-completion",
    verbose = True):

    chat_gpt_prompt_1 = """
        I have a set of images, each containing the same concept / figure / person. I've used an img2txt model to automatically create a description for each image:
        """
    
    chat_gpt_prompt_2 = """
            I need to fix these descriptions so they sound natural and each one contains the name or fixed description [Concept Name] of the concept. I want you to:
            1. Find a good, short name/description of the concept to be learned (1-5 words). This [Concept Name] is likely already present in the auto-captions above, pick the most obvious name or words to describe the concept that's depicted in all the images.
            2. Insert the [Concept Name] (also prepend "TOK, ") into the descriptions above by rephrasing them where needed to naturally contain the string "TOK, [Concept Name]" while keeping as much of the description as possible.

            Reply by first stating the "Concept Name:", then a bullet point list of all the adjusted "Descriptions:".
        """

    final_chatgpt_prompt = chat_gpt_prompt_1 + "\n- " + "\n- ".join(prompts) + "\n\n" + chat_gpt_prompt_2
    print("Final chatgpt prompt:")
    print(final_chatgpt_prompt)

    response = openai.ChatCompletion.create(
        #model="gpt-3.5-turbo",
        model="gpt-4",
        messages=[
                #{"role": "system", "content": settings.system_description},
                {"role": "user", "content": final_chatgpt_prompt},
            ], 
        #max_tokens=max_tokens,
    )
    gpt_completion = response.choices[0].message.content

    if verbose: # pretty print the full response json:
        print(gpt_completion)
    
    # extract the Concept Name from the response:
    gpt_concept_name = ""
    for line in gpt_completion.split("\n"):
        if line.startswith("Concept Name:"):
            gpt_concept_name = line[14:]
            break

    # extract the final rephrased prompts from the response:
    prompts = []
    for line in gpt_completion.split("\n"):
        if line.startswith("-"):
            prompts.append(line[2:])

    # finally, prepend "TOK" to the Concept Name in each prompt:
    trigger_text = "TOK, " + gpt_concept_name
    #prompts = [case_insensitive_replace(prompt, gpt_concept_name, trigger_text) for prompt in prompts]

    return prompts, gpt_concept_name, trigger_text


@torch.no_grad()
def blip_captioning_dataset(
    images: List[Image.Image],
    text: Optional[str] = None,  # caption_prefix="a cartoon of TOK, the yellow bananaman figure, " 
    model_id: Literal[
        "Salesforce/blip-image-captioning-large",
        "Salesforce/blip-image-captioning-base",
    ] = "Salesforce/blip-image-captioning-large",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    substitution_tokens: Optional[List[str]] = None,
    **kwargs,
) -> List[str]:
    """
    Returns a list of captions for the given images
    """
    processor = BlipProcessor.from_pretrained(model_id, cache_dir=MODEL_PATH)
    model = BlipForConditionalGeneration.from_pretrained(
        model_id, cache_dir=MODEL_PATH
    ).to(device)
    captions = []
    text = text.strip()

    print(f"Input captioning text: {text}")
    print("Substitution tokens:", substitution_tokens)

    for image in tqdm(images):
        inputs = processor(image, return_tensors="pt").to("cuda")
        out = model.generate(
            **inputs, max_length=150, do_sample=True, top_k=50, temperature=0.7
        )
        caption = processor.decode(out[0], skip_special_tokens=True)

        # BLIP 2 lowercases all caps tokens. This should properly replace them w/o messing up subwords. I'm sure there's a better way to do this.
        for token in substitution_tokens:
            sub_cap = " " + caption + " "
            sub_cap = sub_cap.replace(" " + token.lower() + " ", " " + token + " ")
            caption = sub_cap.strip()

        print(caption)
        captions.append(caption)

    if len(captions)>2 and len(captions)<40 and (len(text) == 0):
        # use chatgpt to auto-find a good trigger text and insert it naturally into the prompts:
        retry_count = 0
        while retry_count < 4:
            try:
                captions, gpt_concept_name, trigger_text = cleanup_prompts_with_chatgpt(captions)
                break
            except Exception as e:
                retry_count += 1
                print(f"An error occurred after try {retry_count}: {e}")
                time.sleep(1)
        else:
            gpt_concept_name, trigger_text = None, text
    else:
        if len(text) == 0:
            print("WARNING: no captioning text was given and there's too few/many prompts to do chatgpt cleanup...")

        # manually add the trigger_text:
        trigger_text = text
        captions = [trigger_text + " " + caption for caption in captions]
        gpt_concept_name = None

    print("----------------------------------")
    print("Final training captions:")
    for caption in captions:
        print(caption)

    return captions, trigger_text, gpt_concept_name


def face_mask_google_mediapipe(
    images: List[Image.Image], blur_amount: float = 0.0, bias: float = 50.0
) -> List[Image.Image]:
    """
    Returns a list of images with masks on the face parts.
    """
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.1
    )
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.1
    )

    masks = []
    for image in tqdm(images):
        image_np = np.array(image)

        # Perform face detection
        results_detection = face_detection.process(image_np)
        ih, iw, _ = image_np.shape
        if results_detection.detections:
            for detection in results_detection.detections:
                bboxC = detection.location_data.relative_bounding_box

                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )

                # make sure bbox is within image
                bbox = (
                    max(0, bbox[0]),
                    max(0, bbox[1]),
                    min(iw - bbox[0], bbox[2]),
                    min(ih - bbox[1], bbox[3]),
                )

                print(bbox)

                # Extract face landmarks
                face_landmarks = face_mesh.process(
                    image_np[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
                ).multi_face_landmarks

                # https://github.com/google/mediapipe/issues/1615
                # This was def helpful
                indexes = [
                    10,
                    338,
                    297,
                    332,
                    284,
                    251,
                    389,
                    356,
                    454,
                    323,
                    361,
                    288,
                    397,
                    365,
                    379,
                    378,
                    400,
                    377,
                    152,
                    148,
                    176,
                    149,
                    150,
                    136,
                    172,
                    58,
                    132,
                    93,
                    234,
                    127,
                    162,
                    21,
                    54,
                    103,
                    67,
                    109,
                ]

                if face_landmarks:
                    mask = Image.new("L", (iw, ih), 0)
                    mask_np = np.array(mask)

                    for face_landmark in face_landmarks:
                        face_landmark = [face_landmark.landmark[idx] for idx in indexes]
                        landmark_points = [
                            (int(l.x * bbox[2]) + bbox[0], int(l.y * bbox[3]) + bbox[1])
                            for l in face_landmark
                        ]
                        mask_np = cv2.fillPoly(
                            mask_np, [np.array(landmark_points)], 255
                        )

                    mask = Image.fromarray(mask_np)

                    # Apply blur to the mask
                    if blur_amount > 0:
                        mask = mask.filter(ImageFilter.GaussianBlur(blur_amount))

                    # Apply bias to the mask
                    if bias > 0:
                        mask = np.array(mask)
                        mask = mask + bias * np.ones(mask.shape, dtype=mask.dtype)
                        mask = np.clip(mask, 0, 255)
                        mask = Image.fromarray(mask)

                    # Convert mask to 'L' mode (grayscale) before saving
                    mask = mask.convert("L")

                    masks.append(mask)
                else:
                    # If face landmarks are not available, add a black mask of the same size as the image
                    masks.append(Image.new("L", (iw, ih), 255))

        else:
            print("No face detected, adding full mask")
            # If no face is detected, add a white mask of the same size as the image
            masks.append(Image.new("L", (iw, ih), 255))

    return masks


def _crop_to_square(
    image: Image.Image, com: List[Tuple[int, int]], resize_to: Optional[int] = None
):
    cx, cy = com
    width, height = image.size
    if width > height:
        left_possible = max(cx - height / 2, 0)
        left = min(left_possible, width - height)
        right = left + height
        top = 0
        bottom = height
    else:
        left = 0
        right = width
        top_possible = max(cy - width / 2, 0)
        top = min(top_possible, height - width)
        bottom = top + width

    image = image.crop((left, top, right, bottom))

    if resize_to:
        image = image.resize((resize_to, resize_to), Image.Resampling.LANCZOS)

    return image


def _center_of_mass(mask: Image.Image):
    """
    Returns the center of mass of the mask
    """
    x, y = np.meshgrid(np.arange(mask.size[0]), np.arange(mask.size[1]))
    mask_np = np.array(mask) + 0.01
    x_ = x * mask_np
    y_ = y * mask_np

    x = np.sum(x_) / np.sum(mask_np)
    y = np.sum(y_) / np.sum(mask_np)

    return x, y


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


def load_and_save_masks_and_captions(
    files: Union[str, List[str]],
    output_dir: str = "tmp_out",
    caption_text: Optional[str] = None,
    mask_target_prompts: Optional[Union[List[str], str]] = None,
    target_size: int = 1024,
    crop_based_on_salience: bool = True,
    use_face_detection_instead: bool = False,
    temp: float = 1.0,
    n_length: int = -1,
    substitution_tokens: Optional[List[str]] = None,
    add_lr_flips: bool = False,
):
    """
    Loads images from the given files, generates masks for them, and saves the masks and captions and upscale images
    to output dir. If mask_target_prompts is given, it will generate kinda-segmentation-masks for the prompts and save them as well.

    Example:
    >>> x = load_and_save_masks_and_captions(
                files="./data/images",
                output_dir="./data/masks_and_captions",
                caption_text="a photo of",
                mask_target_prompts="cat",
                target_size=768,
                crop_based_on_salience=True,
                use_face_detection_instead=False,
                temp=1.0,
                n_length=-1,
            )
    """
    os.makedirs(output_dir, exist_ok=True)

    # load images
    if isinstance(files, str):
        # check if it is a directory
        if os.path.isdir(files):
            # get all the .png .jpg in the directory
            files = (
                _find_files("*.png", files)
                + _find_files("*.jpg", files)
                + _find_files("*.jpeg", files)
            )

        if len(files) == 0:
            raise Exception(
                f"No files found in {files}. Either {files} is not a directory or it does not contain any .png or .jpg/jpeg files."
            )
        if n_length == -1:
            n_length = len(files)
        files = sorted(files)[:n_length]
        print(files)
        
    images = [load_image_with_orientation(file) for file in files]
    n_training_imgs = len(images)

    if add_lr_flips:
        print(f"Adding LR flips... (doubling the number of images from {n_training_imgs} to {n_training_imgs*2})")
        images = images + [image.transpose(Image.FLIP_LEFT_RIGHT) for image in images]

    # captions
    print(f"Generating {len(images)} captions...")
    captions, trigger_text, gpt_concept_name = blip_captioning_dataset(
        images, text=caption_text, substitution_tokens=substitution_tokens
    )

    if gpt_concept_name is not None and ((mask_target_prompts is None) or (mask_target_prompts == "")):
        print(f"Using GPT concept name as CLIP-segmentation prompt: {gpt_concept_name}")
        mask_target_prompts = gpt_concept_name

    if mask_target_prompts is None:
        print("Disabling CLIP-segmentation")
        mask_target_prompts = ""
        temp = 999

    print(f"Generating {len(images)} masks...")
    if not use_face_detection_instead:
        seg_masks = clipseg_mask_generator(
            images=images, target_prompts=mask_target_prompts, temp=temp
        )
    else:
        mask_target_prompts = "FACE detection was used"
        if add_lr_flips:
            print("WARNING you are applying face detection while also doing left-right flips, this might not be what you intended?")
        seg_masks = face_mask_google_mediapipe(images=images)

    # find the center of mass of the mask
    if crop_based_on_salience:
        coms = [_center_of_mass(mask) for mask in seg_masks]
    else:
        coms = [(image.size[0] / 2, image.size[1] / 2) for image in images]
    # based on the center of mass, crop the image to a square
    images = [
        _crop_to_square(image, com, resize_to=None) for image, com in zip(images, coms)
    ]

    print(f"Upscaling {len(images)} images...")

    if 0:
        # upscale all images:
        images = swin_ir_sr(images, target_size=(target_size, target_size))
    else:
        # upscale images that are smaller than target_size:
        images_to_upscale = []
        indices_to_replace = []
        for idx, image in enumerate(images):
            width, height = image.size
            if width < target_size or height < target_size:
                images_to_upscale.append(image)
                indices_to_replace.append(idx)
                
        upscaled_images = swin_ir_sr(images_to_upscale, target_size=(target_size, target_size))
        
        for i, idx in enumerate(indices_to_replace):
            images[idx] = upscaled_images[i]


    images = [
        image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        for image in images
    ]

    seg_masks = [
        _crop_to_square(mask, com, resize_to=target_size)
        for mask, com in zip(seg_masks, coms)
    ]

    data = []

    # clean TEMP_OUT_DIR first
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

    os.makedirs(output_dir, exist_ok=True)

    # iterate through the images, masks, and captions and add a row to the dataframe for each
    for idx, (image, mask, caption) in enumerate(zip(images, seg_masks, captions)):
        image_name = f"{idx}.src.png"
        mask_file = f"{idx}.mask.png"

        # save the image and mask files
        image.save(os.path.join(output_dir, image_name))
        mask.save(os.path.join(output_dir, mask_file))

        # add a new row to the dataframe with the file names and caption
        data.append(
            {"image_path": image_name, "mask_path": mask_file, "caption": caption},
        )

    df = pd.DataFrame(columns=["image_path", "mask_path", "caption"], data=data)
    # save the dataframe to a CSV file
    df.to_csv(os.path.join(output_dir, "captions.csv"), index=False)

    return n_training_imgs, trigger_text, mask_target_prompts


def _find_files(pattern, dir="."):
    """Return list of files matching pattern in a given directory, in absolute format.
    Unlike glob, this is case-insensitive.
    """

    rule = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
    return [os.path.join(dir, f) for f in os.listdir(dir) if rule.match(f)]
