

# Or save the image
#output_image.save("output_image.jpg")
from os import device_encoding
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch
import numpy as np
import torch
import gc
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
#import pose_estimation as pe
import requests
from rembg import remove
from transformers import BlipProcessor, BlipForConditionalGeneration
import sys
import os
import subprocess
sys.path.append(
    os.path.join(os.path.dirname(__file__), "huggingface-cloth-segmentation"))

from process import load_seg_model, get_palette, generate_mask


device = 'cpu'



def initialize_and_load_models():

    checkpoint_path = 'model/cloth_segm.pth'
    net = load_seg_model(checkpoint_path, device=device)    

    return net

net = initialize_and_load_models()
palette = get_palette(4)


def run(img):

    cloth_seg = generate_mask(img, net=net, palette=palette, device=device)
    return cloth_seg

def image_caption(image_path, img_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = BlipProcessor.from_pretrained("noamrot/FuseCap")
    model = BlipForConditionalGeneration.from_pretrained("noamrot/FuseCap").to(device)
 
    raw_image = Image.open(image_path).convert('RGB')
    if img_type == "dress":
        raw_image = remove(raw_image)
        print("bg removed")
        raw_image.show
    #raw_image = img_np_no_bg
    
    text = "a picture of "
    inputs = processor(raw_image, text, return_tensors="pt").to(device)

    out = model.generate(**inputs, num_beams = 3)
    print(processor.decode(out[0], skip_special_tokens=True))
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def gen_vton(image_input, dress_input):
# Load the pre-trained model
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
    #revision="fp16",  # Or "full" to disable
        torch_dtype=torch.float32,  # Or torch.float32
    )
    image_path = image_input
    #submodule_path =  os.path.join(os.path.dirname(__file__), "huggingface-cloth-segmentation/process.py")
    
    img_open = Image.open(image_path)
#  
    run(img_open) 
    gen_mask_1 = "./huggingface-cloth-segmentation/output/alpha/1.png"
    gen_mask_2 = "./huggingface-cloth-segmentation/output/alpha/2.png"
    gen_mask_3 = "./huggingface-cloth-segmentation/output/alpha/3.png"
    print("mask_generated")
    if gen_mask_1:
         mask_path = gen_mask_1
    elif gen_mask_2:
        mask_path = gen_mask_2
    else:
        mask_path = gen_mask_3

    dress_path = dress_input
    
    image = Image.open(image_path)
    mask = Image.open(mask_path) # Convert mask to grayscale
#image = Image.open("/content/drive/MyDrive/train1/train/image/000025.jpg")
#mask = Image.open("/content/drive/MyDrive/train1/train/image/000014.jpg")# Convert mask to grayscale
#image = download_image(img_url).resize((512, 512))
#mask = download_image(mask_url).resize((512, 512))

#image = Image.open(image_path)
#mask_image = Image.open(mask_path)
    image = image.resize((512, 512))
    mask = mask.resize((512, 512))
# Define your prompt (text input)

    user_caption = image_caption(image_path, "user")
    dress_caption = image_caption(dress_path, "dress")
    print(user_caption)
    print(dress_caption)
    prompt = " a human wearing a {dress_caption} "
    neg_prompt = "{user_caption}"

# Note: `image` and `mask_image` should be PIL images.
# The mask structure is white for inpainting and black for keeping as is.
# Replace `image` and `mask_image` with your actual images.

    guidance_scale=7.5
    denoising_strength=0.9
    num_samples = 2
    generator = torch.Generator(device="cpu")  # Explicitly create a CPU generator




    images = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=image,
        mask_image=mask,
        guidance_scale=guidance_scale,
        denoising_strength=denoising_strength,
        generator=generator,
        num_images_per_prompt=num_samples,
    ).images

#Image_1 = pipe(prompt=prompt, image=image, mask_image=mask).images[0]


#images[0] # Display the image

#img = Image.open(images[0])
#img.show()
#img = Image.open(images[1])
#img.show()

#images[2].show
# Or save the image
    images[0].save("./processed_images/output_image.jpg")
    images[1].save("./processed_images/output_image_1.jpg")

    #images[2].save("output_image_2.jpg")
    #images[3].save("output_image_3.jpg")
    #images[3].save("output_image_4.jpg")
 

#if app == "__main__":
#gen_vton()
#user_image = "C:/Users/Admin/Downloads/woman.jpg"
#dress_image = "C:/Users/Admin/Downloads/dress1.jpg"
#gen_vton(user_image, dress_image)

def predict(dict, prompt):
  image =  dict['image'].convert("RGB").resize((512, 512))
  mask_image = dict['mask'].convert("RGB").resize((512, 512))
  #images = pipe(prompt=prompt, image=image, mask_image=mask_image).images
  return(images[0])
     
