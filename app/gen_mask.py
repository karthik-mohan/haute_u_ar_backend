from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

url = "https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80"

#image = Image.open(requests.get(url, stream=True).raw)
image_path = "C:/Users/Admin/Downloads/dress1.jpg"
image = Image.open(image_path)
                   
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits.cpu()
print("here")
upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)
print(upsampled_logits.argmax(dim=1))

pred_seg = upsampled_logits.argmax(dim=1)[0]
plt.imshow(pred_seg)
import matplotlib as mpl
label_names = list(model.config.id2label)
# Create a color map with the same number of colors as your labels
# Use the updated method to get the colormap
cmap = mpl.colormaps['tab20']

# Create the figure and axes for the plot and the colorbar
fig, ax = plt.subplots()

# Display the segmentation
im = ax.imshow(pred_seg, cmap=cmap)

# Create a colorbar
cbar = fig.colorbar(im, ax=ax, ticks=range(len(label_names)))
cbar.ax.set_yticklabels(label_names)

plt.show()

# Get the number of labels
n_labels = len(label_names)

# Extract RGB values for each color in the colormap
colors = cmap.colors[:n_labels]

# Convert RGBA to RGB by omitting the Alpha value
rgb_colors = [color[:3] for color in colors]

# Create a dictionary mapping labels to RGB colors
label_to_color = dict(zip(label_names, rgb_colors))

# Display the mapping
for label, color in label_to_color.items():
    print(f"{label}: {color}")