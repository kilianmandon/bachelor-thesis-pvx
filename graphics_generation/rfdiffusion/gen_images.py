import matplotlib.pyplot as plt
from PIL import Image
import os
import scienceplots

plt.style.use('science')

# Parameters
image_dir = 'graphics_generation/rfdiffusion/data'
image_files = [
    'trimer_rf_diff_wt.png',
    'trimer_rf_diff_3_0.png',
    'trimer_rf_diff_5_0.png'
]
labels = ['a', 'b', 'c']
figsize = (5.91, 1.5)

# Crop percentages (e.g., keep 80% width, 90% height)
p = 0.7  # percent of original width to keep
q = 0.7  # percent of original height to keep

# Function to center crop to p% width and q% height
def center_crop_percent(image, p, q):
    width, height = image.size
    new_width = int(width * p)
    new_height = int(height * q)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return image.crop((left, top, right, bottom))

# Create plot with total size
fig, axes = plt.subplots(1, 3, figsize=figsize)

for ax, img_file, label in zip(axes, image_files, labels):
    img_path = os.path.join(image_dir, img_file)
    image = Image.open(img_path)
    image = center_crop_percent(image, p, q)
    ax.imshow(image)
    ax.set_title(label, loc='left', fontsize=10, weight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig("graphics_generation/images/rfdiffusion_images.svg", dpi=600, bbox_inches='tight')
# plt.savefig("output.svg", )
plt.show()