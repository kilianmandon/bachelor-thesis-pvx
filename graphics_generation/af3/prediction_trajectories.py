import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import scienceplots

for_powerpoint = True

if for_powerpoint:
    fontsize = 14
    figsize = (7, 1.5)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": fontsize,
    })
else:
    fontsize = 12
    plt.style.use('science')
    figsize = (5.91, 1.5)

# Config
base_path = "/Users/kilianmandon/Desktop/Bachelor/bachelor-thesis-pvx/chimeraX/af3_trajectories"
timesteps = [50, 40, 5]
timestep_names = {
    50: 200,
    40: 160,
    5: 20,
}
# timesteps = [200, 160, 20]
t1_variants = ["atoms"]
margin_ratio = 0.1

row_labels_dict = {
    'basic': 'Default',
    'scaled': 'Scaled',
    'aligned': 'Scaled and \nAligned',
}


def load_image(name, timestep, variant=None):
    if variant:
        filename = f"{name}-timestep1-{variant}.png"
    else:
        filename = f"{name}-timestep{timestep}.png"
    path = os.path.join(base_path, filename)
    if os.path.exists(path):
        return mpimg.imread(path), filename
    else:
        return None, filename

def find_nonwhite_bbox(img, white_threshold=0.99):
    is_not_white = np.any(img[:, :, :3] < white_threshold, axis=-1)
    coords = np.argwhere(is_not_white)
    if coords.size == 0:
        return None
    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0)
    return top_left, bottom_right

def center_crop(img, center, size):
    h, w = img.shape[:2]
    cy, cx = center
    hh, hw = size[0] // 2, size[1] // 2
    y1 = max(cy - hh, 0)
    y2 = min(cy + hh, h)
    x1 = max(cx - hw, 0)
    x2 = min(cx + hw, w)
    return img[y1:y2, x1:x2]

def plot_rows(model_names, title=None, add_row_labels=True, top_row_labels_only=True):
    num_cols = len(timesteps) + len(t1_variants)
    num_rows = len(model_names)
    fig_height = figsize[0] * num_rows / num_cols * 1.15
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize[0], fig_height))

    if num_rows == 1:
        axes = [axes]  # Wrap for consistent indexing

    bboxes = []
    for t in timesteps:
        img, _ = load_image(model_names[-1], t)
        if img is not None:
            bbox = find_nonwhite_bbox(img)
            if bbox:
                bboxes.append(bbox)

    if bboxes:
        tops = np.array([b[0] for b in bboxes])
        bottoms = np.array([b[1] for b in bboxes])
        min_top = np.min(tops, axis=0)
        max_bottom = np.max(bottoms, axis=0)
        bbox_height = max_bottom[0] - min_top[0]
        bbox_width = max_bottom[1] - min_top[1]
        margin_h = int(bbox_height * margin_ratio)
        margin_w = int(bbox_width * margin_ratio)
        crop_height = bbox_height + 2 * margin_h
        crop_width = bbox_width + 2 * margin_w
        cy = (min_top[0] + max_bottom[0]) // 2
        cx = (min_top[1] + max_bottom[1]) // 2

    for row_idx, model_name in enumerate(model_names):
        images = []

        for t in timesteps:
            img, _ = load_image(model_name, t)
            if img is not None:
                images.append((img, f"t={timestep_names[t]}"))
            else:
                images.append((None, f"t={timestep_names[t]}"))

        for variant in reversed(t1_variants):
            img, _ = load_image(model_name, 1, variant)
            label = "t=1"
            if img is not None:
                bbox = find_nonwhite_bbox(img)
                if bbox:
                    bboxes.append(bbox)
                images.append((img, label))
            else:
                images.append((None, label))



        for col, (img, label) in enumerate(images):
            ax = axes[row_idx][col] if num_rows > 1 else axes[col]
            if img is not None:
                cropped = center_crop(img, (cy, cx), (crop_height, crop_width))
                ax.imshow(cropped)
            else:
                ax.set_facecolor("gray")
                ax.text(0.5, 0.5, "Missing", ha='center', va='center', transform=ax.transAxes)
            if (not top_row_labels_only) or row_idx == 0:
                ax.set_title(label, fontsize=fontsize)
            ax.axis("off")

        if add_row_labels:
            ax = axes[row_idx][0] if num_cols > 1 else axes[0]
            ax.text(-0.5, 0.5, row_labels_dict[model_name], va='center', ha='center', multialignment='center',
                    transform=ax.transAxes, fontsize=fontsize)
        else:
            ax = axes[row_idx][0] if num_cols > 1 else axes[0]
            # label = [r'\textbf{a}', r'\textbf{b}'][row_idx]
            label = ['a', 'b'][row_idx]
            ax.set_title(label, loc='left', fontsize=fontsize, weight='bold')

    if title:
        fig.suptitle(title, fontsize=fontsize)

    plt.tight_layout()


# Call the function with desired rows
plot_rows(['basic', 'aligned'], add_row_labels=True, top_row_labels_only=False)
save_path = 'images/modeling/af3_traj_basic_colloq.svg' if for_powerpoint else 'images/modeling/af3_traj_basic.svg'

plt.savefig(save_path, dpi=600)
# plot_rows([6, 7], add_row_labels=True, top_row_labels_only=False)
# plt.savefig('images/modeling/rfdiffusion_traj_symmetry.svg', dpi=600)