import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import scienceplots

plt.style.use('science')

# Config
base_path = "/Users/kilianmandon/Desktop/Bachelor/bachelor-thesis-pvx/chimeraX/rfdiffusion_trajectories"
model_ids = [8, 5, 6, 7]  # Reordered: [Symmetric Noise, Full Symmetrization, Initial Symmetrization, Unconditional]
timesteps = [50, 40, 5]
t1_variants = ["atoms"]
margin_ratio = 0.1
figsize = (5.91, 1.5)

row_labels_dict = {
    8: "Unconditional\nGeneration",
    5: "Symemtric\nNoise",
    6: "Full\nSymmetrization",
    7: "Initial\nSymmetrization"
}

def load_image(model_id, timestep, variant=None):
    if variant:
        filename = f"model{model_id}-timestep1-{variant}.png"
    else:
        filename = f"model{model_id}-timestep{timestep}.png"
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

def plot_rows(model_id_subset, title=None, add_row_labels=True, top_row_labels_only=True):
    num_cols = len(timesteps) + len(t1_variants)
    num_rows = len(model_id_subset)
    fig_height = figsize[0] * num_rows / num_cols * 1.15
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize[0], fig_height))

    if num_rows == 1:
        axes = [axes]  # Wrap for consistent indexing

    for row_idx, model_id in enumerate(model_id_subset):
        images = []
        bboxes = []

        for t in timesteps:
            img, _ = load_image(model_id, t)
            if img is not None:
                bbox = find_nonwhite_bbox(img)
                if bbox:
                    bboxes.append(bbox)
                images.append((img, f"t={t}"))
            else:
                images.append((None, f"t={t}"))

        for variant in reversed(t1_variants):
            img, _ = load_image(model_id, 1, variant)
            label = "t=1"
            if img is not None:
                bbox = find_nonwhite_bbox(img)
                if bbox:
                    bboxes.append(bbox)
                images.append((img, label))
            else:
                images.append((None, label))

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

        for col, (img, label) in enumerate(images):
            ax = axes[row_idx][col] if num_rows > 1 else axes[col]
            if img is not None:
                cropped = center_crop(img, (cy, cx), (crop_height, crop_width))
                ax.imshow(cropped)
            else:
                ax.set_facecolor("gray")
                ax.text(0.5, 0.5, "Missing", ha='center', va='center', transform=ax.transAxes)
            if (not top_row_labels_only) or row_idx == 0:
                ax.set_title(label, fontsize=12)
            ax.axis("off")

        if add_row_labels:
            ax = axes[row_idx][0] if num_cols > 1 else axes[0]
            ax.text(-0.5, 0.5, row_labels_dict[model_id], va='center', ha='center', multialignment='center',
                    transform=ax.transAxes, fontsize=12)
        else:
            ax = axes[row_idx][0] if num_cols > 1 else axes[0]
            # label = [r'\textbf{a}', r'\textbf{b}'][row_idx]
            label = ['a', 'b'][row_idx]
            ax.set_title(label, loc='left', fontsize=12, weight='bold')

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()


# Call the function with desired rows
plot_rows([8, 5], add_row_labels=True, top_row_labels_only=False)
plt.savefig('images/modeling/rfdiffusion_traj_basic.svg', dpi=600)
plot_rows([6, 7], add_row_labels=True, top_row_labels_only=False)
plt.savefig('images/modeling/rfdiffusion_traj_symmetry.svg', dpi=600)