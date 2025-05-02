import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label as cc_label, find_objects
import torchvision.transforms as T
from collections import Counter
from tqdm import tqdm
import cv2
from src.scripts.ChocolateClassifier import ChocolateClassifier
from src.scripts.UNet import UNet


# --- Load fine-tuned classifier ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = ChocolateClassifier().to(device)
classifier.load_state_dict(torch.load("src/models/CNN/chocolate_classifier_final.pth", map_location=device))
classifier.eval()

# --- Load U-Net model ---
unet = UNet().to(device)
unet.load_state_dict(torch.load("src/models/UNET/unet_final.pth", map_location=device))
unet.eval()

# --- Label mappings ---
label_order = ["Jelly White", "Jelly Milk", "Jelly Black", "Amandina", "Crème brulée",
               "Triangolo", "Tentation noir", "Comtesse", "Noblesse", "Noir authentique",
               "Passion au lait", "Arabia", "Stracciatella"]

idx_to_label = {
    0: "Triangolo", 1: "Jelly_White", 2: "Jelly_Milk", 3: "Jelly_Black",
    4: "Amandina", 5: "Crème_brûlée", 6: "Tentation_Noir", 7: "Arabia",
    8: "Comtesse", 9: "Noblesse", 10: "Passion_au_lait", 11: "Stracciatella", 12: "Noir_authentique"
}

# ✅ FIXED label name mapping to match submission headers
label_name_map = {
    "Jelly_White": "Jelly White",
    "Jelly_Milk": "Jelly Milk",
    "Jelly_Black": "Jelly Black",
    "Amandina": "Amandina",
    "Crème_brûlée": "Crème brulée",
    "Triangolo": "Triangolo",
    "Tentation_Noir": "Tentation noir",
    "Comtesse": "Comtesse",
    "Noblesse": "Noblesse",
    "Noir_authentique": "Noir authentique",
    "Passion_au_lait": "Passion au lait",
    "Arabia": "Arabia",
    "Stracciatella": "Stracciatella"
}

# --- Thresholds ---
MIN_BLOB_AREA = 200
MAX_BLOB_AREA = 1200

# --- Transforms ---
resize = T.Resize((256, 256))
to_tensor = T.ToTensor()
clf_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


# --- New split function ---
def split_with_defects(mask, convexity_thresh=0.94, max_depth=3, depth=0, max_blob_area=1400):
    splits = []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BLOB_AREA:
            continue

        hull = cv2.convexHull(cnt, returnPoints=False)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area == 0 or hull is None or len(hull) < 3:
            continue
        convexity = area / hull_area

        x, y, w, h = cv2.boundingRect(cnt)
        region = mask[y:y+h, x:x+w]

        force_split = (area > max_blob_area)

        if (convexity >= convexity_thresh and not force_split) or depth >= max_depth:
            splits.append((y, y+h, x, x+w))
            continue

        defects = cv2.convexityDefects(cnt, hull)
        if defects is not None and defects.shape[0] >= 2:
            defects = sorted(defects, key=lambda d: d[0][3], reverse=True)
            _, _, f1_idx, _ = defects[0][0]
            _, _, f2_idx, _ = defects[1][0]
            pt1 = tuple(cnt[f1_idx][0])
            pt2 = tuple(cnt[f2_idx][0])

            cut_mask = region.copy()
            cv2.line(cut_mask, (pt1[0] - x, pt1[1] - y), (pt2[0] - x, pt2[1] - y), 0, 2)

            num_labels, labeled = cv2.connectedComponents(cut_mask)
            for label_val in range(1, num_labels):
                component_mask = (labeled == label_val).astype(np.uint8)
                if component_mask.sum() < MIN_BLOB_AREA:
                    continue
                ys, xs = np.where(component_mask)
                if ys.size == 0 or xs.size == 0:
                    continue
                sy1, sy2 = ys.min(), ys.max()
                sx1, sx2 = xs.min(), xs.max()
                sub_region = np.zeros_like(mask)
                sub_region[y+sy1:y+sy2, x+sx1:x+sx2] = component_mask[sy1:sy2, sx1:sx2]
                splits.extend(split_with_defects(sub_region, convexity_thresh, max_depth, depth + 1, max_blob_area))
        else:
            splits.append((y, y+h, x, x+w))

    return splits


# --- Process test images ---
test_dir = "src/data/dataset_project_iapr2025/test"
test_img_names = sorted([f for f in os.listdir(test_dir) if f.endswith(".JPG")])
results = []

for name in tqdm(test_img_names, desc="Processing test images"):
    img_path = os.path.join(test_dir, name)
    img_pil = Image.open(img_path).convert("RGB")  # <-- Original image (full res!)
    img_resized = resize(img_pil)  # ← Downsampled for U-Net
    img_tensor = to_tensor(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = unet(img_tensor)
        pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # Convert prediction mask to labeled blobs
    labeled, _ = cc_label(pred_mask)
    blobs = find_objects(labeled)

    label_counts = Counter()

    for obj_slice in blobs:
        y1, y2 = obj_slice[0].start, obj_slice[0].stop
        x1, x2 = obj_slice[1].start, obj_slice[1].stop
        crop_mask = pred_mask[y1:y2, x1:x2]
        area = crop_mask.sum()
        if area < MIN_BLOB_AREA:
            continue

        # Split large blobs
        regions = split_with_defects(crop_mask)
        for sy1, sy2, sx1, sx2 in regions:
            sub_mask = crop_mask[sy1:sy2, sx1:sx2]
            if sub_mask.sum() >= MIN_BLOB_AREA:
                # Compute coords in 256 space
                r_y1, r_y2 = y1 + sy1, y1 + sy2
                r_x1, r_x2 = x1 + sx1, x1 + sx2

                # Convert coords from 256 space to original size
                orig_w, orig_h = img_pil.size
                scale_x = orig_w / 256
                scale_y = orig_h / 256
                o_x1, o_x2 = int(r_x1 * scale_x), int(r_x2 * scale_x)
                o_y1, o_y2 = int(r_y1 * scale_y), int(r_y2 * scale_y)

                # Crop from original high-resolution image
                patch_np = np.array(img_pil.crop((o_x1, o_y1, o_x2, o_y2)))
                patch_img = Image.fromarray(patch_np)

                # Classify
                patch_input = clf_transform(patch_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = classifier(patch_input)
                    probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                    pred_class = probs.argmax()
                    confidence = probs[pred_class]

                    if confidence >= 0.2:
                        label_name = idx_to_label[pred_class]
                        mapped_label = label_name_map.get(label_name)
                        if mapped_label:
                            label_counts[mapped_label] += 1

    row = {"id": int(name.replace("L", "").replace(".JPG", ""))}
    row.update({label: label_counts.get(label, 0) for label in label_order})
    results.append(row)

# --- Save submission ---
df_submission = pd.DataFrame(results)
df_submission = df_submission[["id"] + label_order]
df_submission.sort_values("id", inplace=True)
df_submission.to_csv("src/submissions/submission.csv", index=False)
print("✅ Submission saved to submission.csv")