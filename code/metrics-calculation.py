import numpy as np
from PIL import Image
import os
from skimage.morphology import binary_dilation, disk

# Function to calculate Jaccard recall above a certain threshold
def calculate_j_recall(jaccards, threshold=0.5):
    valid_jaccards = [j for j in jaccards if j >= threshold]
    return np.mean(valid_jaccards) if valid_jaccards else 0

# Function to calculate F-measure recall above a certain threshold
def calculate_f_recall(fmeasures, threshold=0.5):
    valid_fmeasures = [f for f in fmeasures if f >= threshold]
    return np.mean(valid_fmeasures) if valid_fmeasures else 0

# Function to calculate the Jaccard index for true and predicted masks
def calculate_jaccard_index(true_mask, pred_mask):
    intersection = np.sum(true_mask & pred_mask)
    union = np.sum(true_mask | pred_mask)
    return intersection / union if union != 0 else 0

# Function to convert a segmentation into a binary boundary map
def seg2bmap(seg):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries. The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    """
    seg = seg.astype(bool)
    seg[seg > 0] = 1

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0  # Bottom right pixel is always 0

    return b

# Function to calculate F-measure using boundary maps
def calculate_fmeasure(true_mask, pred_mask):
    """
    Calculate F-measure using boundaries approach
    """
    bound_th=0.008
    fg_boundary = seg2bmap(pred_mask)
    gt_boundary = seg2bmap(true_mask)

    bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th*np.linalg.norm(fg_boundary.shape))

    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

    fg_match = fg_boundary * gt_dil
    gt_match = gt_boundary * fg_dil

    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    if precision + recall == 0:
        f_measure = 0
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    return f_measure

# Function to process all files in given directories for true and predicted masks
def process_folder(true_base_path, pred_base_path):
    true_masks = sorted([os.path.join(true_base_path, f) for f in os.listdir(true_base_path) if f.endswith('.png')])
    pred_masks = sorted([os.path.join(pred_base_path, f) for f in os.listdir(pred_base_path) if f.endswith('.png')])
    
    # If the number of true mask files does not match the number of predicted mask files, return None
    if len(true_masks) != len(pred_masks):
        return None  # Number of files do not match, skip processing
    
    frame_jaccards = []
    frame_fmeasures = []

    # Iterate over each pair of true and predicted mask files
    for true_path, pred_path in zip(true_masks, pred_masks):
        # print(f"Processing {true_path}")
        # Attempt to open and process each pair of images
        try:
            true_img = Image.open(true_path).convert("L")
            pred_img = Image.open(pred_path).convert("L")

            if true_img.size != pred_img.size:
                true_img = true_img.resize(pred_img.size, Image.NEAREST)

            true_mask = np.array(true_img)
            pred_mask = np.array(pred_img)

        except Exception as e:
            print(f"Error loading images: {e}")
            continue

        label = np.unique(np.concatenate((true_mask, pred_mask)))
        label = label[label != 0]  # Exclude the background

        jaccards = []
        fmeasures = []

        # Process each label separately
        for l in label:
            true_mask_label = true_mask == l
            pred_mask_label = pred_mask == l
            
            # Calculate Jaccard index and F-measure for each label
            jaccard_index = calculate_jaccard_index(true_mask_label, pred_mask_label)
            f_measure = calculate_fmeasure(true_mask_label, pred_mask_label)
            jaccards.append(jaccard_index)
            fmeasures.append(f_measure)
        
        # Calculate average Jaccard index and F-measure for the current image pair
        if jaccards:
            frame_jaccards.append(np.mean(jaccards))
        if fmeasures:
            frame_fmeasures.append(np.mean(fmeasures))

    # Calculate overall mean Jaccard index and F-measure for all images
    mean_j = np.mean(frame_jaccards) if frame_jaccards else 0
    mean_f = np.mean(frame_fmeasures) if frame_fmeasures else 0
    
    # Calculate recall metrics for Jaccard index and F-measure
    j_recall = calculate_j_recall(frame_jaccards)
    f_recall = calculate_f_recall(frame_fmeasures)

    # Calculate the change in Jaccard index and F-measure from the first to the last image
    j_decay = frame_jaccards[0] - frame_jaccards[-1] if frame_jaccards else 0
    f_decay = frame_fmeasures[0] - frame_fmeasures[-1] if frame_fmeasures else 0

    return mean_j, mean_f, j_recall, f_recall, j_decay, f_decay

# Main logic to process each folder of predicted and true masks
base_pred_path = './result/zrq'
# base_pred_path = './rough_annotation/osvos/selected'
base_true_path = './trainval/DAVIS/Annotations/480p'
results = {}
global_jaccards = []
global_fmeasures = []
global_j_recalls = []  # Store J recall for each folder
global_f_recalls = []  # Store F recall for each folder
global_decays = []  # Store decay metrics for each folder

for pred_folder in os.listdir(base_pred_path):
    pred_path = os.path.join(base_pred_path, pred_folder)
    true_path = os.path.join(base_true_path, pred_folder)
    
    if os.path.isdir(pred_path) and os.path.isdir(true_path):
        print(f"Processing {pred_folder}")
        result = process_folder(true_path, pred_path)
        if result:
            mean_j, mean_f, j_recall, f_recall, j_decay, f_decay = result
            results[pred_folder] = result
            global_jaccards.append(mean_j)
            global_fmeasures.append(mean_f)
            global_j_recalls.append(j_recall)
            global_f_recalls.append(f_recall)
            global_decays.append((j_decay, f_decay))

# Calculate global means across all folders
global_mean_j = np.mean(global_jaccards) if global_jaccards else 0
global_mean_f = np.mean(global_fmeasures) if global_fmeasures else 0
# Calculate global mean Jaccard recall and F-measure recall across all folders
global_mean_j_recall = np.mean(global_j_recalls) if global_j_recalls else 0
global_mean_f_recall = np.mean(global_f_recalls) if global_f_recalls else 0
# Calculate average decay rates for Jaccard and F-measures across all folders
global_j_decay = np.mean([d[0] for d in global_decays]) if global_decays else 0
global_f_decay = np.mean([d[1] for d in global_decays]) if global_decays else 0

# Print all results
for folder, (mean_j, mean_f, j_recall, f_recall, j_decay, f_decay) in results.items():
    print(f"Folder: {folder}")
    print(f"Mean J (Jaccard Index): {mean_j}")
    print(f"Mean F-measure: {mean_f}")
    print(f"J Recall: {j_recall}")
    print(f"F Recall: {f_recall}")
    print(f"J Decay: {j_decay}")
    print(f"F Decay: {f_decay}")
    print("-" * 50)

print(f"Global Mean J (Jaccard Index): {global_mean_j}")
print(f"Global Mean F-measure: {global_mean_f}")
print(f"Global J Recall: {global_mean_j_recall}")
print(f"Global F Recall: {global_mean_f_recall}")
print(f"Global J Decay: {global_j_decay})")
print(f"Global F Decay: {global_f_decay}")

# import subprocess
# import time

# time.sleep(30)

# subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], check=True)