import numpy as np
import cv2
from pathlib import Path
import tifffile as tiff

def extract_classification_patches(
    images_dir,
    masks_dir,
    patch_size=128,
    use_ratio=True,
    threshold=0.05,
    output_dir="dataset"
):

    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    healthy_dir = output_dir / "healthy"
    diseased_dir = output_dir / "diseased"

    healthy_dir.mkdir(parents=True, exist_ok=True)
    diseased_dir.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in [".tif", ".tiff"]]

    total_healthy = 0
    total_diseased = 0

    for image_file in image_files:

        mask_file = masks_dir / image_file.name
        # print(f'[image_file] : {image_file} | [mask_file] : {mask_file}')
     
        if not mask_file.exists():
            continue

        img = tiff.imread(str(image_file))
        mask = tiff.imread(str(mask_file))

        # print("Mask", mask.shape)
        # print("Image", img.shape)
        # break
        # print("Mask unique values:", np.unique(mask))

        H, W = mask.shape
        n_rows = H // patch_size
        n_cols = W // patch_size

        for i in range(n_rows):
            for j in range(n_cols):

                y0, x0 = i * patch_size, j * patch_size
                y1, x1 = y0 + patch_size, x0 + patch_size

                img_patch = img[y0:y1, x0:x1]
                mask_patch = mask[y0:y1, x0:x1]
                

                # # Si au moins un pixel malade
                # if np.any(mask_patch > 0):
                #     label_dir = diseased_dir
                #     total_diseased += 1
                # else:
                #     label_dir = healthy_dir
                #     total_healthy += 1

                # ratio = np.sum(mask_patch > 0) / mask_patch.size
                ratio = (mask_patch > 0).mean()

                is_diseased = ratio > (threshold if use_ratio else 0)

                if is_diseased:
                    label_dir = diseased_dir
                    total_diseased += 1
                else:
                    label_dir = healthy_dir
                    total_healthy += 1

                name = f"{image_file.stem}_{i:03d}_{j:03d}.png"
                cv2.imwrite(str(label_dir / name), img_patch)

    print("Healthy:", total_healthy)
    print("Diseased:", total_diseased)


extract_classification_patches(
    images_dir="/home/adjalil/Working/data_lionel/raw", 
    masks_dir="/home/adjalil/Working/data_lionel/mask", 
    patch_size=224, 
    output_dir="/home/adjalil/Working/data_lionel/train_dataset"
)
