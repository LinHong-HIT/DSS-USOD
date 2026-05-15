import os
import torch
import argparse
import cv2
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.dataset import USODDataset
from src.model import DSSUSOD

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Default config
# =========================================================
DATA_ROOT = "path to the dataset"
MODEL_PATH = "/home/ubuntu/USOD/checkpoints/best.pth"
SAVE_DIR = "prediction_v1"
INFERENCE_SIZE = 352

TARGET_DATASETS = [
    "USOD",
    "USOD10K",
]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_images",
        type=int,
        default=-1,
        help="the maximum outputs, -1 means all",
    )

    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--size", type=int, default=INFERENCE_SIZE)

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="the threhold for generating saliency mask",
    )

    return parser.parse_args()


def load_ckpt(model, ckpt_path: str):
   
    state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    
    state = {k.replace("module.", ""): v for k, v in state.items()}


    try:
        model.load_state_dict(state, strict=True)
        print("[ckpt] strict load OK (no remap).")
        return
    except RuntimeError as e:
        print("[ckpt] strict load failed, try remap linear_pred -> seg_head.")
        print("  reason:", str(e).split("\n")[0])

    remap = {}
    for k, v in state.items():
        nk = k.replace("decoder.linear_pred.", "decoder.seg_head.")
        remap[nk] = v

    model.load_state_dict(remap, strict=True)
    print("[ckpt] strict load OK (after remap).")


def _model_forward_logits(model, rgb):
    """
      model(rgb) -> logits
      model(rgb) -> (final_logits, coarse_full_logits, seg_logits_small)
      model(rgb) -> (final_logits, coarse_full_logits, seg_logits_small, aux)
    """
    out = model(rgb)

    if isinstance(out, (tuple, list)):
        return out[0]

    return out


@torch.no_grad()
def rgb_only_infer(model, rgb):
    """
    RGB-only inference.

    input:
        rgb: [B, 3, H, W]

    output:
        prob: [B, 1, H, W], range [0, 1]
    """
    model.eval()

    logits = _model_forward_logits(model, rgb)
    prob = torch.sigmoid(logits).clamp(0, 1)

    return prob


def inference_one_dataset(model, dataset_root, dataset_name, args, remaining_limit=None):
    test_dataset = USODDataset(
        dataset_root,
        mode="test",
        size=args.size,
    )

    loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print("\n" + "=" * 70)
    print(f"Start inference on [{dataset_name}]")
    print(f"Dataset root: {dataset_root}")
    print(f"Number of test images: {len(test_dataset)}")
    print(f"Threshold for binary mask: {args.threshold}")
    print("=" * 70)

    # save the prob logits
    prob_subdir = os.path.join(args.save_dir, "prob", dataset_name)
    os.makedirs(prob_subdir, exist_ok=True)

    # save mask
    mask_subdir = os.path.join(args.save_dir, "mask", dataset_name)
    os.makedirs(mask_subdir, exist_ok=True)

    saved_count = 0

    for batch in tqdm(loader):
        if remaining_limit is not None and saved_count >= remaining_limit:
            break

        rgb, img_name = batch
        rgb = rgb.to(DEVICE, non_blocking=True)

        if isinstance(img_name, (list, tuple)):
            img_name = img_name[0]

        pred_prob = rgb_only_infer(model, rgb)  # [1, 1, H, W]
        pred_map = pred_prob[0, 0].detach().cpu().numpy()

        prob_save = (pred_map * 255).astype("uint8")

        mask_save = ((pred_map >= float(args.threshold)).astype("uint8") * 255)

        save_name = os.path.splitext(img_name)[0] + ".png"

        prob_path = os.path.join(prob_subdir, save_name)
        mask_path = os.path.join(mask_subdir, save_name)

        cv2.imwrite(prob_path, prob_save)
        cv2.imwrite(mask_path, mask_save)

        saved_count += 1

    print(f"[{dataset_name}] Saved {saved_count} probability maps to: {prob_subdir}")
    print(f"[{dataset_name}] Saved {saved_count} binary masks to: {mask_subdir}")

    return saved_count


def inference():
    args = get_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("Using device:", DEVICE)

    if DEVICE == "cuda":
        print("CUDA visible device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))

    if not os.path.exists(args.model_path):
        print(f"Error: Checkpoint {args.model_path} not found.")
        return

    print(f"Loading checkpoint: {args.model_path}")

    model = DSSUSOD(in_chans=3)
    load_ckpt(model, args.model_path)

    model = model.to(DEVICE)
    model.eval()

    dataset_names = []

    for name in TARGET_DATASETS:
        dataset_root = os.path.join(args.data_root, name)

        if os.path.isdir(dataset_root):
            dataset_names.append(name)
        else:
            print(f"[Info] Dataset {name} not found: {dataset_root}")

    if len(dataset_names) == 0:
        print(f"Error: no dataset found under {args.data_root}")
        print("Expected folders: USOD and/or USOD10K")
        return

    print("Found datasets:", dataset_names)

    total_saved = 0
    total_limit = None if args.num_images <= 0 else args.num_images

    for dataset_name in dataset_names:
        dataset_root = os.path.join(args.data_root, dataset_name)

        remaining_limit = None

        if total_limit is not None:
            remaining_limit = total_limit - total_saved

            if remaining_limit <= 0:
                break

        saved = inference_one_dataset(
            model=model,
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            args=args,
            remaining_limit=remaining_limit,
        )

        total_saved += saved

    print("\nDone.")
    print(f"Total saved {total_saved} predictions.")
    print(f"Probability maps saved to: {os.path.join(args.save_dir, 'prob')}")
    print(f"Binary masks saved to:     {os.path.join(args.save_dir, 'mask')}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    inference()
