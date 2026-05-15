import os
import torch
from evaluator import Eval_thread
from dataloader import EvalDataset


PRED_ROOT = r"/home/ubuntu/USOD/prediction_v1/mask"
# path for the dataset
DATA_ROOT = r"/home/ubuntu/USOD/USOD/data"

# evaluation output
OUTPUT_DIR = "eval_metrics_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def simple_collate(batch):
    return batch[0]


def save_text(text, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)


def append_text(text, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "a", encoding="utf-8") as f:
        f.write(text)


def evaluate_one_dataset(dataset_name, pred_root, gt_root, output_dir, use_cuda=True):
    print("\n" + "=" * 60)
    print(f"Start Evaluation: {dataset_name}")
    print("=" * 60)
    print(f"Preds: {pred_root}")
    print(f"GT:    {gt_root}")
    print(f"Out:   {output_dir}")

    if not os.path.isdir(pred_root):
        msg = f"[Skip] Prediction folder not found: {pred_root}"
        print(msg)
        save_text(msg + "\n", os.path.join(output_dir, "result.txt"))
        return None

    if not os.path.isdir(gt_root):
        msg = f"[Skip] GT folder not found: {gt_root}"
        print(msg)
        save_text(msg + "\n", os.path.join(output_dir, "result.txt"))
        return None

    os.makedirs(output_dir, exist_ok=True)

    loader = EvalDataset(pred_root, gt_root)

    test_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=simple_collate
    )

    evaluator = Eval_thread(
        test_loader,
        method="our",
        dataset=dataset_name,
        output_dir=output_dir,
        cuda=use_cuda
    )

    result_log = evaluator.run()

    if result_log is None:
        result_log = f"{dataset_name}: evaluator.run() returned None"

    print("\n" + "-" * 60)
    print(f"FINAL RESULTS: {dataset_name}")
    print("-" * 60)
    print(result_log)
    print("-" * 60)

    result_txt_path = os.path.join(output_dir, "result.txt")
    save_text(str(result_log) + "\n", result_txt_path)
    print(f"Saved result log to: {result_txt_path}")

    return str(result_log)


def run_evaluation():
    print("Start Evaluation for multiple datasets...")
    print(f"PRED_ROOT : {PRED_ROOT}")
    print(f"DATA_ROOT : {DATA_ROOT}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("Warning: CUDA not found, evaluation will run on CPU if supported.")

    dataset_names = []
    for name in ["USOD", "USOD10K"]:
        pred_dir = os.path.join(PRED_ROOT, name)
        gt_dir = os.path.join(DATA_ROOT, name, "test", "GT")

        if os.path.isdir(pred_dir) and os.path.isdir(gt_dir):
            dataset_names.append(name)
        else:
            print(f"[Info] Dataset {name} not ready:")
            print(f"       pred exists? {os.path.isdir(pred_dir)}")
            print(f"       gt   exists? {os.path.isdir(gt_dir)}")

    if len(dataset_names) == 0:
        print("Error: No valid dataset pair found for evaluation.")
        return

    all_results = {}
    summary_txt_path = os.path.join(OUTPUT_DIR, "summary.txt")

    save_text("Evaluation Summary\n" + "=" * 60 + "\n", summary_txt_path)

    for dataset_name in dataset_names:
        pred_dir = os.path.join(PRED_ROOT, dataset_name)
        gt_dir = os.path.join(DATA_ROOT, dataset_name, "test", "GT")
        out_dir = os.path.join(OUTPUT_DIR, dataset_name)

        result_log = evaluate_one_dataset(
            dataset_name=dataset_name,
            pred_root=pred_dir,
            gt_root=gt_dir,
            output_dir=out_dir,
            use_cuda=use_cuda
        )

        all_results[dataset_name] = result_log

        append_text(
            f"\n[{dataset_name}]\n"
            + (result_log if result_log is not None else "No result") +
            "\n" + "-" * 60 + "\n",
            summary_txt_path
        )

    print("\n" + "=" * 60)
    print("SUMMARY OF ALL DATASETS")
    print("=" * 60)
    for dataset_name, result_log in all_results.items():
        print(f"\n[{dataset_name}]")
        print(result_log)
    print("=" * 60)

    print(f"Saved summary to: {summary_txt_path}")


if __name__ == "__main__":
    run_evaluation()
