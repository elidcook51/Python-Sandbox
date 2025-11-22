import torch
import random
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import numpy as np

def evaluate_label_flips(
    modified_dataset,
    original_labels,
    flip_out_labels=None,         # e.g., [3]; optional—used for targeted metrics
    flip_in_labels=None,          # e.g., [3]; optional—used for targeted metrics
    expected_flip_percentage=None,# e.g., 0.5; optional—used to check deviation
):
    """
    Evaluate effectiveness of label flipping by comparing modified labels against original labels.

    Args:
        modified_dataset: torchvision.datasets.MNIST (or similar) with `.targets` tensor of shape [N].
        original_labels: list or 1D array-like of length N holding original (clean) labels.
        flip_out_labels: list[int] of labels intended to flip OUT OF. Optional.
        flip_in_labels: list[int] of labels intended to flip INTO. Optional.
        expected_flip_percentage: float (0..1), the target fraction flipped. Optional.
        plot_confusion: bool, whether to display a confusion matrix plot.

    Returns:
        results (dict) with keys:
            - overall:
                - n_samples
                - n_flipped
                - flip_rate
                - expected_flip_percentage (if provided)
                - flip_rate_error (if expected provided)
            - per_class:
                - class_counts (dict[label]->count)
                - flip_out_rate (dict[label]->fraction flipped out of that class) if flip_out_labels provided
                - flip_in_rate (dict[label]->fraction that became that class from other labels) if flip_in_labels provided
            - confusion_matrix: 2D numpy array [10x10]
            - flips_detail:
                - flipped_indices (list[int])
                - flipped_from_to (list[tuple(idx, orig, new)])
    """

    # --- Input normalization ---
    if isinstance(original_labels, list):
        orig = np.array(original_labels, dtype=np.int64)
    elif isinstance(original_labels, torch.Tensor):
        orig = original_labels.detach().cpu().numpy().astype(np.int64)
    else:
        orig = np.array(original_labels, dtype=np.int64)

    new = modified_dataset.targets.detach().cpu().numpy().astype(np.int64)
    assert len(orig) == len(new), "Original and modified label arrays must have the same length."

    n = len(orig)

    # --- Which indices flipped? ---
    flipped_mask = (orig != new)
    flipped_indices = np.where(flipped_mask)[0].tolist()
    n_flipped = int(flipped_mask.sum())
    flip_rate = n_flipped / n if n > 0 else 0.0

    # --- Confusion matrix (original -> new) ---
    num_classes = int(max(orig.max(), new.max())) + 1  # usually 10 for MNIST
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for o, m in zip(orig, new):
        cm[o, m] += 1

    # --- Per-class counts ---
    class_counts = {c: int((orig == c).sum()) for c in range(num_classes)}

    # --- Flip-out metrics: fraction of class c that flipped to anything else ---
    flip_out_rate = None
    if flip_out_labels is not None:
        flip_out_rate = {}
        for c in flip_out_labels:
            denom = class_counts.get(c, 0)
            num = int(cm[c, :].sum() - cm[c, c])  # all from c that didn't stay c
            flip_out_rate[c] = (num / denom) if denom > 0 else 0.0

    # --- Flip-in metrics: fraction of final class c that came from other classes ---
    flip_in_rate = None
    if flip_in_labels is not None:
        flip_in_rate = {}
        for c in flip_in_labels:
            final_count_c = int(cm[:, c].sum())            # total samples now labeled c
            from_other = int(cm[:, c].sum() - cm[c, c])    # came from labels != c
            flip_in_rate[c] = (from_other / final_count_c) if final_count_c > 0 else 0.0

    # --- Expected flip percentage deviation (optional) ---
    flip_rate_error = None
    if expected_flip_percentage is not None:
        flip_rate_error = flip_rate - float(expected_flip_percentage)

    # --- Detailed flips list (idx, from, to) ---
    flipped_from_to = [(int(i), int(orig[i]), int(new[i])) for i in flipped_indices]

    # --- Plot confusion matrix (optional) ---

    results = {
        "overall": {
            "n_samples": n,
            "n_flipped": n_flipped,
            "flip_rate": flip_rate,
            "expected_flip_percentage": expected_flip_percentage,
            "flip_rate_error": flip_rate_error,
        },
        "per_class": {
            "class_counts": class_counts,
            "flip_out_rate": flip_out_rate,
            "flip_in_rate": flip_in_rate,
        },
        "confusion_matrix": cm,
        "flips_detail": {
            "flipped_indices": flipped_indices,
            "flipped_from_to": flipped_from_to,
        }
    }
    return results


def flip_labels(dataset = torchvision.datasets.MNIST, flip_out_labels = [0,1,2,3,4,5,6,7,8,9], flip_in_labels = [0,1,2,3,4,5,6,7,8,9], flip_percentage = 0.0):
    if not (0 <= flip_percentage <= 1):
        raise ValueError('flip percentage must be between 0 and 1')

    origional_labels = dataset.targets.clone()
    targets = dataset.targets.clone()
    out_indices = [i for i, lbl in enumerate(targets) if lbl in flip_out_labels]

    numFlips = int(len(origional_labels) * flip_percentage)

    if numFlips > len(out_indices):
        numFlips = len(out_indices)

    flip_out_selected = random.sample(out_indices, numFlips)

    for idx in flip_out_selected:
        new_label = random.choice(flip_in_labels)
        targets[idx] = new_label
    
    dataset.targets = targets
    return dataset, origional_labels.tolist()

def flip_pairs(dataset, pairs, flip_percentage):
    origionalLabels = dataset.targets.clone()
    totFlips = int(len(origionalLabels) * flip_percentage)
    pairFlips = int(totFlips / len(pairs))
    pairPercentage =  pairFlips / len(origionalLabels)
    for first, second in pairs:
        dataset, origionalLabels = flip_labels(dataset, [first], [second], pairPercentage)
    return dataset, origionalLabels

mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
                       
# newDataset, origionalLabels = flip_labels(mnist, flip_out_labels = [6], flip_in_labels = [7], flip_percentage= 0.25)
newDataset, origionalLabels = flip_pairs(mnist, [(1,2), (3,4), (5,6)], 0.1)
# print(newDataset.targets[:150])
# print(origionalLabels[:150])

results = evaluate_label_flips(newDataset, origionalLabels, [0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9], 0.25)
print(results['overall'])
print(results['per_class'])

