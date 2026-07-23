"""Calibrate and diagnose CoP depth routing on the validation split.

The detector is run only once per validation batch.  The small decoder outputs
needed for KITTI decoding are cached on CPU, then reused for absolute and
chain-minus-parallel log-variance scans.  Hungarian-matched ground truths are
also used for branch depth errors and a matched-positive oracle.  No training
state or model configuration is modified.
"""

import argparse
import copy
import csv
import json
import math
import os
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import tqdm
import yaml
from scipy.stats import pearsonr, spearmanr


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.decode_helper import decode_detections, extract_dets_from_outputs
from lib.helpers.model_helper import build_model
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.utils_helper import set_random_seed


CACHE_KEYS = (
    "pred_logits",
    "pred_boxes",
    "pred_3d_dim",
    "pred_angle",
    "pred_depth_chain",
    "pred_depth_parallel",
    "pred_chain_log_variance",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate and diagnose MonoDETR CoP routing on KITTI val")
    parser.add_argument("--config", default="configs/monodetr.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (default: outputs/<model_name>/checkpoint_best.pth)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Calibration directory (default: outputs/<model_name>/tau_calibration)")
    parser.add_argument(
        "--taus",
        default=None,
        help="Optional comma-separated thresholds, e.g. '-1,-0.5,0,0.5,1'. "
             "Automatic quantiles are used when omitted.")
    parser.add_argument(
        "--num-quantiles",
        type=int,
        default=19,
        help="Number of interior quantile thresholds for automatic scanning")
    parser.add_argument(
        "--delta-taus",
        default=None,
        help="Optional comma-separated thresholds for chain_logvar - "
             "parallel_logvar. Automatic quantiles are used when omitted.")
    parser.add_argument(
        "--num-delta-quantiles",
        type=int,
        default=19,
        help="Number of interior relative-uncertainty thresholds")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing calibration output directory")
    parser.add_argument(
        "--keep-predictions",
        action="store_true",
        help="Keep per-tau KITTI txt files (they are removed by default)")
    return parser.parse_args()


def absolute_from_root(path):
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def prepare_config(path, batch_size=None):
    with open(path, "r") as stream:
        cfg = yaml.load(stream, Loader=yaml.Loader)

    teacher_depth_cfg = copy.deepcopy(cfg.get("teacher_depth", {}))
    teacher_depth_cfg.setdefault("min_depth", cfg["model"]["depth_min"])
    teacher_depth_cfg.setdefault("max_depth", cfg["model"]["depth_max"])
    cfg["dataset"]["teacher_depth"] = copy.deepcopy(teacher_depth_cfg)
    cfg["model"]["teacher_depth"] = copy.deepcopy(teacher_depth_cfg)

    cop_cfg = copy.deepcopy(cfg.get("cop", {}))
    if not cop_cfg.get("enabled", False):
        raise ValueError("cop.enabled must be True for uncertainty calibration")
    cfg["model"]["cop"] = copy.deepcopy(cop_cfg)
    if batch_size is not None:
        cfg["dataset"]["batch_size"] = batch_size
    return cfg


def parse_tau(value):
    normalized = value.strip().lower()
    if normalized in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if normalized in {"-inf", "-infinity"}:
        return -math.inf
    return float(normalized)


def tau_text(tau):
    if tau == -math.inf:
        return "-inf"
    if tau == math.inf:
        return "+inf"
    return "{:.8g}".format(tau)


def tau_dir_name(index, tau):
    if tau == -math.inf:
        suffix = "all_parallel"
    elif tau == math.inf:
        suffix = "all_chain"
    else:
        suffix = tau_text(tau).replace("-", "m").replace("+", "p").replace(".", "p")
    return "tau_{:03d}_{}".format(index, suffix)


def unique_sorted(values):
    finite = sorted(set(float(value) for value in values if math.isfinite(value)))
    result = [-math.inf]
    result.extend(finite)
    result.append(math.inf)
    return result


def build_thresholds(raw_values, num_quantiles, samples, configured_tau):
    if raw_values:
        values = [parse_tau(value) for value in raw_values.split(",") if value.strip()]
    else:
        if num_quantiles < 1:
            raise ValueError("The number of quantiles must be at least 1")
        finite = samples[np.isfinite(samples)]
        if finite.size == 0:
            raise RuntimeError("No finite uncertainty values were produced")
        quantiles = np.linspace(
            1.0 / (num_quantiles + 1),
            num_quantiles / (num_quantiles + 1),
            num_quantiles,
        )
        values = np.quantile(finite, quantiles).tolist()

    # Always retain the current config as a directly comparable baseline.
    if configured_tau is not None:
        values.append(float(configured_tau))
    return unique_sorted(values)


def prepare_match_targets(raw_targets, device):
    mask = raw_targets["mask_2d"]
    keys = ("labels", "boxes", "boxes_3d", "depth")
    prepared = []
    for batch_index in range(mask.shape[0]):
        prepared.append({
            key: raw_targets[key][batch_index][mask[batch_index]].to(device)
            for key in keys
        })
    return prepared


def gather_topk_values(logits, values, topk, score_threshold):
    probabilities = logits.sigmoid()
    count = min(topk, probabilities.shape[1] * probabilities.shape[2])
    scores, indexes = torch.topk(probabilities.flatten(1), count, dim=1)
    query_indexes = (indexes // probabilities.shape[2]).unsqueeze(-1)
    selected = torch.gather(values, 1, query_indexes).squeeze(-1)
    return selected[scores >= score_threshold]


def cache_validation_outputs(
        model, matcher, dataloader, device, topk, score_threshold):
    cached_batches = []
    accepted_chain_logvars = []
    accepted_delta_logvars = []
    diagnostics = {
        "target_depth": [],
        "chain_depth": [],
        "parallel_depth": [],
        "chain_logvar": [],
        "parallel_logvar": [],
    }

    model.eval()
    progress = tqdm.tqdm(dataloader, desc="Caching validation outputs")
    with torch.inference_mode():
        for inputs, calibs, targets, info in progress:
            inputs = inputs.to(device, non_blocking=True)
            calibs = calibs.to(device, non_blocking=True)
            img_sizes = info["img_size"].to(device, non_blocking=True)
            outputs = model(inputs, calibs, targets, img_sizes, dn_args=0)

            missing = [key for key in CACHE_KEYS if key not in outputs]
            if missing:
                raise RuntimeError(
                    "Model is missing calibration outputs: {}".format(
                        ", ".join(missing)))

            cache = {
                key: outputs[key].detach().cpu()
                for key in CACHE_KEYS
            }
            match_targets = prepare_match_targets(targets, device)
            match_indices = matcher(
                {
                    "pred_logits": outputs["pred_logits"],
                    "pred_boxes": outputs["pred_boxes"],
                },
                match_targets,
                group_num=1,
            )

            # Unmatched queries default to the stronger chain branch.  For
            # matched positives, the oracle chooses the branch with lower
            # absolute metric-depth error.
            oracle_use_chain = torch.ones_like(
                outputs["pred_chain_log_variance"], dtype=torch.bool)
            for batch_index, (source_indexes, target_indexes) in enumerate(
                    match_indices):
                if len(source_indexes) == 0:
                    continue
                source_indexes = source_indexes.to(device)
                target_indexes = target_indexes.to(device)
                target_depth = match_targets[batch_index]["depth"][
                    target_indexes].reshape(-1)
                chain_depth = outputs["pred_depth_chain"][
                    batch_index, source_indexes, 0]
                parallel_depth = outputs["pred_depth_parallel"][
                    batch_index, source_indexes, 0]
                chain_logvar = outputs["pred_depth_chain"][
                    batch_index, source_indexes, 1]
                parallel_logvar = outputs["pred_depth_parallel"][
                    batch_index, source_indexes, 1]
                chain_error = torch.abs(chain_depth - target_depth)
                parallel_error = torch.abs(parallel_depth - target_depth)
                oracle_use_chain[
                    batch_index, source_indexes, 0] = (
                        chain_error <= parallel_error)

                diagnostics["target_depth"].append(
                    target_depth.detach().cpu().numpy())
                diagnostics["chain_depth"].append(
                    chain_depth.detach().cpu().numpy())
                diagnostics["parallel_depth"].append(
                    parallel_depth.detach().cpu().numpy())
                diagnostics["chain_logvar"].append(
                    chain_logvar.detach().cpu().numpy())
                diagnostics["parallel_logvar"].append(
                    parallel_logvar.detach().cpu().numpy())

            cache["oracle_use_chain"] = oracle_use_chain.detach().cpu()
            info_numpy = {
                key: value.detach().cpu().numpy()
                for key, value in info.items()
            }
            cached_batches.append((cache, info_numpy))

            logits = cache["pred_logits"]
            chain_logvar = cache["pred_chain_log_variance"]
            parallel_logvar = cache["pred_depth_parallel"][..., 1:2]
            accepted_chain = gather_topk_values(
                logits, chain_logvar, topk, score_threshold)
            accepted_delta = gather_topk_values(
                logits, chain_logvar - parallel_logvar, topk, score_threshold)
            if accepted_chain.numel():
                accepted_chain_logvars.append(accepted_chain.numpy())
                accepted_delta_logvars.append(accepted_delta.numpy())

    if not accepted_chain_logvars:
        raise RuntimeError(
            "No top-k detections passed tester.threshold; cannot calibrate tau")
    matched = {
        key: np.concatenate(values) if values else np.empty(0, dtype=np.float32)
        for key, values in diagnostics.items()
    }
    if matched["target_depth"].size == 0:
        raise RuntimeError("Hungarian matching produced no validation targets")
    return (
        cached_batches,
        np.concatenate(accepted_chain_logvars),
        np.concatenate(accepted_delta_logvars),
        matched,
    )


def save_results(results, output_dir, dataset):
    os.makedirs(output_dir, exist_ok=True)
    for img_id, predictions in results.items():
        output_path = os.path.join(output_dir, "{:06d}.txt".format(int(img_id)))
        with open(output_path, "w") as stream:
            for prediction in predictions:
                class_name = dataset.class_name[int(prediction[0])]
                stream.write("{} 0.0 0".format(class_name))
                for value in prediction[1:]:
                    stream.write(" {:.2f}".format(value))
                stream.write("\n")


def decode_with_route(
        cache, info, dataset, use_chain, topk, score_threshold):
    selected_depth = torch.where(
        use_chain,
        cache["pred_depth_chain"],
        cache["pred_depth_parallel"],
    )
    outputs = dict(cache)
    outputs["pred_depth"] = selected_depth
    detections = extract_dets_from_outputs(
        outputs=outputs, K=dataset.max_objs, topk=topk).numpy()
    calibs = [dataset.get_calib(index) for index in info["img_id"]]
    return decode_detections(
        dets=detections,
        info=info,
        calibs=calibs,
        cls_mean_size=dataset.cls_mean_size,
        threshold=score_threshold,
    )


def accepted_route_rate(cache, use_chain, topk, score_threshold):
    selected = gather_topk_values(
        cache["pred_logits"], use_chain.float(), topk, score_threshold)
    return float(selected.sum().item()), int(selected.numel())


def metric_row(threshold_key, threshold, route_rate, result_dict):
    row = {
        threshold_key: tau_text(threshold),
        "chain_rate": float(route_rate),
        "3d_easy_R40": float(result_dict["Car_3d_easy_R40"]),
        "3d_moderate_R40": float(result_dict["Car_3d_moderate_R40"]),
        "3d_hard_R40": float(result_dict["Car_3d_hard_R40"]),
        "bev_easy_R40": float(result_dict["Car_bev_easy_R40"]),
        "bev_moderate_R40": float(result_dict["Car_bev_moderate_R40"]),
        "bev_hard_R40": float(result_dict["Car_bev_hard_R40"]),
    }
    return row


def write_summary(
        rows, output_dir, basename, checkpoint, threshold_key,
        configured_threshold, uncertainty_stats):
    csv_path = os.path.join(output_dir, basename + ".csv")
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: row["3d_moderate_R40"])
    payload = {
        "checkpoint": checkpoint,
        "selection_metric": "Car_3d_moderate_R40",
        "configured_threshold": (
            tau_text(configured_threshold)
            if configured_threshold is not None else None),
        "recommended_threshold": best[threshold_key],
        "uncertainty": uncertainty_stats,
        "best": best,
        "results": rows,
    }
    json_path = os.path.join(output_dir, basename + ".json")
    with open(json_path, "w") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    return best, csv_path, json_path


def evaluate_routes(
        cached, dataset, gt_annos, output_dir, run_name, threshold_key,
        thresholds, route_builder, topk, score_threshold, keep_predictions):
    rows = []
    selector_dir = os.path.join(output_dir, run_name)
    os.makedirs(selector_dir)
    for index, threshold in enumerate(thresholds):
        run_dir = os.path.join(
            selector_dir, tau_dir_name(index, threshold))
        data_dir = os.path.join(run_dir, "data")
        route_sum = 0.0
        route_count = 0
        for cache, info in tqdm.tqdm(
                cached,
                desc="{} {}={}".format(
                    run_name, threshold_key, tau_text(threshold)),
                leave=False):
            use_chain = route_builder(cache, threshold)
            selected_sum, selected_count = accepted_route_rate(
                cache, use_chain, topk, score_threshold)
            route_sum += selected_sum
            route_count += selected_count
            results = decode_with_route(
                cache, info, dataset, use_chain, topk, score_threshold)
            save_results(results, data_dir, dataset)

        dt_annos = kitti.get_label_annos(data_dir)
        result_text, result_dict, _ = get_official_eval_result(
            gt_annos, dt_annos, 0)
        with open(os.path.join(run_dir, "evaluation.txt"), "w") as stream:
            stream.write(result_text)
        if not keep_predictions:
            shutil.rmtree(data_dir)

        row = metric_row(
            threshold_key,
            threshold,
            route_sum / max(route_count, 1),
            result_dict,
        )
        rows.append(row)
        print(
            "{}={:>10s}  chain={:6.2%}  3D R40 E/M/H="
            "{:.4f}/{:.4f}/{:.4f}  BEV-M={:.4f}".format(
                threshold_key,
                row[threshold_key],
                row["chain_rate"],
                row["3d_easy_R40"],
                row["3d_moderate_R40"],
                row["3d_hard_R40"],
                row["bev_moderate_R40"],
            ))
    return rows


def evaluate_oracle(
        cached, dataset, gt_annos, output_dir, topk, score_threshold,
        keep_predictions):
    run_dir = os.path.join(output_dir, "matched_depth_oracle")
    data_dir = os.path.join(run_dir, "data")
    route_sum = 0.0
    route_count = 0
    for cache, info in tqdm.tqdm(
            cached, desc="Matched-depth oracle", leave=False):
        use_chain = cache["oracle_use_chain"]
        selected_sum, selected_count = accepted_route_rate(
            cache, use_chain, topk, score_threshold)
        route_sum += selected_sum
        route_count += selected_count
        results = decode_with_route(
            cache, info, dataset, use_chain, topk, score_threshold)
        save_results(results, data_dir, dataset)

    dt_annos = kitti.get_label_annos(data_dir)
    result_text, result_dict, _ = get_official_eval_result(
        gt_annos, dt_annos, 0)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "evaluation.txt"), "w") as stream:
        stream.write(result_text)
    if not keep_predictions:
        shutil.rmtree(data_dir)
    row = metric_row(
        "selector",
        math.nan,
        route_sum / max(route_count, 1),
        result_dict,
    )
    row["selector"] = "matched_depth_oracle"
    return row


def safe_correlation(function, left, right):
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 2:
        return None
    value = function(left[finite], right[finite])[0]
    return float(value) if np.isfinite(value) else None


def distribution_stats(samples):
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        raise RuntimeError("No finite uncertainty samples were produced")
    return {
        "accepted_detection_count": int(samples.size),
        "finite_accepted_detection_count": int(finite.size),
        "min": float(np.min(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "q75": float(np.quantile(finite, 0.75)),
        "max": float(np.max(finite)),
    }


def selector_depth_metrics(matched, mode, threshold):
    chain_error = np.abs(matched["chain_depth"] - matched["target_depth"])
    parallel_error = np.abs(
        matched["parallel_depth"] - matched["target_depth"])
    if mode == "absolute":
        use_chain = matched["chain_logvar"] < threshold
    elif mode == "delta":
        use_chain = (
            matched["chain_logvar"] - matched["parallel_logvar"] < threshold)
    elif mode == "all_chain":
        use_chain = np.ones_like(chain_error, dtype=bool)
    elif mode == "all_parallel":
        use_chain = np.zeros_like(chain_error, dtype=bool)
    elif mode == "oracle":
        use_chain = chain_error <= parallel_error
    else:
        raise ValueError("Unsupported selector mode: {}".format(mode))

    selected_error = np.where(use_chain, chain_error, parallel_error)
    oracle_error = np.minimum(chain_error, parallel_error)
    chain_is_better = chain_error <= parallel_error
    return {
        "threshold": tau_text(threshold) if threshold is not None else None,
        "chain_rate": float(np.mean(use_chain)),
        "branch_choice_accuracy": float(np.mean(use_chain == chain_is_better)),
        "mae": float(np.mean(selected_error)),
        "rmse": float(np.sqrt(np.mean(selected_error ** 2))),
        "median_absolute_error": float(np.median(selected_error)),
        "mean_regret_vs_oracle": float(np.mean(selected_error - oracle_error)),
    }


def depth_bin_metrics(matched):
    bins = ((0.0, 20.0), (20.0, 40.0), (40.0, math.inf))
    rows = []
    for lower, upper in bins:
        selected = ((matched["target_depth"] >= lower)
                    & (matched["target_depth"] < upper))
        if not np.any(selected):
            continue
        target = matched["target_depth"][selected]
        chain_error = np.abs(matched["chain_depth"][selected] - target)
        parallel_error = np.abs(matched["parallel_depth"][selected] - target)
        rows.append({
            "range_m": "[{:.0f}, {})".format(
                lower, "inf" if math.isinf(upper) else "{:.0f}".format(upper)),
            "count": int(np.count_nonzero(selected)),
            "chain_mae": float(np.mean(chain_error)),
            "parallel_mae": float(np.mean(parallel_error)),
            "oracle_mae": float(np.mean(np.minimum(chain_error, parallel_error))),
            "chain_better_fraction": float(np.mean(chain_error <= parallel_error)),
        })
    return rows


def build_depth_diagnostics(
        matched, configured_mode, configured_tau, best_absolute_tau,
        best_delta_tau):
    chain_error = np.abs(matched["chain_depth"] - matched["target_depth"])
    parallel_error = np.abs(
        matched["parallel_depth"] - matched["target_depth"])
    delta_logvar = matched["chain_logvar"] - matched["parallel_logvar"]
    delta_error = chain_error - parallel_error
    return {
        "matched_object_count": int(matched["target_depth"].size),
        "chain_better_fraction": float(np.mean(chain_error <= parallel_error)),
        "chain": {
            "mae": float(np.mean(chain_error)),
            "rmse": float(np.sqrt(np.mean(chain_error ** 2))),
            "median_absolute_error": float(np.median(chain_error)),
            "pearson_logvar_vs_absolute_error": safe_correlation(
                pearsonr, matched["chain_logvar"], chain_error),
            "spearman_logvar_vs_absolute_error": safe_correlation(
                spearmanr, matched["chain_logvar"], chain_error),
        },
        "parallel": {
            "mae": float(np.mean(parallel_error)),
            "rmse": float(np.sqrt(np.mean(parallel_error ** 2))),
            "median_absolute_error": float(np.median(parallel_error)),
            "pearson_logvar_vs_absolute_error": safe_correlation(
                pearsonr, matched["parallel_logvar"], parallel_error),
            "spearman_logvar_vs_absolute_error": safe_correlation(
                spearmanr, matched["parallel_logvar"], parallel_error),
        },
        "relative_uncertainty": {
            "pearson_delta_logvar_vs_delta_error": safe_correlation(
                pearsonr, delta_logvar, delta_error),
            "spearman_delta_logvar_vs_delta_error": safe_correlation(
                spearmanr, delta_logvar, delta_error),
        },
        "selectors": {
            "all_parallel": selector_depth_metrics(
                matched, "all_parallel", None),
            "all_chain": selector_depth_metrics(
                matched, "all_chain", None),
            "configured_selector": selector_depth_metrics(
                matched, configured_mode, configured_tau),
            "best_absolute_ap": selector_depth_metrics(
                matched, "absolute", best_absolute_tau),
            "best_delta_ap": selector_depth_metrics(
                matched, "delta", best_delta_tau),
            "matched_depth_oracle": selector_depth_metrics(
                matched, "oracle", None),
        },
        "depth_bins": depth_bin_metrics(matched),
    }


def main():
    args = parse_args()
    config_path = absolute_from_root(args.config)
    cfg = prepare_config(config_path, args.batch_size)
    set_random_seed(cfg.get("random_seed", 444))

    model_name = cfg["model_name"]
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = os.path.join(
            cfg["trainer"]["save_path"], model_name, "checkpoint_best.pth")
    checkpoint = absolute_from_root(checkpoint)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            cfg["trainer"]["save_path"], model_name, "tau_calibration")
    output_dir = absolute_from_root(output_dir)
    if os.path.exists(output_dir):
        if not args.overwrite:
            raise FileExistsError(
                "{} already exists; pass --overwrite to replace it".format(output_dir))
        safe_root = os.path.realpath(os.path.join(ROOT_DIR, "outputs"))
        real_output_dir = os.path.realpath(output_dir)
        if (real_output_dir == safe_root
                or os.path.commonpath([safe_root, real_output_dir]) != safe_root):
            raise ValueError(
                "Refusing to overwrite a directory outside outputs/: {}".format(
                    output_dir))
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    print("Config: {}".format(config_path))
    print("Checkpoint: {}".format(checkpoint))
    print("Output: {}".format(output_dir))
    _, val_loader = build_dataloader(cfg["dataset"], workers=args.workers)
    dataset = val_loader.dataset
    if cfg["dataset"]["test_split"] == "test":
        raise ValueError("Calibration requires a labeled validation split, not test")

    model, criterion = build_model(cfg["model"])
    model.to(device)
    load_checkpoint(
        model=model,
        optimizer=None,
        filename=checkpoint,
        map_location=device,
        strict=True,
    )

    topk = int(cfg["tester"].get("topk", 50))
    score_threshold = float(cfg["tester"].get("threshold", 0.2))
    cached, accepted_chain_logvars, accepted_delta_logvars, matched = \
        cache_validation_outputs(
            model, criterion.matcher, val_loader, device, topk,
            score_threshold)
    configured_tau = float(cfg["cop"].get("uncertainty_tau", 0.0))
    configured_selector_mode = str(
        cfg["cop"].get("selector_mode", "absolute")).lower()
    if configured_selector_mode not in {"absolute", "relative"}:
        raise ValueError(
            "cop.selector_mode must be 'absolute' or 'relative'")
    absolute_taus = build_thresholds(
        args.taus, args.num_quantiles, accepted_chain_logvars,
        configured_tau if configured_selector_mode == "absolute" else None)
    delta_taus = build_thresholds(
        args.delta_taus, args.num_delta_quantiles, accepted_delta_logvars,
        configured_tau if configured_selector_mode == "relative" else 0.0)

    chain_logvar_stats = distribution_stats(accepted_chain_logvars)
    delta_logvar_stats = distribution_stats(accepted_delta_logvars)
    print("Accepted chain log-variance: {}".format(chain_logvar_stats))
    print("Accepted delta log-variance: {}".format(delta_logvar_stats))
    print("Evaluating {} absolute and {} delta thresholds...".format(
        len(absolute_taus), len(delta_taus)))

    image_ids = [int(image_id) for image_id in dataset.idx_list]
    gt_annos = kitti.get_label_annos(dataset.label_dir, image_ids)
    absolute_rows = evaluate_routes(
        cached,
        dataset,
        gt_annos,
        output_dir,
        "absolute_selector",
        "tau",
        absolute_taus,
        lambda cache, tau: cache["pred_chain_log_variance"] < tau,
        topk,
        score_threshold,
        args.keep_predictions,
    )
    best_absolute, absolute_csv, absolute_json = write_summary(
        absolute_rows,
        output_dir,
        "summary_absolute",
        checkpoint,
        "tau",
        configured_tau if configured_selector_mode == "absolute" else None,
        chain_logvar_stats,
    )

    delta_rows = evaluate_routes(
        cached,
        dataset,
        gt_annos,
        output_dir,
        "delta_selector",
        "tau_delta",
        delta_taus,
        lambda cache, tau: (
            cache["pred_chain_log_variance"]
            - cache["pred_depth_parallel"][..., 1:2] < tau),
        topk,
        score_threshold,
        args.keep_predictions,
    )
    best_delta, delta_csv, delta_json = write_summary(
        delta_rows,
        output_dir,
        "summary_delta",
        checkpoint,
        "tau_delta",
        configured_tau if configured_selector_mode == "relative" else 0.0,
        delta_logvar_stats,
    )

    oracle_row = evaluate_oracle(
        cached,
        dataset,
        gt_annos,
        output_dir,
        topk,
        score_threshold,
        args.keep_predictions,
    )
    best_absolute_tau = parse_tau(best_absolute["tau"])
    best_delta_tau = parse_tau(best_delta["tau_delta"])
    depth_diagnostics = build_depth_diagnostics(
        matched, configured_selector_mode, configured_tau,
        best_absolute_tau, best_delta_tau)
    diagnostics_path = os.path.join(output_dir, "depth_diagnostics.json")
    with open(diagnostics_path, "w") as stream:
        json.dump(depth_diagnostics, stream, indent=2)
        stream.write("\n")

    all_parallel = next(
        row for row in absolute_rows if row["tau"] == "-inf")
    all_chain = next(
        row for row in absolute_rows if row["tau"] == "+inf")
    comparison = {
        "checkpoint": checkpoint,
        "all_parallel": all_parallel,
        "all_chain": all_chain,
        "best_absolute_selector": best_absolute,
        "best_delta_selector": best_delta,
        "matched_depth_oracle": oracle_row,
        "depth_diagnostics": diagnostics_path,
    }
    comparison_path = os.path.join(output_dir, "comparison.json")
    with open(comparison_path, "w") as stream:
        json.dump(comparison, stream, indent=2)
        stream.write("\n")

    print("\nBest absolute tau: {} (Moderate {:.4f})".format(
        best_absolute["tau"], best_absolute["3d_moderate_R40"]))
    print("Best delta tau: {} (Moderate {:.4f})".format(
        best_delta["tau_delta"], best_delta["3d_moderate_R40"]))
    print("All-chain Moderate: {:.4f}".format(
        all_chain["3d_moderate_R40"]))
    print("Matched-depth oracle Moderate: {:.4f}".format(
        oracle_row["3d_moderate_R40"]))
    print("Absolute summary: {}, {}".format(absolute_csv, absolute_json))
    print("Delta summary: {}, {}".format(delta_csv, delta_json))
    print("Depth diagnostics: {}".format(diagnostics_path))
    print("Comparison: {}".format(comparison_path))


if __name__ == "__main__":
    main()
