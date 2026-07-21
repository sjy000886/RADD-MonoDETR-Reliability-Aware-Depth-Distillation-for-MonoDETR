import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherDepthFeatureLoss(nn.Module):
    """Regularize student depth features with local teacher-depth geometry.

    CompletionFormer predicts visible surface depth, whereas MonoDETR's native
    DDN target represents object-centre depth inside each 2D box.  To avoid
    forcing those two quantities into the same logits, this loss uses teacher
    depth only to describe whether neighbouring locations should have similar
    features.  It has no learnable parameters and is used only during training.
    """

    def __init__(self, cfg):
        super().__init__()
        self.loss_type = cfg.get("loss_type", "feature_affinity").lower()
        self.depth_min = float(cfg.get("min_depth", 1e-3))
        self.depth_max = float(cfg.get("max_depth", 60.0))
        self.target_interpolation = cfg.get(
            "target_interpolation", "masked_bilinear").lower()
        self.min_valid_ratio = float(cfg.get("min_valid_ratio", 0.5))
        self.affinity_temperature = float(
            cfg.get("affinity_temperature", 0.1))
        self.smooth_l1_beta = float(cfg.get("smooth_l1_beta", 0.1))
        self.feature_normalization_eps = float(
            cfg.get("feature_normalization_eps", 1e-6))
        self.eps = float(cfg.get("eps", 1e-6))

        spatial_cfg = cfg.get("spatial_weighting", {})
        self.spatial_weighting_enabled = bool(spatial_cfg.get("enabled", True))
        self.background_weight = float(spatial_cfg.get("background_weight", 0.1))
        self.background_start_ratio = float(
            spatial_cfg.get("background_start_ratio", 0.45))
        self.box_edge_weight = float(spatial_cfg.get("box_edge_weight", 0.3))
        self.box_interior_weight = float(
            spatial_cfg.get("box_interior_weight", 1.0))
        self.box_center_weight = float(spatial_cfg.get("box_center_weight", 1.5))
        self.box_center_ratio = float(spatial_cfg.get("box_center_ratio", 0.5))
        self.box_edge_ratio = float(spatial_cfg.get("box_edge_ratio", 0.1))

        if self.loss_type != "feature_affinity":
            raise ValueError(
                "teacher_depth.loss_type must be 'feature_affinity', got '{}'"
                .format(self.loss_type))
        if self.target_interpolation not in {"nearest", "masked_bilinear"}:
            raise ValueError(
                "teacher_depth.target_interpolation must be 'nearest' or "
                "'masked_bilinear'")
        if self.depth_min <= 0 or self.depth_max <= self.depth_min:
            raise ValueError("Teacher depth range must satisfy 0 < min_depth < max_depth")
        if self.affinity_temperature <= 0:
            raise ValueError("affinity_temperature must be positive")
        if self.smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        if self.feature_normalization_eps <= 0 or self.eps <= 0:
            raise ValueError("Feature regularization eps values must be positive")
        spatial_weights = [
            self.background_weight, self.box_edge_weight,
            self.box_interior_weight, self.box_center_weight]
        if any(weight < 0 for weight in spatial_weights):
            raise ValueError("Teacher spatial weights must be non-negative")
        if not 0 <= self.background_start_ratio <= 1:
            raise ValueError("background_start_ratio must be in [0, 1]")
        if not 0 < self.box_center_ratio <= 1:
            raise ValueError("box_center_ratio must be in (0, 1]")
        if not 0 <= self.box_edge_ratio < 0.5:
            raise ValueError("box_edge_ratio must be in [0, 0.5)")

    def _resize_target(self, depth, valid, confidence, size):
        if depth.shape[-2:] == size:
            return depth[:, 0], valid[:, 0].bool(), confidence[:, 0]

        if self.target_interpolation == "nearest":
            depth = F.interpolate(depth, size=size, mode="nearest")
            valid = F.interpolate(valid.float(), size=size, mode="nearest") > 0.5
            confidence = F.interpolate(confidence, size=size, mode="nearest")
            return depth[:, 0], valid[:, 0], confidence[:, 0]

        valid_float = valid.float()
        denominator = F.interpolate(
            valid_float, size=size, mode="bilinear", align_corners=False)
        depth = F.interpolate(
            depth * valid_float, size=size, mode="bilinear", align_corners=False
        ) / denominator.clamp_min(self.eps)
        confidence = F.interpolate(
            confidence * valid_float, size=size, mode="bilinear",
            align_corners=False) / denominator.clamp_min(self.eps)
        valid = denominator >= self.min_valid_ratio
        return depth[:, 0], valid[:, 0], confidence[:, 0]

    def build_spatial_weight(self, target_boxes, shape, device, dtype):
        """Build background/edge/interior/centre weights on the feature grid."""
        batch_size, height, width = shape
        if not self.spatial_weighting_enabled:
            return torch.ones(shape, device=device, dtype=dtype)

        y_coordinate = (
            torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        x_coordinate = (
            torch.arange(width, device=device, dtype=dtype) + 0.5) / width
        y_grid, x_grid = torch.meshgrid(y_coordinate, x_coordinate)
        spatial_weight = torch.zeros(shape, device=device, dtype=dtype)

        background_mask = y_grid >= self.background_start_ratio
        spatial_weight[:, background_mask] = self.background_weight

        inner_limit = 1.0 - 2.0 * self.box_edge_ratio
        for batch_index, boxes in enumerate(target_boxes):
            for box in boxes:
                center_x, center_y, box_width, box_height = box
                normalized_x = torch.abs(x_grid - center_x) / (
                    box_width.clamp_min(self.eps) * 0.5)
                normalized_y = torch.abs(y_grid - center_y) / (
                    box_height.clamp_min(self.eps) * 0.5)
                inside = (normalized_x <= 1.0) & (normalized_y <= 1.0)
                interior = (
                    (normalized_x <= inner_limit)
                    & (normalized_y <= inner_limit))
                center = (
                    (normalized_x <= self.box_center_ratio)
                    & (normalized_y <= self.box_center_ratio))

                object_weight = torch.zeros_like(x_grid)
                object_weight = torch.where(
                    inside, torch.full_like(object_weight, self.box_edge_weight),
                    object_weight)
                object_weight = torch.where(
                    interior,
                    torch.full_like(object_weight, self.box_interior_weight),
                    object_weight)
                object_weight = torch.where(
                    center,
                    torch.full_like(object_weight, self.box_center_weight),
                    object_weight)
                spatial_weight[batch_index] = torch.maximum(
                    spatial_weight[batch_index], object_weight)
        return spatial_weight

    def _direction_statistics(self, first_feature, second_feature,
                              first_depth, second_depth, first_valid,
                              second_valid, first_confidence,
                              second_confidence, first_spatial,
                              second_spatial):
        student_similarity = (first_feature * second_feature).sum(dim=1)
        teacher_log_difference = torch.abs(
            torch.log(first_depth.clamp_min(self.depth_min))
            - torch.log(second_depth.clamp_min(self.depth_min)))
        teacher_similarity = torch.exp(
            -teacher_log_difference / self.affinity_temperature)

        pair_valid = first_valid & second_valid
        pair_confidence = torch.minimum(first_confidence, second_confidence)
        pair_spatial = torch.minimum(first_spatial, second_spatial)
        pair_weight = pair_confidence * pair_spatial
        # pair_weight = (
        #     pair_confidence.clamp_min(0)
        #     * first_spatial.clamp_min(0)
        #     * second_spatial.clamp_min(0))
        pair_weight = torch.where(
            pair_valid & torch.isfinite(pair_weight), pair_weight,
            torch.zeros_like(pair_weight))

        pair_loss = F.smooth_l1_loss(
            student_similarity, teacher_similarity, reduction="none",
            beta=self.smooth_l1_beta)
        return (pair_loss * pair_weight).sum(), pair_weight.sum()

    def forward(self, student_feature, target_depth, target_valid,
                target_confidence, target_boxes):
        if student_feature.ndim != 4:
            raise ValueError(
                "Teacher feature regularization expects BxCxHxW features, got {}"
                .format(tuple(student_feature.shape)))

        target_depth, target_valid, target_confidence = self._resize_target(
            target_depth, target_valid, target_confidence,
            student_feature.shape[-2:])
        target_valid = (
            target_valid.bool()
            & torch.isfinite(target_depth)
            & torch.isfinite(target_confidence)
            & (target_confidence > 0)
            & (target_depth >= self.depth_min)
            & (target_depth <= self.depth_max))

        spatial_weight = self.build_spatial_weight(
            target_boxes, target_depth.shape, target_depth.device,
            target_depth.dtype)
        feature = F.normalize(
            student_feature, p=2, dim=1, eps=self.feature_normalization_eps)

        horizontal = self._direction_statistics(
            feature[:, :, :, :-1], feature[:, :, :, 1:],
            target_depth[:, :, :-1], target_depth[:, :, 1:],
            target_valid[:, :, :-1], target_valid[:, :, 1:],
            target_confidence[:, :, :-1], target_confidence[:, :, 1:],
            spatial_weight[:, :, :-1], spatial_weight[:, :, 1:])
        vertical = self._direction_statistics(
            feature[:, :, :-1, :], feature[:, :, 1:, :],
            target_depth[:, :-1, :], target_depth[:, 1:, :],
            target_valid[:, :-1, :], target_valid[:, 1:, :],
            target_confidence[:, :-1, :], target_confidence[:, 1:, :],
            spatial_weight[:, :-1, :], spatial_weight[:, 1:, :])

        loss_sum = horizontal[0] + vertical[0]
        weight_sum = horizontal[1] + vertical[1]
        return loss_sum / (weight_sum + self.eps)
