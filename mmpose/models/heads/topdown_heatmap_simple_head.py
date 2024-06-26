# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)

from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.core.post_processing import flip_back
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from ..builder import HEADS
import torch.nn.functional as F
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead
import random


@HEADS.register_module()
class TopdownHeatmapSimpleHead(TopdownHeatmapBaseHead):
    """Top-down heatmap simple head. paper ref: Bin Xiao et al. ``Simple
    Baselines for Human Pose Estimation and Tracking``.

    TopdownHeatmapSimpleHead is consisted of (>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.
            - 'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None,
                 in_index=0,
                 input_transform=None,
                 align_corners=False,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None,
                 upsample=0,):
        super().__init__()

        self.in_channels = in_channels
        self.loss = build_loss(loss_keypoint)
        self.upsample = upsample

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[
                -1] if num_deconv_layers > 0 else self.in_channels

            layers = []
            if extra is not None:
                num_conv_layers = extra.get('num_conv_layers', 0)
                num_conv_kernels = extra.get('num_conv_kernels',
                                             [1] * num_conv_layers)

                for i in range(num_conv_layers):
                    layers.append(
                        build_conv_layer(
                            dict(type='Conv2d'),
                            in_channels=conv_channels,
                            out_channels=conv_channels,
                            kernel_size=num_conv_kernels[i],
                            stride=1,
                            padding=(num_conv_kernels[i] - 1) // 2))
                    layers.append(
                        build_norm_layer(dict(type='BN'), conv_channels)[1])
                    layers.append(nn.ReLU(inplace=True))

            layers.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=conv_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding))

            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

    # COCO 데이터셋에서 17개 관절의 번호와 관련된 근처 관절을 정의하는 함수
    def get_related_joints(self, target_joint_num):
    
        joint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

        target_joint = joint_names[target_joint_num]

        related_joints_dict = {
            # "nose": ['left_eye', 'right_eye','left_ear','right_ear'],
            # "left_eye": ['nose', 'left_ear', 'left_shoulder'],
            # "right_eye": ['nose', 'right_ear', 'right_shoulder'],
            # "left_ear": ['left_eye', 'left_shoulder'],
            # "right_ear": ['right_eye', 'right_shoulder'],
            # "left_shoulder": ['nose', 'left_elbow', 'left_hip', 'left_ear'],
            # "right_shoulder": ['nose', 'right_elbow', 'right_hip', 'right_ear'],
            # "left_elbow": ['left_shoulder', 'left_wrist'],
            # "right_elbow": ['right_shoulder', 'right_wrist'],
            # "left_wrist": ['left_elbow'],
            # "right_wrist": ['right_elbow'],
            "left_hip": ['left_shoulder', 'left_knee'],
            "right_hip": ['right_shoulder', 'right_knee'],
            "left_knee": ['left_hip', 'left_ankle'],
            "right_knee": ['right_hip', 'right_ankle'],
            "left_ankle": ['left_knee'],
            "right_ankle": ['right_knee']
        }
        
        related_joints_names = related_joints_dict.get(target_joint, [])

        # 관련된 관절 이름을 관절 번호로 변환
        related_joints_nums = [joint_names.index(name) for name in related_joints_names]

        return related_joints_nums

    def get_max_preds(self, heatmaps):
        """Get predictions from heatmaps.

        Args:
            heatmaps (torch.Tensor[N, K, H, W]): fs with shape (N, K, H, W).

        Returns:
            torch.Tensor[N, K, 2]: Predicted keypoint locations in (x, y) format.
            torch.Tensor[N, K, 1]: Confidence scores (max values) of the keypoints.
        """
        assert isinstance(heatmaps, torch.Tensor)

        # Extract shape information
        N, K, H, W = heatmaps.shape

        # Reshape heatmaps to flatten the spatial dimensions (H, W)
        heatmaps_reshaped = heatmaps.view(N, K, -1)

        # Find the index of the maximum value in each heatmap
        max_indices = heatmaps_reshaped.argmax(dim=2)

        # Convert the 1D index to 2D indices (row, col)
        max_row_indices = max_indices // W
        max_col_indices = max_indices % W

        # Create predicted keypoint locations (x, y)
        preds = torch.stack((max_col_indices.float(), max_row_indices.float()), dim=2)

        # Gather confidence scores (max values) from heatmaps
        max_vals = heatmaps_reshaped.max(dim=2, keepdim=True)[0]

        # Calculate the threshold for the bottom 10% of confidence scores
        confidence_thresholds = []
        for k in range(K):
            # Get confidence scores for a specific keypoint across all samples
            keypoint_scores = max_vals[:, k]

            # Sort the confidence scores
            sorted_scores, _ = torch.sort(keypoint_scores)

            # Calculate the index for the 10th percentile (bottom 10%)
            percentile_index = int(0.1 * len(sorted_scores))

            # Get the confidence score at the percentile index
            confidence_threshold = sorted_scores[percentile_index].item()
            confidence_thresholds.append(confidence_threshold)

        return max_vals, confidence_thresholds
    
    """
    def get_loss(self, output, target, target_weight):
        max_vals, confidence_thresholds = self.get_max_preds(output)

        losses = dict()

        delta = 0.01
        if target is not None:
            cloned_target = target.clone()  # 원본 데이터 복제
            for bi in range(cloned_target.size(0)):
                for ki in range(cloned_target.size(1)):
                    related_joints = self.get_related_joints(ki)
                    summan = 0.0
                    for related_ki in related_joints:
                        summan += delta * cloned_target[bi][related_ki]
                        target[bi][ki] += summan        
                assert not isinstance(self.loss, nn.Sequential)
                assert cloned_target.dim() == 4 and target_weight.dim() == 3
                losses['heatmap_loss'] = self.loss(output, target, target_weight)

        return losses
    """

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """

        losses = dict()

        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 4 and target_weight.dim() == 3
        losses['heatmap_loss'] = self.loss(output, target, target_weight)

        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """

        accuracy = dict()

        if self.target_type == 'GaussianHeatmap':
            _, avg_acc, _ = pose_pck_accuracy(
                output.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                target_weight.detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['acc_pose'] = float(avg_acc)

        return accuracy

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    """
    def inference_model(self, x, flip_pairs=None):
        output = self.forward(x)

        max_vals, confidence_thresholds = self.get_max_preds(output)

        # Convert tensors to numpy arrays
        output_heatmap = output.detach().cpu().numpy()
        max_vals = max_vals.detach().cpu().numpy()

        # Initialize a copy of the output heatmap for manipulation
        modified_heatmap = output_heatmap.copy()

        # Iterate over each keypoint
        for i in range(output_heatmap.shape[0]):
            for k in range(17):
                # Check if the max value of the current keypoint is below the threshold
                if max_vals[i, k, 0] < confidence_thresholds[k]:
                    # Get related joints for the current keypoint
                    related_joints = self.get_related_joints(k)

                    # Mix related joint heatmaps into the current keypoint heatmap with delta
                    for related_ki in related_joints:
                        # Apply delta scaling to the related joint heatmap
                        modified_heatmap[i, k] += 0.01 * output_heatmap[i, related_ki]

        output_heatmap = modified_heatmap

        if flip_pairs is not None:
            output_heatmap = flip_back(
                output_heatmap,
                flip_pairs,
                target_type=self.target_type)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]

        return output_heatmap
    """

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple]):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)

        if flip_pairs is not None:
            output_heatmap = flip_back(
                output.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.target_type)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
        return output_heatmap
    

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.

                - 'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                - 'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                - None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            if not isinstance(inputs, list):
                if self.upsample > 0:
                    inputs = resize(
                        input=F.relu(inputs),
                        scale_factor=self.upsample,
                        mode='bilinear',
                        align_corners=self.align_corners
                        )
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
