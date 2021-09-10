#***************************************************************************************/
#
#    Based on MMDetection3D Library (Apache 2.0 license):
#    https://github.com/open-mmlab/mmdetection3d
#
#    Copyright 2018-2019 Open-MMLab.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#***************************************************************************************/

import torch
import pickle
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from functools import partial
import numpy as np
import os

from open3d.ml.torch.ops import voxelize, ragged_to_dense
import MinkowskiEngine as ME

from .base_model_objdet import BaseModel
from .point_pillars import PointPillars, PointPillarsVoxelization, PointPillarsScatter, PillarFeatureNet, SECOND, SECONDFPN, Anchor3DHead
from .sparse_point_pillars import SparseSECOND, SparseSECONDFPN, SparsePointPillarsScatter, SparsePointPillars

from ...utils import MODEL
from ..utils.objdet_helper import Anchor3DRangeGenerator, BBoxCoder, multiclass_nms, limit_period, get_paddings_indicator, bbox_overlaps, box3d_to_bev2d
from ..modules.losses.focal_loss import FocalLoss
from ..modules.losses.smooth_L1 import SmoothL1Loss
from ..modules.losses.cross_entropy import CrossEntropyLoss
from ...datasets.utils import ObjdetAugmentation, BEVBox3D
from ...datasets.utils.operations import filter_by_min_points


class SparsePointPillarsWide(SparsePointPillars):
    """Object detection model. Based on the PointPillars architecture
    https://github.com/nutonomy/second.pytorch.

    Args:
        name (string): Name of model.
            Default to "SparsePointPillarsWide".
        voxel_size: voxel edge lengths with format [x, y, z].
        point_cloud_range: The valid range of point coordinates as
            [x_min, y_min, z_min, x_max, y_max, z_max].
        voxelize: Config of PointPillarsVoxelization module.
        voxelize_encoder: Config of PillarFeatureNet module.
        scatter: Config of PointPillarsScatter module.
        backbone: Config of backbone module (SECOND).
        neck: Config of neck module (SECONDFPN).
        head: Config of anchor head module.
    """

    def __init__(self,
                 name="SparsePointPillarsWide",
                 device="cuda",
                 point_cloud_range=[0, -40.0, -3, 70.0, 40.0, 1],
                 classes=['car'],
                 voxelize={},
                 voxel_encoder={},
                 scatter={},
                 backbone={},
                 neck={},
                 head={},
                 loss={},
                 **kwargs):

        super().__init__(name=name,
                         device=device,
                         point_cloud_range=point_cloud_range,
                         classes=classes,
                         voxelize=voxelize,
                         voxel_encoder=voxel_encoder,
                         scatter=scatter,
                         backbone=backbone,
                         neck=neck,
                         head=head,
                         loss=loss,
                         **kwargs)
        self.backbone = SparseSECONDWide(**backbone)
        self.to(self.device)


MODEL._register_module(SparsePointPillarsWide, 'torch')

class SparseSECONDWide(SparseSECOND):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
    """

    def __init__(self,
                 in_channels=64,
                 out_channels=[64, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2]):
        super(SparseSECOND, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                ME.MinkowskiConvolution(in_filters[i],
                                        out_channels[i],
                                        9,
                                        bias=False,
                                        stride=layer_strides[i],
                                        dimension=2,
                                        expand_coordinates=True),
                ME.MinkowskiBatchNorm(out_channels[i], eps=1e-3, momentum=0.01),
                ME.MinkowskiReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    ME.MinkowskiConvolution(out_channels[i],
                                            out_channels[i],
                                            3,
                                            bias=False,
                                            dimension=2))
                block.append(
                    ME.MinkowskiBatchNorm(out_channels[i],
                                          eps=1e-3,
                                          momentum=0.01))
                block.append(ME.MinkowskiReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.layer_strides = layer_strides


