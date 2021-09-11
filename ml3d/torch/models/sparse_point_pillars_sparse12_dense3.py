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


class SparsePointPillarsSparse12Dense3(SparsePointPillars):
    """Object detection model. Based on the PointPillars architecture
    https://github.com/nutonomy/second.pytorch.

    Args:
        name (string): Name of model.
            Default to "SparsePointPillarsSparse1Dense23".
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
                 name="SparsePointPillarsSparse12Dense3",
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
        self.backbone = SparseSECONDSparse12Dense3(**backbone)
        self.neck = SparseSECONDFPNSparse12Dense3(**neck)

        print("type(self.middle_encoder)", type(self.middle_encoder))
        print("type(self.backbone)", type(self.backbone))
        print("type(self.neck)", type(self.neck))
        print("type(self.bbox_head)", type(self.bbox_head))
        self.to(self.device)


MODEL._register_module(SparsePointPillarsSparse12Dense3, 'torch')


class SparseSECONDSparse12Dense3(SparseSECOND):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
    """

    def make_sparse_block(self, in_filters, out_channels, layer_strides,
                          layer_num):
        block = [
            ME.MinkowskiConvolution(in_filters,
                                    out_channels,
                                    2,
                                    bias=False,
                                    stride=layer_strides,
                                    dimension=2,
                                    expand_coordinates=True),
            ME.MinkowskiBatchNorm(out_channels, eps=1e-3, momentum=0.01),
            ME.MinkowskiReLU(inplace=True),
        ]
        for j in range(layer_num):
            block.append(
                ME.MinkowskiConvolution(out_channels,
                                        out_channels,
                                        3,
                                        bias=False,
                                        dimension=2))
            block.append(
                ME.MinkowskiBatchNorm(out_channels, eps=1e-3, momentum=0.01))
            block.append(ME.MinkowskiReLU(inplace=True))

        block = nn.Sequential(*block)
        return block

    def make_dense_block(self, in_filters, out_channels, layer_strides,
                         layer_num):
        block = [
            nn.Conv2d(in_filters,
                      out_channels,
                      3,
                      bias=False,
                      stride=layer_strides,
                      padding=1),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        ]
        for j in range(layer_num):
            block.append(
                nn.Conv2d(out_channels, out_channels, 3, bias=False, padding=1))
            block.append(nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01))
            block.append(nn.ReLU(inplace=True))

        block = nn.Sequential(*block)
        return block

    def __init__(self,
                 in_channels=64,
                 out_channels=[64, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2]):
        super(SparseSECONDSparse12Dense3, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            if i <= 1:
                block = self.make_sparse_block(in_filters[i], out_channels[i],
                                               layer_strides[i], layer_num)
            else:
                block = self.make_dense_block(in_filters[i], out_channels[i],
                                              layer_strides[i], layer_num)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.layer_strides = layer_strides

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        input_shape = x.shape
        vals = x._values()
        idxs = x._indices().permute(1, 0).contiguous().int()
        x = ME.SparseTensor(vals, idxs)
        outs = []
        for i in range(len(self.blocks)):
            if i == 2:
                dense_shape = torch.Size([input_shape[0], input_shape[3], input_shape[1] // 4, input_shape[2] // 4])
                x = x.dense(min_coordinate=torch.zeros((2,), dtype=torch.int), shape=dense_shape)[0]
            x = self.blocks[i](x)
            outs.append(x)
        return input_shape, self.layer_strides, tuple(outs)


class ToDenseMink(nn.Module):

    def __init__(self, input_shape, first_shrink_stride, first_upsample_stride,
                 out_size):
        super(ToDenseMink, self).__init__()
        batch_size, x_size, y_size, _ = input_shape
        scale = first_shrink_stride // first_upsample_stride
        self.output_shape = torch.Size(
            [batch_size, out_size, x_size // scale, y_size // scale])
        self.min_coord = torch.zeros((2, ), dtype=torch.int)

    def forward(self, x):
        return x.dense(shape=self.output_shape,
                       min_coordinate=self.min_coord)[0]


class SparseSECONDFPNSparse12Dense3(nn.Module):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def make_dense_deblock(self, stride, in_channels, out_channel,
                           use_conv_for_no_stride):
        if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
            upsample_layer = nn.ConvTranspose2d(in_channels=in_channels,
                                                out_channels=out_channel,
                                                kernel_size=stride,
                                                stride=stride,
                                                bias=False)
        else:
            stride = np.round(1 / stride).astype(np.int64)
            upsample_layer = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channel,
                                       kernel_size=stride,
                                       stride=stride,
                                       bias=False)

        deblock = nn.Sequential(
            upsample_layer, nn.BatchNorm2d(out_channel,
                                           eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True))
        return deblock

    def make_sparse_deblock(self, stride, in_channels, out_channel,
                            use_conv_for_no_stride):
        if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
            upsample_layer = ME.MinkowskiConvolutionTranspose(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=stride,
                stride=stride,
                bias=False,
                dimension=2)
        else:
            stride = np.round(1 / stride).astype(np.int64)
            upsample_layer = ME.MinkowskiConvolution(in_channels=in_channels,
                                                     out_channels=out_channel,
                                                     kernel_size=stride,
                                                     stride=stride,
                                                     bias=False,
                                                     dimension=2)

        deblock = nn.Sequential(
            upsample_layer,
            ME.MinkowskiBatchNorm(out_channel, eps=1e-3, momentum=0.01),
            ME.MinkowskiReLU(inplace=True))
        return deblock

    def __init__(self,
                 in_channels=[64, 128, 256],
                 out_channels=[128, 128, 128],
                 upsample_strides=[1, 2, 4],
                 use_conv_for_no_stride=False):
        super(SparseSECONDFPNSparse12Dense3, self).__init__()
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if i <= 1:
                deblock = self.make_sparse_deblock(stride, in_channels[i],
                                              out_channel,
                                              use_conv_for_no_stride)
            else:
                deblock = self.make_dense_deblock(stride, in_channels[i],
                                              out_channel,
                                              use_conv_for_no_stride)
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)
        self.init_weights()
        self.upsample_strides = upsample_strides

    def init_weights(self):
        """Initialize weights of FPN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            torch.Tensor: Feature maps.
        """
        input_shape, layer_strides, x = x
        # assert len(x) == len(self.in_channels)
        to_dense = ToDenseMink(input_shape, layer_strides[0],
                               self.upsample_strides[0], self.out_channels[0])
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
        ups[0] = to_dense(ups[0])
        ups[1] = to_dense(ups[1])
        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return out