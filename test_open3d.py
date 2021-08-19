import numpy as np
from open3d.ml.contrib import iou_3d_cpu

a1 = np.array([[0., 0., 0., 1., 1., 1., 0.], [0., 0., 0., 1., 1., 1.,
                                              0.]]).astype(np.float32)
a2 = np.array([[0., 0., 0., 1., 1., 1., 0.], [0., 0., 0., 1., 1., 1., 0.],
               [0., 0., 0., 1., 1., 1., 0.]]).astype(np.float32)
a3 = np.array([[3., 0., 0., 1., 1., 1., 0.]]).astype(np.float32)
print(iou_3d_cpu(a1, a2).shape)
print(iou_3d_cpu(a1, a3))
