# Sparse PointPillars

This is the official repo for our implementation of _Sparse PointPillars: Maintaining and Exploiting Input Sparsity to Improve Runtime on Embedded Systems_, accepted to IROS 2022.

It is based on the [Open3D-ML](https://github.com/isl-org/Open3D-ML) codebase.

## Datasets Used

For our experiments we used [KITTI](http://www.cvlibs.net/datasets/kitti/), a standard 3D object detection datset, and [Matterport-Chair](https://github.com/kylevedder/MatterportDataSampling), a 3D chair detection task dataset generated from multiple houses in the [Matterport3D](https://niessner.github.io/Matterport/) dataset. Model weights and data files are available on our [project page](https://vedder.io/sparse_point_pillars).

## Citation

Please cite our work ([pdf](https://arxiv.org/abs/2106.06882)) if you use Sparse PointPillars.

```bib
@article{Vedder2022,
    author    = {Kyle Vedder and Eric Eaton},
    title     = {{Sparse PointPillars: Maintaining and Exploiting Input Sparsity to Improve Runtime on Embedded Systems}},
    journal   = {International Conference on Intelligent Robots and Systems (IROS)},
    year      = {2022},
}
```
