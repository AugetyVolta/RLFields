# esdf_2d_simple.py
# -*- coding: utf-8 -*-

"""
ESDF（Signed Distance Field）生成器

考虑展示范围外的障碍物信息, 直接从Lidar数据转换sdf

功能说明：
- `ESDF_Cfg`: 配置类，定义了网格的范围、分辨率、最大有效距离等参数。
- `ESDF_2d`: 主要类，包含生成SDF的方法 `generate_sdf_from_pointcloud` 和可视化方法 `plot`。

使用方法：
1. 创建 `ESDF_Cfg` 配置类来定义网格的范围、分辨率等参数。
2. 使用 `ESDF_2d` 类实例化对象并传入配置类。
3. 调用 `generate_sdf_from_pointcloud` 方法传入点云数据生成SDF。
4. 调用 `plot` 方法可视化SDF，并将图像保存到指定的文件中。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from ..params import ESDF_Cfg

class ESDF_2d:
    def __init__(self, cfg: ESDF_Cfg = ESDF_Cfg()):
        """
        初始化SDF生成器。

        参数：
            cfg: ESDF配置对象，用于设置网格大小、分辨率等参数。
        """
        self.cfg = cfg
        self.grid_size = cfg.resolution
        self.grid_range = (cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max)
        self.r_max = cfg.r_max
        self.sdf_grid = None  # 用于存储生成的SDF网格

    def generate_sdf_from_pointcloud(self, pointcloud):
        """
        将点云转换为SDF（Signed Distance Field）。

        参数：
            pointcloud: Nx2的点云数组，N为点的个数。

        返回：
            SDF网格
        """
        # 网格范围
        x_min, x_max, y_min, y_max = self.grid_range[0], self.grid_range[1], self.grid_range[2], self.grid_range[3]
        
        # 生成网格坐标
        x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, self.grid_size), 
                                     np.arange(y_min, y_max, self.grid_size))
        
        # 筛选距离小于r_max的点
        valid = (np.linalg.norm(pointcloud, axis=1) <= self.r_max)
        pointcloud = pointcloud[valid]
        
        # 将网格坐标展平并堆叠
        grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
        
        # 计算每个网格点到点云中所有点的距离
        distances = cdist(grid_points, pointcloud, metric='euclidean')
        
        # 对于每个网格点，取到最近的点云距离
        min_distances = np.min(distances, axis=1)
        
        # 计算SDF: 在点云内部距离为负，在外部为正
        self.sdf_grid = min_distances.reshape(x_grid.shape)
        
        return self.sdf_grid

    def plot(self, fname='logs/esdf_image.png'):
        """
        可视化生成的SDF（Signed Distance Field）网格。
        """
        if self.sdf_grid is None:
            raise RuntimeError("请先调用 generate_sdf_from_pointcloud() 生成 SDF 网格")

        plt.figure(figsize=(6, 6))
        plt.imshow(self.sdf_grid, cmap="plasma", origin="lower", extent=self.grid_range*2)
        plt.colorbar(label="Distance to nearest obstacle [m]")
        plt.title("Signed Distance Field (SDF)")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.1, dpi=160)
        plt.close()