import numpy as np
from scipy.spatial.distance import cdist

from .esdf_2d_simple import ESDF_2d


class ESDF_2d_Safe(ESDF_2d):
    """
    兼容空点云/全被 r_max 过滤掉的点云。

    触发场景：
      - lidar_to_pointcloud 返回空（全是 inf / nan 被丢弃）
      - 或者点云全在 r_max 之外，经过过滤后为空

    处理策略：
      - 若点云为空，返回整张 sdf_grid = r_max（表示“障碍物很远/未知”）。
        后续会 clip 到 [0, r_max] 并归一化，得到全 1.0 的 free-space 图。
    """

    def generate_sdf_from_pointcloud(self, pointcloud):
        # 网格范围
        x_min, x_max, y_min, y_max = self.grid_range[0], self.grid_range[1], self.grid_range[2], self.grid_range[3]

        # 生成网格坐标
        x_grid, y_grid = np.meshgrid(
            np.arange(x_min, x_max, self.grid_size),
            np.arange(y_min, y_max, self.grid_size),
        )

        pc = np.asarray(pointcloud, dtype=np.float32)
        if pc.size == 0:
            self.sdf_grid = np.full_like(x_grid, fill_value=float(self.r_max), dtype=np.float32)
            return self.sdf_grid

        # 期望 pc: [N,2]；如果传成了 [2,N]，尝试转置一次
        if pc.ndim == 2 and pc.shape[0] == 2 and pc.shape[1] != 2:
            pc = pc.T

        # 筛选距离小于 r_max 的点（可能导致为空）
        valid = (np.linalg.norm(pc, axis=1) <= self.r_max)
        pc = pc[valid]
        if pc.shape[0] == 0:
            self.sdf_grid = np.full_like(x_grid, fill_value=float(self.r_max), dtype=np.float32)
            return self.sdf_grid

        # 将网格坐标展平并堆叠
        grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T.astype(np.float32)

        # 计算距离并取最小
        distances = cdist(grid_points, pc, metric='euclidean')
        min_distances = np.min(distances, axis=1)

        self.sdf_grid = min_distances.reshape(x_grid.shape).astype(np.float32)
        return self.sdf_grid
