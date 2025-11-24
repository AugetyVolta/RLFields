import numpy as np
import yaml
import math
import trimesh
import torch
from shapely.geometry import Point, LineString

def lidar_to_pointcloud(lidar_data, min_range: float = 0.2) -> np.ndarray:
    # 获取激光束的角度范围和距离数据
    angle_min = lidar_data['angle_min']
    angle_increment = lidar_data['angle_increment']
    ranges = np.array(lidar_data['ranges'], dtype=np.float32)
    
    # 计算每个激光束的角度
    angles = angle_min + np.arange(len(ranges)) * angle_increment
    
    # 过滤掉无效或小于最小距离的激光束
    mask = np.isfinite(ranges) & (ranges > min_range)
    ranges = ranges[mask]
    angles = angles[mask]
    
    # 计算每个激光束的 x, y 坐标
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    
    return np.vstack((x, y)).astype(np.float32).T   # 生成2D点云

# 自定义 Dumper：小 dict / list 用行内风格
class PrettyDumper(yaml.Dumper):
    """让 dict 总是多行缩进，list 里全是标量时用行内格式。"""
    pass

def _repr_dict(dumper, data):
    # 关键：flow_style=False，所有 dict 都用多行缩进格式
    return dumper.represent_mapping(
        "tag:yaml.org,2002:map", data, flow_style=False
    )

def _repr_list(dumper, data):
    # 如果列表元素都是标量，就用行内 [a, b, c]，否则用多行
    scalar_types = (int, float, str, bool)
    flow = all(isinstance(x, scalar_types) for x in data)
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=flow
    )

PrettyDumper.add_representer(dict, _repr_dict)
PrettyDumper.add_representer(list, _repr_list)

def save_to_yaml(config, filename="random_env.yaml"):
    with open(filename, "w") as f:
        yaml.dump(
            config,
            f,
            Dumper=PrettyDumper,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

def export_obstacles_to_obj_trimesh(
    obstacles,
    obj_path: str,
    thickness: float = 2.0,
    wall_thickness: float = 0.1,
):
    """
    用 trimesh 将 2D 障碍物列表导出为 3D obj.
    障碍物和墙都有厚度.

    thickness: z 方向高度
    wall_thickness: 墙在平面内的厚度
    """
    meshes = []

    # 圆形和矩形
    for obs in obstacles:
        shape = obs["shape"]
        name = shape["name"]

        # 边界的 linestring 单独处理
        if name in ("circle", "rectangle"):
            cx, cy = obs["state"][:2]

        if name == "circle":
            radius = shape["radius"]
            # 用 cylinder 近似圆柱
            m = trimesh.creation.cylinder(
                radius=radius,
                height=thickness,
            )
            # cylinder 默认中心在 (0,0,0)，z 方向高度，平移到 (cx, cy, 0)
            m.apply_translation([cx, cy, 0.0])
            meshes.append(m)

        elif name == "rectangle":
            length = shape["length"]  # x 方向
            width = shape["width"]    # y 方向 对于irsim里面的参数约定
            # box 在 x,y,z 方向的尺寸
            m = trimesh.creation.box(extents=[length, width, thickness])
            # box 默认中心在 (0,0,0)，平移到 (cx, cy, 0)
            m.apply_translation([cx, cy, 0.0])
            meshes.append(m)

    # 处理边界墙 linestring
    for obs in obstacles:
        shape = obs["shape"]
        if shape["name"] != "linestring":
            continue

        vertices_2d = shape["vertices"]
        line = LineString(vertices_2d)

        coords = list(line.coords)
        for i in range(len(coords) - 1):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len <= 0:
                continue

            # 边中点
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)

            # box 在局部坐标中的尺寸: x 方向 seg_len, y 方向 wall_thickness
            box = trimesh.creation.box(extents=[seg_len, wall_thickness, thickness])

            # 边方向角度
            angle = math.atan2(y1 - y0, x1 - x0)
            rot = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])

            # 先旋转，再平移到“边中点”
            trans = trimesh.transformations.translation_matrix([cx, cy, 0.0])
            transform = trans @ rot
            box.apply_transform(transform)

            # 计算“内法向”并沿内法向偏移 half thickness
            # 你的边界 vertices 是按逆时针给的：
            # [-w/2,-h/2] -> [w/2,-h/2] -> [w/2,h/2] -> [-w/2,h/2] -> ...
            # 对这样的边，左侧是内部，所以内法向 = 左法向
            nx = -(y1 - y0) / seg_len   # 左法向 x 分量
            ny =  (x1 - x0) / seg_len   # 左法向 y 分量

            # 向内偏移 wall_thickness/2
            box.apply_translation([nx * wall_thickness / 2.0,
                                   ny * wall_thickness / 2.0,
                                   0.0])
            meshes.append(box)

    if not meshes:
        print("没有任何障碍物可导出")
        return

    combined = trimesh.util.concatenate(meshes)
    combined.export(obj_path)
    print(f"trimesh OBJ 导出完成: {obj_path}")