import random
import numpy as np
from env.params import SceneConfig
from .utils import save_to_yaml, export_obstacles_to_obj_trimesh

def circle_to_circle_distance(center1, radius1, center2, radius2) -> float:
    """
    计算两个圆形障碍物之间的边缘距离.
    center1, center2: 圆心坐标 (x, y)
    radius1, radius2: 圆的半径
    返回: 两圆边缘距离, 重叠时为 0
    """
    c1 = np.asarray(center1, dtype=float)
    c2 = np.asarray(center2, dtype=float)
    dist_centers = np.linalg.norm(c1 - c2)
    return max(0.0, dist_centers - (radius1 + radius2))

def rect_bounds(center, length, width):
    """
    由轴对齐矩形的中心和长宽计算包围盒.
    length: y 方向尺寸
    width:  x 方向尺寸
    返回: (x_min, x_max, y_min, y_max)
    """
    cx, cy = center
    half_l = length / 2.0
    half_w = width / 2.0
    x_min = cx - half_w
    x_max = cx + half_w
    y_min = cy - half_l
    y_max = cy + half_l
    return x_min, x_max, y_min, y_max

def rectangle_to_rectangle_distance(center1, length1, width1,
                                    center2, length2, width2) -> float:
    """
    计算两个轴对齐矩形障碍物之间的边缘距离.
    center*: (x, y)
    length*: y 方向尺寸
    width*:  x 方向尺寸
    """
    x1_min, x1_max, y1_min, y1_max = rect_bounds(center1, length1, width1)
    x2_min, x2_max, y2_min, y2_max = rect_bounds(center2, length2, width2)

    # x / y 方向的间距, 如果有重叠则为 0
    dist_x = max(0.0, max(x1_min - x2_max, x2_min - x1_max))
    dist_y = max(0.0, max(y1_min - y2_max, y2_min - y1_max))
    return float(np.hypot(dist_x, dist_y))

def circle_to_rectangle_distance(circle_center, circle_radius,
                                 rect_center, rect_length, rect_width) -> float:
    """
    计算圆形障碍物与轴对齐矩形障碍物之间的边缘距离.
    circle_center: 圆心 (x, y)
    circle_radius: 半径
    rect_center:   矩形中心 (x, y)
    rect_length:   y 方向尺寸
    rect_width:    x 方向尺寸
    """
    x_min, x_max, y_min, y_max = rect_bounds(rect_center, rect_length, rect_width)
    cx, cy = circle_center

    # 圆心到矩形最近点
    nearest_x = min(max(cx, x_min), x_max)
    nearest_y = min(max(cy, y_min), y_max)

    d = float(np.hypot(cx - nearest_x, cy - nearest_y))
    return max(0.0, d - circle_radius)

def obstacle_to_wall_distance(obstacle, env_width, env_height) -> float:
    """
    计算一个障碍物与外边界框之间的最小边缘距离.
    外边界框为 [ -env_width/2,  env_width/2 ] x [ -env_height/2,  env_height/2 ].
    """
    ox, oy = obstacle["state"][:2]

    xmin = -env_width / 2.0
    xmax = env_width / 2.0
    ymin = -env_height / 2.0
    ymax = env_height / 2.0

    if obstacle["shape"]["name"] == "circle":
        r = obstacle["shape"]["radius"]
        # 圆心到四条边的距离减半径
        dist_left = ox - xmin
        dist_right = xmax - ox
        dist_bottom = oy - ymin
        dist_top = ymax - oy
        min_edge = min(dist_left, dist_right, dist_bottom, dist_top) - r
        return float(min_edge)

    if obstacle["shape"]["name"] == "rectangle":
        length = obstacle["shape"]["length"]
        width = obstacle["shape"]["width"]
        x_min, x_max, y_min, y_max = rect_bounds((ox, oy), length, width)
        dist_left = x_min - xmin
        dist_right = xmax - x_max
        dist_bottom = y_min - ymin
        dist_top = ymax - y_max
        min_edge = min(dist_left, dist_right, dist_bottom, dist_top)
        return float(min_edge)

    # 其它类型这里不参与生成约束
    return float("inf")

def generate_obstacles(env_width, env_height, min_distance, num_obstacles, circle_radius_range, rect_edge_range):
    """
    在 [-env_width/2, env_width/2] x [-env_height/2, env_height/2] 内随机生成圆形/矩形障碍物.

    约束:
      1) 障碍物之间的边缘距离 >= min_distance
      2) 障碍物到边界框的边缘距离 >= min_distance

    返回:
      obstacle 列表(不包含边界墙), 可直接写入 YAML.
    """
    obstacles = []

    xmin = -env_width / 2.0
    xmax = env_width / 2.0
    ymin = -env_height / 2.0
    ymax = env_height / 2.0

    def is_valid_position(new_state, new_shape) -> bool:
        # 先检查与外边界的距离
        tmp_ob = {"shape": new_shape, "state": new_state}
        if obstacle_to_wall_distance(tmp_ob, env_width, env_height) < min_distance:
            return False

        # 再检查与已有障碍物的距离
        for obs in obstacles:
            shape_a = obs["shape"]["name"]
            shape_b = new_shape["name"]

            if shape_a == "circle" and shape_b == "circle":
                dist = circle_to_circle_distance(
                    obs["state"][:2], obs["shape"]["radius"],
                    new_state[:2], new_shape["radius"],
                )
            elif shape_a == "rectangle" and shape_b == "rectangle":
                dist = rectangle_to_rectangle_distance(
                    obs["state"][:2],
                    obs["shape"]["length"], obs["shape"]["width"],
                    new_state[:2],
                    new_shape["length"], new_shape["width"],
                )
            elif shape_a == "circle" and shape_b == "rectangle":
                dist = circle_to_rectangle_distance(
                    obs["state"][:2], obs["shape"]["radius"],
                    new_state[:2], new_shape["length"], new_shape["width"],
                )
            elif shape_a == "rectangle" and shape_b == "circle":
                dist = circle_to_rectangle_distance(
                    new_state[:2], new_shape["radius"],
                    obs["state"][:2], obs["shape"]["length"], obs["shape"]["width"],
                )
            else:
                dist = float("inf")

            if dist < min_distance:
                return False

        return True

    max_tries = 1000
    tries = 0
    while len(obstacles) < num_obstacles and tries < max_tries:
        tries += 1
        shape_type = random.choice(["circle", "rectangle"])

        if shape_type == "circle":
            radius = random.uniform(circle_radius_range[0], circle_radius_range[1])
            x = random.uniform(xmin + radius, xmax - radius)
            y = random.uniform(ymin + radius, ymax - radius)
            shape = {"name": "circle", "radius": radius}
            state = [x, y, 0.0]
        else:
            length = random.uniform(rect_edge_range[0], rect_edge_range[1])
            width = random.uniform(rect_edge_range[0], rect_edge_range[1])
            x = random.uniform(xmin + width / 2.0, xmax - width / 2.0)
            y = random.uniform(ymin + length / 2.0, ymax - length / 2.0)
            shape = {"name": "rectangle", "length": length, "width": width}
            state = [x, y, 0.0]

        if is_valid_position(state, shape):
            obstacles.append({"shape": shape, "state": state})
    
    # 环境边界框
    boundary = {
        'shape': {'name': 'linestring', 'vertices': [
            [-env_width/2, -env_height/2], [env_width/2, -env_height/2], [env_width/2, env_height/2], [-env_width/2, env_height/2], [-env_width/2, -env_height/2]
        ]},
        'kinematics': {'name': 'static'},
        'state': [0, 0, 0]
    }
    obstacles.append(boundary)

    if len(obstacles) < num_obstacles:
        print(f"warning: only placed {len(obstacles)} / {num_obstacles} obstacles under constraints")

    return obstacles

# 生成随机环境配置
def generate_random_scene(cfg : SceneConfig = SceneConfig()):

    env_config = {
        'world': {
            'width': cfg.scene_params.width,
            'height': cfg.scene_params.height,
            'offset': [-cfg.scene_params.width / 2, -cfg.scene_params.height / 2],
            'step_time': cfg.scene_params.step_time,
            'sample_time': cfg.scene_params.sample_time,
            'collision_mode': cfg.scene_params.collision_mode,
        },
        'robot': [{
            'number': cfg.robot_params.number_robots,
            'kinematics': {'name': cfg.robot_params.kinematics},
            'shape': {'name': cfg.robot_params.shape, 'radius': cfg.robot_params.radius},
            'vel_min': [-1.0, -1.0],
            'vel_max': [1.0, 1.0],
            'color': ['royalblue', 'red', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'lime', 'pink'],
            'arrive_mode': 'position',
            'goal_threshold': cfg.robot_params.goal_threshold,
            'sensors': [{
                'type': 'lidar2d',
                'range_min': cfg.robot_params.lidar_range_min,
                'range_max': cfg.robot_params.lidar_range_max,
                'angle_range': cfg.robot_params.lidar_angle_range,
                'number': cfg.robot_params.lidar_number,
                'alpha': 0.0
            }],
            'plot': {'show_trajectory': False}
        }],
        'obstacle': generate_obstacles(cfg.scene_params.width, cfg.scene_params.height,
                                        cfg.scene_params.min_distance_betw_obs, cfg.scene_params.number_obstacles,
                                        [cfg.scene_params.circle_radius_min, cfg.scene_params.circle_radius_max],
                                        [cfg.scene_params.rect_edge_min, cfg.scene_params.rect_edge_max])
    }

    return env_config

# 示例：生成一个环境并保存到文件
if __name__ == "__main__":
    yaml_save_path = 'config/random_env.yaml'
    obj_save_path = 'config/random_env.obj'
    
    scene_cfg = SceneConfig()
    # 生成随机环境配置
    random_env_config = generate_random_scene(cfg=scene_cfg)
    
    # 将配置保存到YAML文件
    save_to_yaml(random_env_config, yaml_save_path)

    export_obstacles_to_obj_trimesh(
        random_env_config["obstacle"],
        obj_save_path,
        thickness=2.0,
        wall_thickness=0.2
    )
