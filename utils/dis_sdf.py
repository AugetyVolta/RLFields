# import irsim
from env.sim.esdf_2d import Costmap_2d, CostmapCfg
from env.sim.esdf_2d_simple import ESDF_2d, ESDF_Cfg
from env.sim.marl_sim_env import MARL_SIM
from utils.utils import lidar_to_pointcloud

# env = irsim.make('config/test.yaml') # initialize the environment with the configuration file
# env.reset()

cfg = CostmapCfg()
esdf = Costmap_2d(params=cfg, device='cpu')

cfg = ESDF_Cfg()
esdf = ESDF_2d(cfg)

sim = MARL_SIM('config/random_env.yaml', disable_plotting=False)
t = 0
sim.reset()
while True:
    t+=1
    sim.env.step([[0.1, 0.1]], action_id=[0])   # update the environment
    # sim.env.step()
    sim.env.render() # render the environment

    # sim.robot_list[0].get_lidar_scan()
    # 假设 lidar 是一个Lidar2D对象
    lidar_data = sim.env.robot.lidar.get_scan()
    pointcloud = lidar_to_pointcloud(lidar_data)
    # print(pointcloud.shape)

    # points = torch.from_numpy(pointcloud).reshape(-1, 2).unsqueeze(0)
    # print(points.shape)
    # esdf.generate(points)
    # esdf.inflate()

    # esdf.debug_plot(inflated=True)
    sdf_grid = esdf.generate_sdf_from_pointcloud(pointcloud)

    esdf.plot()

    if sim.env.robot_list[0].collision:
        # env.reset()
        sim.reset()
        t = 0
    if (t%100) == 0:
        t = 0
        sim.reset()
    # if env.done(): 
    #     break # check if the simulation is done
        
# env.end() # close the environment