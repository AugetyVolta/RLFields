import irsim

env = irsim.make('config/random_env.yaml') # initialize the environment with the configuration file
env.reset()

while True:
    env.render()
