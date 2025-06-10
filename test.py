import numpy as np
import robosuite as suite
from cam_policy import CamPolicy
from robosuite.utils.placement_samplers import UniformRandomSampler

placement_initializer = UniformRandomSampler(
    name="FixedOriSampler",
    mujoco_objects=None,            
    x_range=[-0.115, -0.11],       
    y_range=[0.05, 0.225],
    rotation=np.pi,
    rotation_axis="z",
    ensure_object_boundary_in_range=False,
    ensure_valid_placement=False,
    reference_pos=(0,0,0.82),
    z_offset=0.02,
)

env = suite.make(
    env_name="Stack", 
    robots="Panda", 
    has_renderer=True,
    render_camera="frontview", # ('frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand')
    has_offscreen_renderer=False,
    use_camera_obs=False,
    #placement_initializer=placement_initializer,
    ignore_done=True  
)


success_rate = 0
# reset the environment
for _ in range(5):
    obs = env.reset()
    policy = CamPolicy(obs) 
    for _ in range(2500):
        action = policy.get_action(obs['robot0_eef_pos'], obs['robot0_eef_quat'])
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display
        if reward == 1.0:
            success_rate += 1
            break

success_rate /= 5.0
print('success rate:', success_rate)

