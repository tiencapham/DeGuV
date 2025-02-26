"""
Play random actions in an environment and render a video that demonstrates segmentation.
"""
import argparse
import colorsys
import json
import random

import imageio
import matplotlib.cm as cm
import numpy as np
from PIL import Image

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix
from robosuite.utils.input_utils import input2action
def randomize_colors(N, bright=True):
    """
    Modified from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py#L59
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.5
    hsv = [(1.0 * i / N, 1, brightness) for i in range(N)]
    colors = np.array(list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv)))
    rstate = np.random.RandomState(seed=20)
    np.random.shuffle(colors)
    return colors


def segmentation_to_rgb(seg_im, random_colors=False):
    """
    Helper function to visualize segmentations as RGB frames.
    NOTE: assumes that geom IDs go up to 255 at most - if not,
    multiple geoms might be assigned to the same color.
    """
    # ensure all values lie within [0, 255]
    seg_im = np.mod(seg_im, 256)

    if random_colors:
        colors = randomize_colors(N=256, bright=True)
        return (255.0 * colors[seg_im]).astype(np.uint8)
    else:
        # deterministic shuffling of values to map each geom ID to a random int in [0, 255]
        rstate = np.random.RandomState(seed=8)
        inds = np.arange(256)
        rstate.shuffle(inds)

        # use @inds to map each geom ID to a color
        return (255.0 * cm.rainbow(inds[seg_im], 3)).astype(np.uint8)[..., :3]

def rescale(array):
    return (array-array.min())/(array.max()-array.min())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, default="video.mp4", help="Path to video file")
    parser.add_argument("--random-colors", action="store_true", help="Radnomize segmentation colors")
    parser.add_argument("--segmentation-level", type=str, default="element", help="instance, class, or element")
    
    args = parser.parse_args()

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # Choose environment and add it to options
    options["env_name"] = 'Lift'   #"TwoArmHandover"
    options["robots"] = 'Panda' # ["Panda", "Panda"]

    # Choose controller
    controller_name = "OSC_POSE"

    # Choose camera
    camera = "frontview"

    # Choose segmentation type
    segmentation_level = args.segmentation_level  # Options are {instance, class, element}

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        camera_names=camera,
        camera_depths = True,
        camera_segmentations=segmentation_level,
        camera_heights=512,
        camera_widths=512,
    )
    
    env.reset()
    # print(get_camera_intrinsic_matrix(env.sim, camera_name='frontview',camera_height=512,
    #     camera_width=512,))
    video_writer = imageio.get_writer('video/segmentation.mp4', fps=20)
    pix_video_writer = imageio.get_writer('video/pixel.mp4', fps=20)
    depth_video_writer = imageio.get_writer('video/depth.mp4', fps=20)
    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(100):
        
        action = 0.5 * np.random.uniform(low, high)
        action, grasp = input2action(device = '')
        # for j in range(50):
        #     obs, reward, done, _ = env.step(action)
        obs, reward, done, _ = env.step(action)
    #     video_img = obs[f"{camera}_segmentation_{segmentation_level}"].squeeze(-1)[::-1]
    #     pixel_frame = obs[camera + "_image"]
    #     depth_frame = obs[camera + "_depth"]
    #     depth_frame = rescale(depth_frame)
    #     # print(f'{depth_frame.shape}')
    #     depth_data = np.resize(depth_frame, (512,512))
    #     np.savetxt("frame/depth_{}.txt".format(i), depth_data, fmt="%.2f")
    #     np.savetxt("frame/seg_{}.txt".format(i), video_img, fmt="%.2f")
    #     pixel_frame = Image.fromarray(pixel_frame)
    #     pixel_frame.save("frame/image_{}.png".format(i))
    #     video_img = segmentation_to_rgb(video_img, args.random_colors)
    #     video_writer.append_data(video_img)
    #     # pix_video_writer.append_data(pixel_frame)
    #     # depth_video_writer.append_data(depth_frame)

    #     image = Image.fromarray(video_img)
    #     image.save("frame/seg_{}.png".format(i))
    #     if i % 5 == 0:
    #         print("Step #{} / 100".format(i))
    # env.close()
    # video_writer.close()
    # pix_video_writer.close()
    # depth_video_writer.close()
    
    # print("Video saved to {}".format(args.video_path))
