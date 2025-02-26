# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import imageio
import numpy as np
import random

class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.store = 0
    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)
    
    def init_dmc(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record_dmc(env, video=True)
    
    def record(self, env, att = None):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=0)
            else:
                frame = env.render()

            if att is not None:
                # print(f'size of att: {att.shape}')
                if att.shape[0]==3:
                    overlay = att.reshape(84,84,3)
                    frame_new = frame*overlay
                    # frame_new[overlay < 1] = random.uniform(frame.flatten().min(), frame.flatten().max())
                    # print(f'size of frame: {frame.shape}')  
                # overlay = cv2.resize(np.array(original_att), (84,84), cv2.INTER_NEAREST)
                else:
                    overlay = np.stack([att] * 3, axis=-1)
                # print(f'size of att: {att.shape}, overlay: {overlay.shape}')
                    frame_new = frame*overlay
                frame = cv2.resize(frame, (480,480), cv2.INTER_NEAREST)
                frame_new = cv2.resize(frame_new, (480,480), cv2.INTER_NEAREST)
                frame = np.hstack((frame, frame_new))
            # frame = cv2.resize(frame, (480,480), cv2.INTER_NEAREST)
            else:
                frame = cv2.resize(frame, (480,480), cv2.INTER_NEAREST)
                no_frame = np.zeros_like(frame)
                frame = np.hstack((frame, no_frame))
            self.frames.append(frame.astype(np.uint8))


    def record_dmc(self, env, video=False):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=0)
            else:
                if video:
                    frame = env._env._env._env._gym_env.env.env.render()
                else:
                    frame = env.render()
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)


class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                               dsize=(self.render_size, self.render_size),
                               interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
