import os
import numpy as np
import torch
import tqdm
import time
import imageio
from bridge_policy_3d.env import AdroitEnv
from bridge_policy_3d.gym_util.mjpc_diffusion_wrapper import MujocoPointcloudWrapperAdroit
from bridge_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from bridge_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from bridge_policy_3d.policy.base_policy import BasePolicy
from bridge_policy_3d.common.pytorch_util import dict_apply
from bridge_policy_3d.env_runner.base_runner import BaseRunner
import bridge_policy_3d.common.logger_util as logger_util
from termcolor import cprint


class AdroitRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_point_crop=True,
                 sde=None
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MujocoPointcloudWrapperAdroit(env=AdroitEnv(env_name=task_name, use_point_cloud=True),
                                                  env_name='adroit_'+task_name, use_point_crop=use_point_crop)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn()
        # breakpoint()
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

        self.sde = sde
        

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_goal_achieved = []
        all_success_rates = []
        
        # breakpoint()

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
                
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            num_goal_achieved = 0
            actual_step_count = 0
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input, sde=self.sde)
                    

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)
                # step env
                obs, reward, done, info = env.step(action)
                # all_goal_achieved.append(info['goal_achieved']
                num_goal_achieved += np.sum(info['goal_achieved'])
                done = np.all(done)
                actual_step_count += 1

            all_success_rates.append(info['goal_achieved'])
            all_goal_achieved.append(num_goal_achieved)


        # log
        log_data = dict()
        

        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        # 保存视频到本地
        videos = env.env.get_video()
        # if len(videos.shape) == 5:
        #     videos = videos[:, 0]  # 选择第一个视角

        # breakpoint()
        # 确保保存目录存在
        save_dir = os.path.join(self.output_dir or ".", "videos")
        os.makedirs(save_dir, exist_ok=True)
        # 保存为mp4
        save_path = os.path.join(save_dir, f"sim_video_eval_{self.task_name}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        # videos: (batch, T, H, W, C) or (T, H, W, C)
        # 只保存第一个batch
        if len(videos.shape) == 5:
            video_to_save = videos[0]
        else:
            video_to_save = videos.transpose(0, 2, 3, 1)
            
        # breakpoint()
        # 转换为uint8
        if video_to_save.dtype != np.uint8:
            video_to_save = (np.clip(video_to_save, 0, 1) * 255).astype(np.uint8)
        # imageio保存视频
        imageio.mimsave(save_path, video_to_save, fps=self.fps, macro_block_size=None)
        log_data[f'sim_video_eval_path'] = save_path

        # 清空视频缓存
        _ = env.reset()
        # 清理内存
        videos = None
        del env

        return log_data
