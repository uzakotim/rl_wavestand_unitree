# import numpy as np
import math
import mujoco
from mujoco import MjModel, MjData, mj_step, mj_forward, mj_name2id, mj_resetDataKeyframe
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco.viewer


class UnitreeWaveEnv(gym.Env):
    """
    MuJoCo-based env for Unitree G1: stand + wave.
    """

    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, model_path="g1/scene_29dof.xml",
                 control_joints=None, dt=0.02, render_mode="none"):

        super().__init__()
        self.render_mode = render_mode

        # -------- Load MuJoCo Model --------
        self.model = MjModel.from_xml_path(model_path)
        self.data = MjData(self.model)

        self.dt = dt
        self.model.opt.timestep = dt
        self.viewer = None

        # -------- Control Joints --------
        if control_joints is None:
            control_joints = [
                # Legs (L)
                'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
                'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',

                # Legs (R)
                'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
                'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',

                # Waist
                'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',

                # Left arm
                'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
                'left_shoulder_yaw_joint', 'left_elbow_joint',
                'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',

                # Right arm
                'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
                'right_shoulder_yaw_joint', 'right_elbow_joint',
                'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
            ]

        self.control_joints = control_joints
        self.joint_indices = [
            mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in control_joints
        ]
        assert all(idx != -1 for idx in self.joint_indices), "Missing joint(s)"

        # -------- DOF to Actuator map --------
        self.dof_to_actuator = {
            self.model.actuator_trnid[a][1]: a
            for a in range(self.model.nu)
        }

        # -------- Action Space --------
        self.n_actions = len(self.control_joints)
        max_delta = 1.0  # 1.0
        self.action_space = spaces.Box(
            low=-max_delta, high=max_delta,
            shape=(self.n_actions,), dtype=np.float32
        )

        # -------- Observation Space --------
        obs_dim = 3 + 3 + self.n_actions + self.n_actions + 2 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # -------- Initial Pose --------
        self.qpos0 = self.model.qpos0.copy()
        self.stand_height = 0.8

        # -------- Reward Parameters --------
        self.w_stand = 1.0
        self.w_posture = 0.8
        self.w_wave = 0.0
        self.w_torque_pen = 1e-4

        # -------- Wave Parameters --------
        self.wave_freq = 0.8
        self.wave_amp = 0.6
        self.time = 0.0

        self.max_episode_steps = 1000
        self.step_count = 0

    # =========================================================
    #                     GYMNASIUM API
    # =========================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mj_resetDataKeyframe(self.model, self.data, 0)

        self.data.qpos[:] = self.qpos0 + 0.001 * \
            np.random.randn(*self.qpos0.shape)
        self.data.qvel[:] = 0

        mj_forward(self.model, self.data)

        self.time = 0.0
        self.step_count = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.time += self.dt
        self.step_count += 1

        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        kp = 20.0
        kd = 2.0
        tau = np.zeros(self.model.nu)

        # These lists should match the number of joints
        upper_limits = [2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618, 2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618, 2.618, 0.52, 0.52, 2.6704,
                        2.2515, 2.618, 2.0944, 1.972222054, 1.614429558, 1.614429558, 2.6704, 1.5882, 2.618, 2.0944, 1.972222054, 1.614429558, 1.614429558]
        lower_limits = [-2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618, -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618, -2.618, -0.52, -0.52, -
                        3.0892, -1.5882, -2.618, -1.0472, -1.972222054, -1.614429558, -1.614429558, -3.0892, -2.2515, -2.618, -1.0472, -1.972222054, -1.614429558, -1.614429558]

        # ------------ PD control ------------
        for i, jid in enumerate(self.joint_indices):
            qpos_addr = self.model.jnt_qposadr[jid]
            dof_addr = self.model.jnt_dofadr[jid]

            target = self.qpos0[qpos_addr] + action[i] * \
                (upper_limits[i]-lower_limits[i])/2
            qpos_des = target
            qvel_des = 0.0

            dq = qvel_des - qvel[dof_addr]
            tau_val = kp*(qpos_des - qpos[qpos_addr]) + kd*dq

            if dof_addr in self.dof_to_actuator:
                act = self.dof_to_actuator[dof_addr]
                tau[act] = tau_val

        self.data.ctrl[:] = tau
        mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, info = self._compute_reward(obs, action, tau)

        terminated = self._is_done(obs)
        truncated = self.step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    # =========================================================
    #                    Observations / Reward
    # =========================================================
    def _get_obs(self):
        base_quat_xyzw = self.data.qpos[3:7]
        quat = np.array([base_quat_xyzw[3], *base_quat_xyzw[:3]])  # wxyz
        euler = self._quat_to_euler(quat)

        base_ang_vel = self.data.qvel[3:6]

        joint_angles = np.array([
            self.data.qpos[self.model.jnt_qposadr[j]]
            for j in self.joint_indices
        ])
        joint_vels = np.array([
            self.data.qvel[self.model.jnt_dofadr[j]]
            for j in self.joint_indices
        ])

        foot_contacts = np.array([0.0, 0.0])
        phase = np.array([self.time % (1.0/self.wave_freq)])

        return np.concatenate([
            euler, base_ang_vel,
            joint_angles, joint_vels,
            foot_contacts, phase
        ]).astype(np.float32)

    def _compute_reward(self, obs, action, tau):
        roll, pitch, yaw = obs[0:3]
        upright = math.exp(-(abs(roll) + abs(pitch)))

        joint_angles = obs[6:6+self.n_actions]
        qref = np.array([self.model.qpos0[self.model.jnt_qposadr[j]]
                         for j in self.joint_indices])
        posture_err = np.linalg.norm(joint_angles - qref)
        r_posture = math.exp(-5 * posture_err)

        r_wave = 0.0

        torque_pen = self.w_torque_pen * np.sum(tau**2)

        reward = (
            self.w_stand * upright +
            self.w_posture * r_posture +
            self.w_wave * r_wave -
            torque_pen
        )
        reward = np.clip(reward, -10, 10)
        return reward, {
            "upright": upright,
            "posture": r_posture,
            "wave": r_wave,
            "torque_pen": torque_pen
        }

    def _is_done(self, obs):
        roll, pitch = obs[0:2]
        # print("cos roll,  cos pitch", math.cos(roll), math.cos(pitch))
        # print("base height", self._get_base_height())

        roll, pitch = obs[0], obs[1]

        # upright check: cos(roll) ~ 1 when standing
        if abs(math.cos(roll)) < 0.5 or abs(math.cos(pitch)) < 0.5:
            return True

        if self._get_base_height() < 0.3:  # allow some tolerance
            return True

        return False
    # =========================================================
    #                       Utilities
    # =========================================================

    def _get_base_height(self):
        base_body_name = "pelvis"
        base_body_id = mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        if base_body_id != -1:
            return self.data.xpos[base_body_id, 2]
        return 0.0

    def _quat_to_euler(self, q):
        w, x, y, z = q
        t0 = 2*(w*x + y*z)
        t1 = 1 - 2*(x*x + y*y)
        roll = math.atan2(t0, t1)

        t2 = 2*(w*y - z*x)
        t2 = np.clip(t2, -1, 1)
        pitch = math.asin(t2)

        t3 = 2*(w*z + x*y)
        t4 = 1 - 2*(y*y + z*z)
        yaw = math.atan2(t3, t4)

        return np.array([roll, pitch, yaw])

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
