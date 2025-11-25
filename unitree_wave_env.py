# file: unitree_wave_env.py
import numpy as np
import gym
from gym import spaces
import math

# Replace with your mujoco import depending on binding:
# e.g. import mujoco_py as mjpy or import mujoco
import mujoco  # if using mujoco-py
from mujoco import load_model_from_path, MjSim, MjViewer


class UnitreeWaveEnv(gym.Env):
    """
    MuJoCo-based env for Unitree G1: stand + wave.
    Action: target joint position offsets for arm + stabilizing joints
    Observation: base orientation, joint angles, velocities, contact sensors, phase
    """

    def __init__(self, model_path="/home/timur/git/rl_wavehand_unitree/g1_29dof.xml", control_joints=None, dt=0.02):
        super().__init__()
        self.model = load_model_from_path(model_path)
        # Confirm model loaded correctly
        assert self.model is not None, "Failed to load MuJoCo model."

        self.sim = MjSim(self.model)
        self.dt = dt
        self.viewer = None

        # Decide which joints to control. Example:
        # control_joints = ['hip_pitch_r', 'knee_r', ..., 'shoulder_pitch_r', 'shoulder_roll_r']
        if control_joints is None:
            control_joints = [
                # Legs (L)
                'L_LEG_HIP_PITCH', 'L_LEG_HIP_ROLL', 'L_LEG_HIP_YAW',
                'L_LEG_KNEE', 'L_LEG_ANKLE_PITCH', 'L_LEG_ANKLE_ROLL',
                # Legs (R)
                'R_LEG_HIP_PITCH', 'R_LEG_HIP_ROLL', 'R_LEG_HIP_YAW',
                'R_LEG_KNEE', 'R_LEG_ANKLE_PITCH', 'R_LEG_ANKLE_ROLL',
                # Waist
                'WAIST_YAW', 'WAIST_ROLL', 'WAIST_PITCH',
                # Waving arm (R)
                'R_SHOULDER_PITCH', 'R_SHOULDER_ROLL', 'R_ELBOW'
            ]

        self.control_joints = control_joints
        self.joint_indices = [
            self.model.joint_name2id(j) for j in control_joints]
        self.n_actions = len(self.control_joints)

        # Action space: delta joint positions (radians) clipped
        max_delta = 0.6  # ±0.6 rad offsets
        self.action_space = spaces.Box(
            low=-max_delta, high=max_delta, shape=(self.n_actions,), dtype=np.float32)

        # Observation space: concat of base orientation (quat or euler), base ang vel, joint angles, joint velocities, foot contacts, phase
        n_joints = self.model.nq
        obs_dim = 3 + 3 + self.n_actions + self.n_actions + 2 + 1  # example dims
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Standing reference pose (read from model initial qpos)
        self.qpos0 = self.sim.data.qpos.copy()
        self.stand_height = self._get_base_height()

        # Reward weights
        self.w_stand = 1.0
        self.w_posture = 0.8
        self.w_wave = 1.2
        self.w_torque_pen = 1e-4

        # Wave parameters
        self.wave_freq = 0.8  # Hz
        self.wave_amp = 0.6  # rad amplitude at shoulder
        self.time = 0.0
        self.max_episode_steps = 1000
        self.step_count = 0

    def reset(self):
        # Reset sim to initial pose with small perturbations
        self.sim.set_state(self.sim.get_state())  # or set qpos to qpos0
        self.sim.data.qpos[:] = self.qpos0[:] + \
            0.001 * np.random.randn(*self.qpos0.shape)
        self.sim.data.qvel[:] = 0
        self.sim.forward()
        self.time = 0.0
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.time += self.dt
        self.step_count += 1

        # compute desired joint positions = stand_ref + action for control joints
        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()

        # Simple PD controller: for every controlled joint, compute torque = kp*(q_target - q) - kd*dq
        kp = 150.0
        kd = 2.0 * np.sqrt(kp)
        tau = np.zeros(self.model.nu)

        # Build desired positions
        desired_qpos = qpos.copy()
        for i, jid in enumerate(self.joint_indices):
            # in MuJoCo, joint index mapping depends on model; here assume positional mapping aligns
            # may need adaptation for your mujoco binding
            idx_qpos = self.model.jnt_qposadr[jid]
            # target = reference + delta
            target = self.qpos0[idx_qpos] + action[i]
            desired_qpos[idx_qpos] = target

            # desired velocity = 0
            dq = 0.0 - qvel[idx_qpos]
            # PD torque:
            tau_val = kp * (target - qpos[idx_qpos]) + kd * dq
            # Map torque to actuator indices - simple assumption: actuator index == joint index
            # You may need to map joint id -> actuator id
            if idx_qpos < len(tau):
                tau[idx_qpos] = tau_val

        # Apply torques (for mujoco-py)
        self.sim.data.ctrl[:] = tau
        self.sim.step()

        obs = self._get_obs()
        reward, info = self._compute_reward(obs, action, tau)
        done = self._is_done(obs)
        if self.step_count >= self.max_episode_steps:
            done = True

        return obs, reward, done, info

    def _get_obs(self):
        # base euler:
        # depending on model ordering: [x,y,z,w]
        base_quat = self.sim.data.qpos[3:7]
        # convert to euler (roll,pitch,yaw)
        # naive conversion: replace with proper quaternion->euler
        qw, qx, qy, qz = base_quat[3], base_quat[0], base_quat[1], base_quat[2]
        # compute small-angle approx or use a function/library
        # For brevity, include placeholder
        euler = self._quat_to_euler([qw, qx, qy, qz])
        base_ang_vel = self.sim.data.qvel[3:6].copy()
        joint_angles = np.array(
            [self.sim.data.qpos[self.model.jnt_qposadr[j]] for j in self.joint_indices])
        joint_vels = np.array(
            [self.sim.data.qvel[self.model.jnt_dofadr[j]] for j in self.joint_indices])
        # contacts: choose two feet geoms indices (you must set names in your model)
        foot_contacts = np.array([0.0, 0.0])
        # phase/time
        phase = np.array([self.time % (1.0/self.wave_freq)])
        obs = np.concatenate(
            [euler, base_ang_vel, joint_angles, joint_vels, foot_contacts, phase])
        return obs.astype(np.float32)

    def _compute_reward(self, obs, action, tau):
        # Extract helpful quantities
        # r_stand: torso alignment -> use body z-axis dot world z
        # For now estimate via base pitch/roll from obs
        roll, pitch, yaw = obs[0], obs[1], obs[2]
        # 1 when upright, decays with tilt
        upright_reward = math.exp(-(abs(roll) + abs(pitch)))

        # posture: closeness of controlled joints to stand reference
        joint_angles = obs[6:6+self.n_actions]
        qref = np.array([self.qpos0[self.model.jnt_qposadr[j]]
                        for j in self.joint_indices])
        posture_err = np.linalg.norm(joint_angles - qref)
        r_posture = math.exp(-5.0 * posture_err)

        # wave reward: encourage shoulder joints to follow a sine wave
        # assume the shoulder is index = last two in control_joints (adjust to your mapping)
        shoulder_idx = None
        for i, name in enumerate(self.control_joints):
            if 'shoulder' in name and 'r' in name:  # example detection
                shoulder_idx = i
                break
        # desired shoulder angle:
        if shoulder_idx is not None:
            desired = self.qpos0[self.model.jnt_qposadr[self.joint_indices[shoulder_idx]]] + \
                self.wave_amp * \
                math.sin(2 * math.pi * self.wave_freq * self.time)
            err = abs(joint_angles[shoulder_idx] - desired)
            r_wave = math.exp(-10.0 * err)
        else:
            r_wave = 0.0

        # torque penalty
        torque_pen = self.w_torque_pen * np.sum(np.square(tau))

        reward = self.w_stand * upright_reward + self.w_posture * \
            r_posture + self.w_wave * r_wave - torque_pen
        info = {'upright': upright_reward, 'posture': r_posture,
                'wave': r_wave, 'torque_pen': torque_pen}
        return reward, info

    def _is_done(self, obs):
        # done if tilt too large or base height below threshold
        roll, pitch = obs[0], obs[1]
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            return True
        # or low height
        base_height = self._get_base_height()
        if base_height < 0.2:  # example threshold
            return True
        return False

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
        self.viewer.render()

    def _get_base_height(self):
        # base z -> depending on qpos ordering; often qpos[2] is base z
        return float(self.sim.data.qpos[2])

    def _quat_to_euler(self, q):
        # q = [w,x,y,z]
        w, x, y, z = q
        # convert to roll, pitch, yaw
        # apply robust conversion
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return np.array([roll, pitch, yaw])
