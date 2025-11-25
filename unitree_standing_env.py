# file: unitree_wave_env.py
import numpy as np
import gym
from gym import spaces
import math

# Replace with your mujoco import depending on binding:
# e.g. import mujoco_py as mjpy or import mujoco
import mujoco
from mujoco import MjModel, MjData, mj_step, mj_forward, mj_name2id, mj_resetDataKeyframe


class UnitreeWaveEnv(gym.Env):
    """
    MuJoCo-based env for Unitree G1: stand + wave.
    Action: target joint position offsets for arm + stabilizing joints
    Observation: base orientation, joint angles, velocities, contact sensors, phase
    """

    def __init__(self, model_path="g1/g1_29dof.xml", control_joints=None, dt=0.02):
        super().__init__()
        # Load the model using the new API method
        self.model = MjModel.from_xml_path(model_path)
        assert self.model is not None, "Failed to load MuJoCo model."

        self.data = MjData(self.model)

        self.dt = dt
        # Set the simulation timestep in the model
        self.model.opt.timestep = dt
        self.viewer = None

        # Decide which joints to control.
        if control_joints is None:
            control_joints = [
                # Legs (L)
                'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
                'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
                # Legs (R)
                'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
                'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
                # Waist
                'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']

        self.control_joints = control_joints
        # Use mj_name2id helper function
        self.joint_indices = [
            mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in control_joints]

        # Ensure all joints were found
        assert all(
            idx != -1 for idx in self.joint_indices), "One or more control joints not found in model."

        self.n_actions = len(self.control_joints)

        # Action space: delta joint positions (radians) clipped
        max_delta = 0.6  # ±0.6 rad offsets
        self.action_space = spaces.Box(
            low=-max_delta, high=max_delta, shape=(self.n_actions,), dtype=np.float32)

        # Observation space: concat of base orientation (quat or euler), base ang vel, joint angles, joint velocities, foot contacts, phase
        # Note: self.model.nq includes the 7 DoF for the free base joint [pos(3), quat(4)]
        n_joints_qpos = self.model.nq
        # number of velocities (nv usually equals nq - 1 if no free joints)
        n_joints_qvel = self.model.nv

        # Updated observation dim definition
        # base euler, base ang vel, controlled joint angles, controlled joint velocities, foot contacts, phase
        obs_dim = 3 + 3 + self.n_actions + self.n_actions + 2 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Standing reference pose (read from model initial qpos from MjModel)
        self.qpos0 = self.model.qpos0.copy()  # Use model.qpos0 instead of sim.data.qpos
        # Need to implement _get_base_height properly using the new API's physics state data
        self.stand_height = 0.8  # Placeholder until _get_base_height is implemented correctly

        # Reward weights
        self.w_stand = 1.0
        self.w_posture = 0.8
        self.w_wave = 0.0
        self.w_torque_pen = 1e-4

        # Wave parameters
        self.wave_freq = 0.8  # Hz
        self.wave_amp = 0.6  # rad amplitude at shoulder
        self.time = 0.0
        self.max_episode_steps = 1000
        self.step_count = 0

    def reset(self):
        # Reset sim data to the initial keyframe configuration (qpos0, etc.)
        # The new API provides mj_resetDataKeyframe
        mj_resetDataKeyframe(self.model, self.data, 0)

        # Apply small perturbations
        self.data.qpos[:] = self.qpos0[:] + 0.001 * \
            np.random.randn(*self.qpos0.shape)
        self.data.qvel[:] = 0

        # Forward pass to update physics state
        mj_forward(self.model, self.data)

        self.time = 0.0
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.time += self.dt
        self.step_count += 1

        # compute desired joint positions = stand_ref + action for control joints
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # Simple PD controller:
        kp = 150.0
        kd = 2.0 * np.sqrt(kp)
        # Actuator force vector (nu = number of actuators)
        tau = np.zeros(self.model.nu)

        # Build desired positions and compute torques
        desired_qpos = qpos.copy()
        for i, jid in enumerate(self.joint_indices):
            # Use model attributes to find position and velocity addresses
            idx_qpos = self.model.jnt_qposadr[jid]
            # Index in velocity/actuator space
            idx_dof = self.model.jnt_dofadr[jid]

            # target = reference + delta
            target = self.qpos0[idx_qpos] + action[i]
            desired_qpos[idx_qpos] = target

            # desired velocity = 0
            # Note: qvel indices align with dof indices
            dq = 0.0 - qvel[idx_dof]

            # PD torque calculation
            tau_val = kp * (target - qpos[idx_qpos]) + kd * dq

            # Map torque to actuator indices
            # ASSUMPTION: This code assumes there is a direct 1:1 mapping between
            # the controlled joint's DOF index (idx_dof) and an actuator index.
            # You must ensure your model XML defines actuators correctly for these joints.
            if idx_dof < len(tau):
                tau[idx_dof] = tau_val
            # If you have specific actuator names, you should map to self.model.actuator_name2id
            # Example: act_id = mj_name2id(self.model, mjOBJ_ACTUATOR, "actuator_name")
            # tau[act_id] = tau_val

        # Apply torques/control signals (for new mujoco api)
        self.data.ctrl[:] = tau

        # Step the simulation forward
        mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, info = self._compute_reward(obs, action, tau)
        done = self._is_done(obs)
        if self.step_count >= self.max_episode_steps:
            done = True

        return obs, reward, done, info

    def _get_obs(self):
        # base quat: in new API, it's model.qpos[3:7] (wxyz order internally in MjData but accessed differently)
        # MuJoCo internal order for quat is [x,y,z,w] when stored in data.qpos
        base_quat_xyzw = self.data.qpos[3:7]

        # convert to euler (roll,pitch,yaw) using mujoco's utility function
        # mju.quat2euler takes a 4-element numpy array [w, x, y, z] -> returns [r, p, y]
        quat_wxyz = np.array(
            [base_quat_xyzw[3], base_quat_xyzw[0], base_quat_xyzw[1], base_quat_xyzw[2]])
        euler = np.zeros(3)
        euler = self._quat_to_euler(quat_wxyz)

        # base angular velocity: MjData.qvel[3:6]
        base_ang_vel = self.data.qvel[3:6].copy()

        # controlled joint angles:
        joint_angles = np.array(
            [self.data.qpos[self.model.jnt_qposadr[j]] for j in self.joint_indices])

        # controlled joint velocities: use dofadr
        joint_vels = np.array(
            [self.data.qvel[self.model.jnt_dofadr[j]] for j in self.joint_indices])

        # contacts: you need to define geom names for feet in your XML and retrieve forces/contacts here.
        # This part requires model-specific implementation (e.g., checking mj_contact array)
        foot_contacts = np.array([0.0, 0.0])  # Placeholder

        # phase/time
        phase = np.array([self.time % (1.0/self.wave_freq)])

        obs = np.concatenate(
            [euler, base_ang_vel, joint_angles, joint_vels, foot_contacts, phase])
        return obs.astype(np.float32)

    def _compute_reward(self, obs, action, tau):
        # Extract helpful quantities
        roll, pitch, yaw = obs[0], obs[1], obs[2]
        upright_reward = math.exp(-(abs(roll) + abs(pitch)))

        # posture: closeness of controlled joints to stand reference
        joint_angles = obs[6:6+self.n_actions]
        # Reference positions are from the initial model configuration
        qref = np.array([self.model.qpos0[self.model.jnt_qposadr[j]]
                        for j in self.joint_indices])
        posture_err = np.linalg.norm(joint_angles - qref)
        r_posture = math.exp(-5.0 * posture_err)

        # wave reward: encourage shoulder joints to follow a sine wave
        shoulder_idx = None
        for i, name in enumerate(self.control_joints):
            # example detection
            if 'shoulder' in name and ('r' in name or 'R' in name):
                shoulder_idx = i
                break

        if shoulder_idx is not None:
            # Use model.qpos0 for reference
            ref_qpos_idx = self.model.jnt_qposadr[self.joint_indices[shoulder_idx]]
            desired = self.model.qpos0[ref_qpos_idx] + \
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
            # The viewer is typically handled outside the Env for the new API
            # but if you want to integrate it here for simplicity:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        if self.viewer:
            self.viewer.sync()

    def _get_base_height(self):
        # The base body (usually "torso" or similar) position z-coordinate in world frame
        # You'll need the body ID of your robot's main base link
        base_body_name = "base_link"  # Replace with actual base link name from XML
        base_body_id = mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        if base_body_id != -1:
            # The xpos array stores the Cartesian position of each body
            # Index 2 is the Z coordinate
            return self.data.xpos[base_body_id, 2]
        return 0.0  # Default fallback

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

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
