import importlib.util
import inspect
import logging # Added for logging errors/warnings
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

import mujoco

import numpy as np
from pathlib import Path
from collections import OrderedDict # Import OrderedDict


DEFAULT_CAMERA_CONFIG = {
    "azimuth": 90.0,
    "distance": 3.0,
    "elevation": -25.0,
    "lookat": np.array([0., 0., 0.]),
    "fixedcamid": 0,
    "trackbodyid": -1,
    "type": 2,
}


class Go1MujocoEnv(MujocoEnv):
    """Custom Environment that follows gym interface."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, ctrl_type="torque", reward_function_path=None, **kwargs):
        # Initialize paths and MujocoEnv first
        # model_path = Path(f"./unitree_go1/scene_{ctrl_type}.xml") # Original relative path
        # Use absolute path for robustness, assuming unitree_go1 is in the same dir as the script/project root
        # TODO: Make this path handling more robust (e.g., relative to script file or project root env var)
        project_root = Path(__file__).parent.parent # Assumes envs/ is one level down from project root
        model_path = project_root / f"unitree_go1/scene_{ctrl_type}.xml"
        logging.info(f"Loading model from: {model_path.absolute().as_posix()}")

        # Define observation space structure *before* calling super().__init__ or _get_obs
        # We know the keys and shapes from _get_obs implementation
        self._clip_obs_threshold = 100.0
        # Determine num_joints based on the model (e.g., 12 for Go1)
        # We need to load the model briefly to get this, or hardcode it if stable
        # Temporary model load to get num_joints (nu)
        try:
            _temp_model = mujoco.MjModel.from_xml_path(model_path.absolute().as_posix())
            _num_joints = _temp_model.nu
            del _temp_model # Don't keep it loaded
        except Exception as e:
            logging.warning(f"Could not pre-load model to get num_joints: {e}. Assuming 12 joints.")
            _num_joints = 12

        obs_space_dict = OrderedDict([
            ("linear_velocity", spaces.Box(low=-self._clip_obs_threshold, high=self._clip_obs_threshold, shape=(3,), dtype=np.float32)),
            ("angular_velocity", spaces.Box(low=-self._clip_obs_threshold, high=self._clip_obs_threshold, shape=(3,), dtype=np.float32)),
            ("projected_gravity", spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)), # Normalized vector
            ("desired_velocity", spaces.Box(low=-self._clip_obs_threshold, high=self._clip_obs_threshold, shape=(3,), dtype=np.float32)),
            ("dofs_position", spaces.Box(low=-self._clip_obs_threshold, high=self._clip_obs_threshold, shape=(_num_joints,), dtype=np.float32)),
            ("dofs_velocity", spaces.Box(low=-self._clip_obs_threshold, high=self._clip_obs_threshold, shape=(_num_joints,), dtype=np.float32)),
            ("last_action", spaces.Box(low=-self._clip_obs_threshold, high=self._clip_obs_threshold, shape=(_num_joints,), dtype=np.float32)), # Assumes action dim == num joints
        ])
        defined_observation_space = spaces.Dict(obs_space_dict)
        logging.info(f"Defined Observation Space: {defined_observation_space}")

        # Initialize MujocoEnv with the defined observation space
        super().__init__(
            model_path=model_path.absolute().as_posix(),
            frame_skip=10,
            observation_space=defined_observation_space, # Pass the defined space
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # Store ctrl_type
        self.ctrl_type = ctrl_type

        # Metadata, timing, health setup
        self._last_render_time = -1.0
        self._max_episode_time_sec = 15.0
        self._step = 0
        self.unhealthy_steps = 0
        self.grace_period = 5 

        # Reward/Cost Weights (used if no custom reward)
        self.reward_weights = {
            "linear_vel_tracking": 2.0,
            "angular_vel_tracking": 1.0,
            "healthy": 0.0,
            "feet_airtime": 1.0,
        }
        self.cost_weights = {
            "torque": 0.0002,
            "vertical_vel": 2.0,
            "xy_angular_vel": 0.05,
            "action_rate": 0.01,
            "joint_limit": 10.0,
            "joint_velocity": 0.01,
            "joint_acceleration": 2.5e-7, 
            "orientation": 1.0,
            "collision": 1.0,
            "default_joint_position": 0.1
        }

        # Internal state variables
        self._curriculum_base = 0.3
        self._gravity_vector = np.array(self.model.opt.gravity)
        self._default_joint_position = self._get_default_joint_pos()
        self._desired_velocity_min = np.array([0.5, -0.0, -0.0])
        self._desired_velocity_max = np.array([0.5, 0.0, 0.0])
        self._desired_velocity = self._sample_desired_vel()
        self._obs_scale = {
            "linear_velocity": 2.0,
            "angular_velocity": 0.25,
            "dofs_position": 1.0,
            "dofs_velocity": 0.05,
        }
        self._tracking_velocity_sigma = 0.25
        self._healthy_z_range = (0.22, 0.65)
        self._healthy_pitch_range = (-np.deg2rad(10), np.deg2rad(10))
        self._healthy_roll_range = (-np.deg2rad(10), np.deg2rad(10))
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)

        # --- Corrected Body Indices --- #
        # Feet bodies are the calf bodies in this model
        feet_body_names = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
        self._cfrc_ext_feet_indices = [self._body_name_to_id(name) for name in feet_body_names]

        # Collision bodies: thighs and calves
        collision_body_names = [
            "FR_thigh", "FR_calf", "FL_thigh", "FL_calf",
            "RR_thigh", "RR_calf", "RL_thigh", "RL_calf"
        ]
        self._cfrc_ext_contact_indices = [self._body_name_to_id(name) for name in collision_body_names]
        # Log if any indices were not found
        if any(idx == -1 for idx in self._cfrc_ext_feet_indices):
            logging.warning(f"Could not find all feet body indices for names: {feet_body_names}")
        if any(idx == -1 for idx in self._cfrc_ext_contact_indices):
            logging.warning(f"Could not find all collision body indices for names: {collision_body_names}")
        # --- End Corrected Body Indices --- #

        self._soft_joint_range = self._calculate_soft_joint_range()
        self._reset_noise_scale = 0.1
        # Initialize last action based on the now-defined action space
        self._last_action = np.zeros(self.action_space.shape)
        # self._clip_obs_threshold = 100.0 # Defined earlier

        # Feet site names for reference (if needed)
        feet_site_names = ["FR", "FL", "RR", "RL"]
        self._feet_site_name_to_id = {
            f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site_names
        }
        self._main_body_id = self._body_name_to_id("trunk")

        # Load Custom Reward Function
        self._reward_function_path = reward_function_path
        self._custom_reward_func = self._load_custom_reward()

        # Observation space is now defined and passed to super().__init__

    def _get_default_joint_pos(self):
        # Helper to get default joint positions robustly
        if hasattr(self.model, 'key_ctrl') and len(self.model.key_ctrl) > 0 and self.model.key_ctrl[0] is not None:
            return np.array(self.model.key_ctrl[0])
        elif hasattr(self.model, 'key_qpos') and len(self.model.key_qpos) > 0 and self.model.key_qpos[0] is not None:
            default_qpos = self.model.key_qpos[0]
            num_joints = self.model.nu
            if len(default_qpos) >= 7 + num_joints:
                logging.warning(f"Using key_qpos[0, 7:] for default joint position ({num_joints} joints).")
                return np.array(default_qpos[7 : 7 + num_joints])
            else:
                logging.warning(f"key_qpos found but length insufficient ({len(default_qpos)}). Using zeros for default joints.")
                return np.zeros(num_joints)
        else:
            num_joints = self.model.nu
            logging.error(f"Neither key_ctrl nor key_qpos found/valid. Using zeros for default {num_joints} joints.")
            return np.zeros(num_joints)

    def _calculate_soft_joint_range(self):
        # Helper to calculate soft joint limits
        if not hasattr(self.model, 'actuator_ctrlrange'):
            logging.warning("actuator_ctrlrange not found in model. Cannot calculate soft joint limits.")
            return None # Or return default range?

        ctrl_range = self.model.actuator_ctrlrange
        dof_position_limit_multiplier = 0.9
        ctrl_range_offset = 0.5 * (1 - dof_position_limit_multiplier) * (ctrl_range[:, 1] - ctrl_range[:, 0])

        soft_range = np.copy(ctrl_range)
        soft_range[:, 0] += ctrl_range_offset
        soft_range[:, 1] -= ctrl_range_offset
        return soft_range

    def _body_name_to_id(self, name):
        # Helper to get body ID safely
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, name)
        if body_id == -1:
            logging.warning(f"Body '{name}' not found in model.")
        return body_id

    def _load_custom_reward(self):
        # Helper to load custom reward function
        if not self._reward_function_path:
            return None
        try:
            spec = importlib.util.spec_from_file_location("custom_reward_module", self._reward_function_path)
            if spec is None:
                raise ImportError(f"Could not load spec from {self._reward_function_path}")
            custom_reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_reward_module)
            func_name = 'custom_reward_function'
            if hasattr(custom_reward_module, func_name):
                logging.info(f"Successfully loaded custom reward function from {self._reward_function_path}")
                return getattr(custom_reward_module, func_name)
            else:
                logging.warning(f"Function '{func_name}' not found in {self._reward_function_path}. Using default reward.")
                return None
        except Exception as e:
            logging.error(f"Error loading custom reward function from {self._reward_function_path}: {e}. Using default reward.")
            return None

    def step(self, action):
        # Store observation before step for reward calculation if needed
        obs_before_step = self._get_obs()

        # Apply action and step simulation
        self.do_simulation(action, self.frame_skip)
        self._step += 1

        # Get new observation
        observation = self._get_obs()

        # Calculate reward
        reward, reward_info = self._calc_reward(action, observation)

        # Check termination conditions
        terminated = self._check_termination()
        truncated = self._step >= (self._max_episode_time_sec / self.dt)

        # Update info dictionary
        info = self._update_info(reward_info)

        # Render if needed
        if self.render_mode == "human":
            self.render()

        self._last_action = action

        # Ensure observation conforms to space (important for Dict space)
        # No explicit clipping needed here as Box spaces within Dict handle it

        return observation, reward, terminated, truncated, info

    def _check_termination(self):
        # Check health status
        if not self.is_healthy:
            self.unhealthy_steps += 1
        else:
            self.unhealthy_steps = 0
        # Terminate if unhealthy for too long
        terminated = self.unhealthy_steps >= self.grace_period
        return terminated

    def _update_info(self, reward_info):
        # Consolidate info updates
            info = {
                "x_position": self.data.qpos[0],
                "y_position": self.data.qpos[1],
                "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "unhealthy_steps": self.unhealthy_steps,
        }
            info.update(reward_info) # Add reward breakdown
            return info

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z

        min_roll, max_roll = self._healthy_roll_range
        is_healthy = is_healthy and min_roll <= state[4] <= max_roll

        min_pitch, max_pitch = self._healthy_pitch_range
        is_healthy = is_healthy and min_pitch <= state[5] <= max_pitch

        return is_healthy

    @property
    def projected_gravity(self):
        q = self.data.qpos[3:7]
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-6: # Avoid division by zero if quaternion is zero
            w, x, y, z = 1, 0, 0, 0 # Default to identity quaternion
        else:
             w, x, y, z = q / q_norm

        euler_orientation = np.array(self.euler_from_quaternion(w, x, y, z)) # Use normalized quat

        # Using Euler angles dot product approach (as was present):
        projected_gravity_not_normalized = (
            np.dot(self._gravity_vector, euler_orientation) * euler_orientation
        )

        norm_val = np.linalg.norm(projected_gravity_not_normalized)
        if norm_val == 0:
            return np.zeros_like(projected_gravity_not_normalized)
        else:
            return projected_gravity_not_normalized / norm_val

    @property
    def desired_velocity(self):
        """Returns the current target velocity vector [vx, vy, wz]."""
        return self._desired_velocity

    @property
    def feet_contact_forces(self):
        # Filter out invalid indices (-1)
        valid_indices = [idx for idx in self._cfrc_ext_feet_indices if idx != -1]
        if not valid_indices:
            # logging.warning("No valid feet body indices found. Returning zero contact forces.") # Reduce verbosity
            return np.zeros(4) # Return array of expected size
        feet_contact_forces = self.data.cfrc_ext[valid_indices]
        # Need to return a fixed size array (4), pad if necessary?
        # For now, assume if some feet are found, we return forces for those, others implicitly zero.
        # This might require adjustment based on how the reward uses this.
        # Let's return the norm for valid indices, assuming reward handles shape flexibly or sums it.
        # Returning norms for only valid feet:
        # return np.linalg.norm(feet_contact_forces, axis=1)
        # Alternative: Return fixed size (4), placing norms in correct spots
        all_forces = np.zeros(4)
        valid_norms = np.linalg.norm(feet_contact_forces, axis=1)
        valid_idx_map = {original_idx: i for i, original_idx in enumerate(valid_indices)}
        for i, original_idx in enumerate(self._cfrc_ext_feet_indices):
            if original_idx in valid_idx_map:
                all_forces[i] = valid_norms[valid_idx_map[original_idx]]
        return all_forces

    ######### Positive Reward functions #########
    @property
    def linear_velocity_tracking_reward(self):
        vel_sqr_error = np.sum(
            np.square(self._desired_velocity[:2] - self.data.qvel[:2])
        )
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    @property
    def angular_velocity_tracking_reward(self):
        vel_sqr_error = np.square(self._desired_velocity[2] - self.data.qvel[5])
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    @property
    def heading_tracking_reward(self):
        # TODO: qpos[3:7] are the quaternion values
        pass

    @property
    def feet_air_time_reward(self):
        """Award strides depending on their duration only when the feet makes contact with the ground"""
        feet_contact_force_mag = self.feet_contact_forces
        curr_contact = feet_contact_force_mag > 1.0
        contact_filter = np.logical_or(curr_contact, self._last_contacts)
        self._last_contacts = curr_contact

        # if feet_air_time is > 0 (feet was in the air) and contact_filter detects a contact with the ground
        # then it is the first contact of this stride
        first_contact = (self._feet_air_time > 0.0) * contact_filter
        self._feet_air_time += self.dt

        # Award the feets that have just finished their stride (first step with contact)
        air_time_reward = np.sum((self._feet_air_time - 1.0) * first_contact)
        # No award if the desired velocity is very low (i.e. robot should remain stationary and feet shouldn't move)
        air_time_reward *= np.linalg.norm(self._desired_velocity[:2]) > 0.1

        # zero-out the air time for the feet that have just made contact (i.e. contact_filter==1)
        self._feet_air_time *= ~contact_filter

        return air_time_reward

    @property
    def healthy_reward(self):
        return self.is_healthy

    ######### Negative Reward functions #########
    @property  # TODO: Not used
    def feet_contact_forces_cost(self):
        return np.sum(
            (self.feet_contact_forces - self._max_contact_force).clip(min=0.0)
        )

    @property
    def non_flat_base_cost(self):
        # Penalize the robot for not being flat on the ground
        return np.sum(np.square(self.projected_gravity[:2]))

    @property
    def collision_cost(self):
        # Penalize collisions on selected bodies
        valid_indices = [idx for idx in self._cfrc_ext_contact_indices if idx != -1]
        if not valid_indices:
            return 0.0
        contact_forces = self.data.cfrc_ext[valid_indices]
        collision_magnitudes = np.linalg.norm(contact_forces, axis=1)
        threshold = 0.1
        return np.sum(np.maximum(0, collision_magnitudes - threshold))

    @property
    def joint_limit_cost(self):
        # Penalize the robot for joints exceeding the soft control range
        if self._soft_joint_range is None:
            return 0.0
        joint_positions = self.data.qpos[7 : 7 + self.model.nu]
        lower_limit_violation = (self._soft_joint_range[:, 0] - joint_positions).clip(
            min=0.0
        )
        upper_limit_violation = (joint_positions - self._soft_joint_range[:, 1]).clip(min=0.0)
        return np.sum(lower_limit_violation + upper_limit_violation)

    @property
    def torque_cost(self):
        # Last 12 values are the motor torques
        if self.model.nu > 0:
            # Assuming actuator forces correspond to joint torques for torque control
            joint_torques = self.data.actuator_force[:self.model.nu]
            return np.sum(np.square(joint_torques))
        else:
            return 0.0

    @property
    def vertical_velocity_cost(self):
        return np.square(self.data.qvel[2])

    @property
    def xy_angular_velocity_cost(self):
        return np.sum(np.square(self.data.qvel[3:5]))

    def action_rate_cost(self, action):
        return np.sum(np.square(self._last_action - action))

    @property
    def joint_velocity_cost(self):
        joint_velocities = self.data.qvel[6 : 6 + self.model.nu]
        return np.sum(np.square(joint_velocities))

    @property
    def acceleration_cost(self):
        if len(self.data.qacc) >= 6 + self.model.nu:
            joint_accelerations = self.data.qacc[6 : 6 + self.model.nu]
            return np.sum(np.square(joint_accelerations))
        else:
            # logging.warning(f"qacc length ({len(self.data.qacc)}) insufficient for num joints ({self.model.nu}). Skipping accel cost.") # Reduce verbosity
            return 0.0

    @property
    def default_joint_position_cost(self):
        joint_positions = self.data.qpos[7 : 7 + self.model.nu]
        return np.sum(np.square(joint_positions - self._default_joint_position))

    @property
    def smoothness_cost(self):
        return np.sum(np.square(self.data.qpos[7:] - self._last_action))

    @property
    def curriculum_factor(self):
        return self._curriculum_base**0.997

    def _calc_reward(self, action, observation_dict_after_step):
        # The custom reward function now receives the observation *after* the action was taken.
        # observation_dict_after_step is the result from self._get_obs() called *after* self.do_simulation()

        # --- Use Custom Reward Function if available ---
        if self._custom_reward_func:
            try:
                terminated = self._check_termination() # Checks based on the state *after* the step
                # truncated = self._step >= (self._max_episode_time_sec / self.dt) # Handled by Gymnasium/SB3 wrapper
                done = terminated # Use only termination from health checks for custom reward 'done' flag? Or also time limit? Let's stick to terminated for now.

                # Pass observation_dict_after_step (current state) and done status
                reward = self._custom_reward_func(obs=observation_dict_after_step, action=action, done=done, env=self)

                # Basic validation: Check if reward is a float/int
                if not isinstance(reward, (float, int, np.float32, np.float64)):
                     logging.warning(f"Custom reward function returned non-numeric value: {reward} (type: {type(reward)}). Setting reward to 0.")
                     reward = 0.0
                # Check for NaN/inf
                elif not np.isfinite(reward):
                     logging.warning(f"Custom reward function returned non-finite value: {reward}. Setting reward to 0.")
                     reward = 0.0


                reward_info = {"custom_reward": reward}
                return reward, reward_info
            except Exception as e:
                # Log the error and re-raise it instead of falling back
                logging.error(f"Error executing custom reward function: {e}")
                # Include traceback in log for better debugging
                logging.exception("Custom reward function traceback:")
                raise # Re-raise the exception

        # --- Default Internal Reward Calculation (Fallback) ---
        # Positive Rewards
        linear_vel_tracking_reward = (
            self.linear_velocity_tracking_reward
            * self.reward_weights["linear_vel_tracking"]
        )
        angular_vel_tracking_reward = (
            self.angular_velocity_tracking_reward
            * self.reward_weights["angular_vel_tracking"]
        )
        healthy_reward = self.healthy_reward * self.reward_weights["healthy"]
        feet_air_time_reward = (
            self.feet_air_time_reward * self.reward_weights["feet_airtime"]
        )
        rewards = (
            linear_vel_tracking_reward
            + angular_vel_tracking_reward
            + healthy_reward
            + feet_air_time_reward
        )

        # Negative Costs
        ctrl_cost = self.torque_cost * self.cost_weights["torque"]
        action_rate_cost = (
            self.action_rate_cost(action) * self.cost_weights["action_rate"]
        )
        vertical_vel_cost = (
            self.vertical_velocity_cost * self.cost_weights["vertical_vel"]
        )
        xy_angular_vel_cost = (
            self.xy_angular_velocity_cost * self.cost_weights["xy_angular_vel"]
        )
        joint_limit_cost = self.joint_limit_cost * self.cost_weights["joint_limit"]
        joint_velocity_cost = (
            self.joint_velocity_cost * self.cost_weights["joint_velocity"]
        )
        joint_acceleration_cost = (
            self.acceleration_cost * self.cost_weights["joint_acceleration"]
        )
        orientation_cost = self.non_flat_base_cost * self.cost_weights["orientation"]
        collision_cost = self.collision_cost * self.cost_weights["collision"]
        default_joint_position_cost = (
            self.default_joint_position_cost
            * self.cost_weights["default_joint_position"]
        )
        costs = (
            ctrl_cost
            + action_rate_cost
            + vertical_vel_cost
            + xy_angular_vel_cost
            + joint_limit_cost
            + joint_velocity_cost
            + joint_acceleration_cost
            + orientation_cost
            + collision_cost
            + default_joint_position_cost
        )

        reward = max(0.0, rewards - costs)
        # reward = rewards - self.curriculum_factor * costs
        reward_info = {
            "linear_vel_tracking_reward": linear_vel_tracking_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def _get_obs(self) -> OrderedDict:
        # --- Gather Observation Components --- #
        # Base Velocities
        velocity = self.data.qvel.flatten()
        base_linear_velocity = velocity[:3]
        base_angular_velocity = velocity[3:6]

        # Joint State (relative to default)
        dofs_position = self.data.qpos[7 : 7 + self.model.nu].flatten() - self._default_joint_position
        dofs_velocity = velocity[6 : 6 + self.model.nu]

        # Other components
        desired_vel = self._desired_velocity
        last_action = self._last_action
        projected_gravity = self.projected_gravity
        # --- End Gather Observation Components --- #

        # --- Create Observation Dictionary --- #
        obs_dict = OrderedDict([
            ("linear_velocity", (base_linear_velocity * self._obs_scale["linear_velocity"]).astype(np.float32)),
            ("angular_velocity", (base_angular_velocity * self._obs_scale["angular_velocity"]).astype(np.float32)),
            ("projected_gravity", projected_gravity.astype(np.float32)),
            ("desired_velocity", (desired_vel * self._obs_scale["linear_velocity"]).astype(np.float32)), # Scale desired vel similarly?
            ("dofs_position", (dofs_position * self._obs_scale["dofs_position"]).astype(np.float32)),
            ("dofs_velocity", (dofs_velocity * self._obs_scale["dofs_velocity"]).astype(np.float32)),
            ("last_action", last_action.astype(np.float32)),
        ])

        # --- Clip and Ensure Finite --- #
        for key, value in obs_dict.items():
            low = self.observation_space[key].low
            high = self.observation_space[key].high
            # Clip values to their respective bounds defined in the space
            obs_dict[key] = np.clip(value, low, high)
            # Ensure finite values
            if not np.isfinite(obs_dict[key]).all():
                logging.warning(f"Non-finite values detected in observation key '{key}'. Clipping/Replacing with zeros.")
                obs_dict[key] = np.nan_to_num(obs_dict[key], nan=0.0, posinf=high, neginf=low)

        return obs_dict

    def reset_model(self):
        # ... (reset_model logic largely remains the same, but returns dict obs) ...
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.model.key_qpos[0].copy() if self.model.nq > 0 else np.array([])
        qpos_noise = self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        self.data.qpos[:] = qpos + qpos_noise

        qvel = self.model.key_qvel[0].copy() if self.model.nv > 0 else np.array([])
        self.data.qvel[:] = qvel # Reset to key_qvel, no noise usually needed here

        if self.model.nu > 0:
            ctrl = self.model.key_ctrl[0].copy() if hasattr(self.model, 'key_ctrl') and len(self.model.key_ctrl) > 0 else np.zeros(self.model.nu)
            ctrl_noise = self._reset_noise_scale * self.np_random.standard_normal(self.model.nu)
            self.data.ctrl[:] = ctrl + ctrl_noise

        mujoco.mj_forward(self.model, self.data)

        self._desired_velocity = self._sample_desired_vel()
        self._step = 0
        self._last_action = np.zeros(self.action_space.shape)
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._last_render_time = -1.0
        self.unhealthy_steps = 0

        # Get the observation dictionary
        observation = self._get_obs()
        return observation

    # ... (_get_reset_info, _sample_desired_vel, euler_from_quaternion remain same) ...

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "reset_desired_velocity": self._desired_velocity.tolist()
        }

    def _sample_desired_vel(self):
        return self.np_random.uniform(
            low=self._desired_velocity_min, high=self._desired_velocity_max
        )

    @staticmethod
    def euler_from_quaternion(w, x, y, z):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
        return roll_x, pitch_y, yaw_z

# --- Factory Function --- #
def make_custom_go1(reward_function_path, ctrl_type="torque", render_mode=None, **kwargs):
    """Creates the Go1MujocoEnv with a custom reward function path."""
    logging.info(f"Creating Go1 environment (ctrl: {ctrl_type}, render: {render_mode}) with reward path: {reward_function_path}")
    try:
        # Pass render_mode explicitly to the constructor
        env = Go1MujocoEnv(reward_function_path=reward_function_path, ctrl_type=ctrl_type, render_mode=render_mode, **kwargs)
        # No reset needed here as it's handled by SB3 or Monitor wrapper
        return env
    except Exception as e:
        logging.error(f"Failed to create Go1 environment: {e}")
        logging.exception("Exception details:") # Log full traceback
        raise
# --- End Factory Function --- #