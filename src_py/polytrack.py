"""A Gymnasium environment for PolyTrack."""

# pylint: disable=no-name-in-module

import functools
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from simulation_worker import SimulationWorkerPy, PlayerControllerPy, CarStatePy
from ray.rllib.utils.typing import EnvConfigDict

from src_py.reward import RewardSystem

# pylint: disable=pointless-string-statement
"""
* speed
* is_next_checkpoint_finishline
* 3 for yaw, pitch, roll
* 4 for collision impulses
* 4 * wheel contacts (wheel contact is a position (x, y, z) and a normal (x, y, z))
* 4 * wheel suspension lengths
* 4 * wheel suspension velocties
* 4 * wheel delta rotations
* 4 * wheel skid infos
* steering value
* lastFrame up, right, down, left
* 50 raycasts uniform for the front half sphere
    (basically what is infront and to the sides of the car)
* 15 raycasts uniform for the back half sphere
    (basically what is behind but excluding directly to the sides of the car)
* relative checkpoint position
"""
NUM_INPUTS = 1 + 1 + 3 + 4 + (4 * 6) + 4 + 4 + 4 + 4 + 1 + 4 + 50 + 15 + 3
NUM_RAYCASTS_FRONT = 50
NUM_RAYCASTS_BACK = 15
RAYCAST_MAX_DISTANCE = 10.0
MAX_FRAMES = 5000

DEFAULT_REWARD_SYSTEM = RewardSystem()


class PolyTrackEnv(gym.Env):
    """Gymnasium environment wrapping the PolyTrack simulation."""

    _ALL_DIRS = None
    _sim_worker: SimulationWorkerPy
    _last_car_id: int = 0

    def __init__(self, config: EnvConfigDict | None = None):
        """Initialise the environment, simulation worker, and viewer process."""
        assert config is not None, "Config cannot be None."
        export_string = config.get("export_string", "")
        assert (
            export_string or PolyTrackEnv._sim_worker is not None
        ), "Export string cannot be empty."

        self.current_car = 0
        self._prev_data: CarStatePy | None = None
        self._reward_system: RewardSystem = config.get(
            "reward_system", DEFAULT_REWARD_SYSTEM
        )

        if PolyTrackEnv._sim_worker is None:
            PolyTrackEnv._sim_worker = SimulationWorkerPy(export_string)

        PolyTrackEnv._sim_worker.create_car(PolyTrackEnv._last_car_id)
        self.car_id = PolyTrackEnv._last_car_id
        PolyTrackEnv._last_car_id += 1

        if PolyTrackEnv._ALL_DIRS is None:
            PolyTrackEnv._ALL_DIRS = np.array(
                self._fibonacci_hemisphere(NUM_RAYCASTS_FRONT, True)
                + self._fibonacci_hemisphere(NUM_RAYCASTS_BACK, False),
                dtype=np.float32,
            )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(NUM_INPUTS,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiBinary(4)

    @staticmethod
    @functools.lru_cache(maxsize=2)
    def _fibonacci_hemisphere(
        n: int, front: bool = True
    ) -> tuple[tuple[float, float, float], ...]:
        """Distribute n directions evenly over a hemisphere using the Fibonacci lattice."""
        dirs = []
        golden = (1 + np.sqrt(5)) / 2
        for i in range(n):
            theta = np.arccos(1 - (i + 0.5) / n)
            phi = 2 * np.pi * i / golden
            x = np.sin(theta) * np.cos(phi)
            y = np.cos(theta)
            z = np.sin(theta) * np.sin(phi)
            if front:
                dirs.append((x, y, z))
            else:
                dirs.append((-x, y, z))
        return tuple(dirs)

    @staticmethod
    def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
        """Convert a unit quaternion to a 3x3 rotation matrix."""
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=np.float32,
        )

    # pylint: disable=too-many-locals, too-many-statements
    def _build_obs(self, data: CarStatePy) -> np.ndarray:
        """Pack a CarStatePy snapshot into a flat observation vector."""
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)  # type: ignore
        obs[0] = data.get_speed_kmh()
        obs[1] = 1.0 if data.get_is_finishline_cp() else 0.0
        quat_x, quat_y, quat_z, quat_w = data.get_quaternion()
        yaw = np.arctan2(
            2.0 * (quat_w * quat_z + quat_x * quat_y),
            1.0 - 2.0 * (quat_y**2 + quat_z**2),
        )
        pitch = np.arcsin(2.0 * (quat_w * quat_y - quat_z * quat_x))
        roll = np.arctan2(
            2.0 * (quat_w * quat_x + quat_y * quat_z),
            1.0 - 2.0 * (quat_x**2 + quat_y**2),
        )
        obs[2] = yaw
        obs[3] = pitch
        obs[4] = roll
        collision_impulses = data.get_collision_impulses()
        obs[5:9] = collision_impulses
        wheel_contacts = data.get_wheel_contacts()
        for i in range(4):
            contact = wheel_contacts[i]
            if contact is not None:
                obs[9 + (i * 6) : 9 + (i * 6) + 3] = contact.get_position()
                obs[9 + (i * 6) + 3 : 9 + (i * 6) + 6] = contact.get_normal()
            else:
                obs[9 + (i * 6) : 9 + (i * 6) + 6] = 0.0
        obs[33:37] = data.get_wheel_suspension_lengths()
        obs[37:41] = data.get_wheel_suspension_velocities()
        obs[41:45] = data.get_wheel_delta_rotations()
        obs[45:49] = data.get_wheel_skid_info()
        obs[49] = data.get_steering()
        controls = data.get_controls()
        obs[50] = 1.0 if controls.get_up() else 0.0
        obs[51] = 1.0 if controls.get_right() else 0.0
        obs[52] = 1.0 if controls.get_down() else 0.0
        obs[53] = 1.0 if controls.get_left() else 0.0

        pos_x, pos_y, pos_z = data.get_position()
        rot_matrix = self.quat_to_matrix(quat_x, quat_y, quat_z, quat_w)
        origin = (pos_x, pos_y, pos_z)

        world_dirs = self._ALL_DIRS @ rot_matrix.T
        distances = PolyTrackEnv._sim_worker.raycast_batch(
            origin, world_dirs, RAYCAST_MAX_DISTANCE
        )

        obs[54 : 54 + NUM_RAYCASTS_FRONT + NUM_RAYCASTS_BACK] = distances

        cp_x, cp_y, cp_z = data.get_next_checkpoint_position()
        world_offset = np.array([cp_x - pos_x, cp_y - pos_y, cp_z - pos_z])
        local_offset = rot_matrix.T @ world_offset
        obs[
            54
            + NUM_RAYCASTS_FRONT
            + NUM_RAYCASTS_BACK : 54
            + NUM_RAYCASTS_FRONT
            + NUM_RAYCASTS_BACK
            + 3
        ] = local_offset

        return obs

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the car and reward state, returning the initial observation."""
        super().reset(seed=seed, options=options)

        PolyTrackEnv._sim_worker.delete_car(self.car_id)
        PolyTrackEnv._sim_worker.create_car(self.car_id)

        PolyTrackEnv._sim_worker.set_car_controls(
            self.car_id, PlayerControllerPy(False, False, False, False, False)
        )

        data = PolyTrackEnv._sim_worker.update_car(self.car_id)

        self._prev_data = None
        self._reward_system.reset()

        observation = self._build_obs(data)
        info = {}
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Advance the simulation by one step and return
        (obs, reward, terminated, truncated, info).
        """
        assert self.action_space.contains(action), "Invalid action."

        up = bool(action[0])
        right = bool(action[1])
        down = bool(action[2])
        left = bool(action[3])

        controls = PlayerControllerPy(up, right, down, left, False)
        PolyTrackEnv._sim_worker.set_car_controls(self.car_id, controls)
        data = PolyTrackEnv._sim_worker.update_car(self.car_id)

        observation = self._build_obs(data)
        reward = self._reward_system.compute(data, self._prev_data, {})
        terminated = data.get_is_finished()
        truncated = data.get_frames() >= MAX_FRAMES
        info = {}

        self._prev_data = data
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Rendering is handled by the viewer process; no-op here."""
        return None
