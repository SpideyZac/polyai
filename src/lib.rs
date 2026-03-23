mod data_reader;
mod physics_worker;
mod simulation;
#[cfg(test)]
mod world_explorer;

use std::sync::OnceLock;

use pyo3::pymodule;
use wasmtime::{Engine, Module};

use crate::physics_worker::create_engine;

static ENGINE: OnceLock<Engine> = OnceLock::new();
static MODULE: OnceLock<Module> = OnceLock::new();

fn engine() -> &'static Engine {
    ENGINE.get_or_init(create_engine)
}

#[pymodule]
pub mod simulation_worker {
    use numpy::PyReadonlyArray2;
    use pyo3::{PyResult, exceptions::PyRuntimeError, pyclass, pymethods};
    use pyo3_stub_gen::{
        define_stub_info_gatherer,
        derive::{gen_stub_pyclass, gen_stub_pymethods},
    };

    use crate::{
        MODULE, engine,
        physics_worker::PolyTrackPhysics,
        simulation::{CarState, PlayerController, SimulationWorker},
    };

    #[gen_stub_pyclass]
    #[pyclass]
    #[derive(Clone)]
    pub struct PlayerControllerPy {
        up: bool,
        right: bool,
        down: bool,
        left: bool,
        reset: bool,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PlayerControllerPy {
        #[new]
        fn new(up: bool, right: bool, down: bool, left: bool, reset: bool) -> Self {
            Self {
                up,
                right,
                down,
                left,
                reset,
            }
        }

        fn get_up(&self) -> bool {
            self.up
        }

        fn get_right(&self) -> bool {
            self.right
        }

        fn get_down(&self) -> bool {
            self.down
        }

        fn get_left(&self) -> bool {
            self.left
        }

        fn get_reset(&self) -> bool {
            self.reset
        }
    }

    #[gen_stub_pyclass]
    #[pyclass]
    #[derive(Clone)]
    pub struct WheelContactPy {
        pub position: [f32; 3],
        pub normal: [f32; 3],
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl WheelContactPy {
        fn get_position(&self) -> [f32; 3] {
            self.position
        }

        fn get_normal(&self) -> [f32; 3] {
            self.normal
        }
    }

    #[gen_stub_pyclass]
    #[pyclass]
    struct CarStatePy {
        car_state: CarState,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl CarStatePy {
        fn get_frames(&self) -> u32 {
            self.car_state.frames
        }

        fn get_speed_kmh(&self) -> f32 {
            self.car_state.speed_kmh
        }

        fn get_has_started(&self) -> bool {
            self.car_state.has_started
        }

        fn get_is_finished(&self) -> bool {
            self.car_state.finish_frames.is_some()
        }

        fn get_finish_frames(&self) -> u32 {
            self.car_state.finish_frames.unwrap_or(0)
        }

        fn get_next_checkpoint_index(&self) -> u16 {
            self.car_state.next_checkpoint_index
        }

        fn get_has_checkpoint_to_respawn_at(&self) -> bool {
            self.car_state.has_checkpoint_to_respawn_at
        }

        fn get_position(&self) -> [f32; 3] {
            self.car_state.position
        }

        fn get_quaternion(&self) -> [f32; 4] {
            self.car_state.quaternion
        }

        fn get_collision_impulses(&self) -> Vec<f32> {
            self.car_state.collision_impulses.clone()
        }

        fn get_wheel_contacts(&self) -> [Option<WheelContactPy>; 4] {
            self.car_state
                .wheel_contacts
                .iter()
                .map(|contact| {
                    contact.as_ref().map(|c| WheelContactPy {
                        position: c.position,
                        normal: c.normal,
                    })
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap_or_else(|_| {
                    panic!(
                        "Expected 4 wheel contacts, got {}",
                        self.car_state.wheel_contacts.len()
                    )
                })
        }

        fn get_wheel_suspension_lengths(&self) -> [f32; 4] {
            self.car_state.wheel_suspension_lengths
        }

        fn get_wheel_suspension_velocities(&self) -> [f32; 4] {
            self.car_state.wheel_suspension_velocities
        }

        fn get_wheel_delta_rotations(&self) -> [f32; 4] {
            self.car_state.wheel_delta_rotations
        }

        fn get_wheel_skid_info(&self) -> [f32; 4] {
            self.car_state.wheel_skid_info
        }

        fn get_steering(&self) -> f32 {
            self.car_state.steering
        }

        fn get_brake_light_enabled(&self) -> bool {
            self.car_state.brake_light_enabled
        }

        fn get_controls(&self) -> PlayerControllerPy {
            PlayerControllerPy {
                up: self.car_state.controls.up,
                right: self.car_state.controls.right,
                down: self.car_state.controls.down,
                left: self.car_state.controls.left,
                reset: self.car_state.controls.reset,
            }
        }

        fn get_is_finishline_cp(&self) -> bool {
            self.car_state.is_finishline_cp
        }

        fn get_next_checkpoint_position(&self) -> [f32; 3] {
            self.car_state.next_checkpoint_position
        }
    }

    #[gen_stub_pyclass]
    #[pyclass]
    struct SimulationWorkerPy {
        simulation: SimulationWorker,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl SimulationWorkerPy {
        #[new]
        fn new(export_string: String) -> PyResult<Self> {
            let engine = engine();
            let module = MODULE.get();
            let pt_physics = if let Some(module) = module {
                PolyTrackPhysics::from_module(engine, module).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create physics engine: {e}"))
                })?
            } else {
                let (pt_physics, module) =
                    PolyTrackPhysics::from_file(engine, "wasm/polytrack_physics.wasm").map_err(
                        |e| {
                            PyRuntimeError::new_err(format!("Failed to create physics engine: {e}"))
                        },
                    )?;
                MODULE
                    .set(module)
                    .map_err(|_| PyRuntimeError::new_err("Failed to set module"))?;
                pt_physics
            };

            let pt_physics = pt_physics;
            let simulation = SimulationWorker::new(pt_physics, &export_string);

            Ok(Self { simulation })
        }

        fn init(&mut self) -> PyResult<()> {
            self.simulation.init().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to initialize simulation: {e}"))
            })
        }

        fn test_determinism(&mut self) -> PyResult<bool> {
            self.simulation.determinism_test().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to run determinism test: {e}"))
            })
        }

        fn create_car(&mut self, car_id: u32) -> PyResult<()> {
            self.simulation
                .create_car(car_id)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create car: {e}")))
        }

        fn delete_car(&mut self, car_id: u32) -> PyResult<()> {
            self.simulation
                .delete_car(car_id)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete car: {e}")))
        }

        fn set_car_controls(&mut self, car_id: u32, controls: PlayerControllerPy) -> PyResult<()> {
            let controls = PlayerController {
                up: controls.up,
                right: controls.right,
                down: controls.down,
                left: controls.left,
                reset: controls.reset,
            };
            self.simulation
                .set_car_controls(car_id, controls)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to set car controls: {e}")))
        }

        fn update_car(&mut self, car_id: u32) -> PyResult<CarStatePy> {
            let car_state = self
                .simulation
                .update_car(car_id)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to update car state: {e}")))?;
            Ok(CarStatePy { car_state })
        }

        fn raycast_batch(
            &mut self,
            origin: [f32; 3],
            directions: PyReadonlyArray2<f32>,
            max_distance: f32,
        ) -> Vec<(u32, f32)> {
            let dirs = directions.as_array();
            dirs.outer_iter()
                .map(|row| {
                    let dir = [row[0], row[1], row[2]];
                    self.simulation.raycast(origin, dir, max_distance)
                })
                .collect()
        }
    }

    define_stub_info_gatherer!(stub_info);
}
