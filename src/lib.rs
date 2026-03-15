//! Simulation Worker ported to Rust and exposed as a Python module via PyO3.

mod data_reader;
mod physics_worker;
mod simulation;

use std::sync::OnceLock;

use pyo3::pymodule;
use wasmtime::{Engine, Module};

use crate::physics_worker::create_engine;

static ENGINE: OnceLock<Engine> = OnceLock::new();
static MODULE: OnceLock<Module> = OnceLock::new();

fn engine() -> &'static Engine {
    ENGINE.get_or_init(create_engine)
}

/// Python bindings for the simulation runtime.
#[pymodule]
pub mod simulation_worker {
    use pyo3::{PyResult, exceptions::PyRuntimeError, pyclass, pymethods};
    use pyo3_stub_gen::{
        define_stub_info_gatherer,
        derive::{gen_stub_pyclass, gen_stub_pymethods},
    };

    use crate::{MODULE, engine, physics_worker::PolyTrackPhysics, simulation::SimulationWorker};

    #[gen_stub_pyclass]
    #[pyclass]
    struct SimulationWorkerPy {
        simulation: SimulationWorker,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl SimulationWorkerPy {
        #[new]
        fn new() -> PyResult<Self> {
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
            let simulation = SimulationWorker::new(pt_physics);

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
    }

    define_stub_info_gatherer!(stub_info);
}
