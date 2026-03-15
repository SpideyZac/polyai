//! Simulation Worker ported to Rust and exposed as a Python module via PyO3.

mod data_reader;
mod physics_worker;
mod simulation;

use pyo3::pymodule;

/// Python bindings for the simulation runtime.
#[pymodule]
mod simulation_worker {
    use pyo3::{PyResult, exceptions::PyRuntimeError, pyfunction};

    use crate::{
        physics_worker::{PolyTrackPhysics, create_engine},
        simulation::SimulationWorker,
    };

    #[pyfunction]
    fn test() -> PyResult<i32> {
        let engine = create_engine();
        let (mut pt_physics, _) =
            PolyTrackPhysics::from_file(&engine, "wasm/polytrack_physics.wasm").map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create physics engine: {e}"))
            })?;

        let mut simulation = SimulationWorker::new(&mut pt_physics);
        simulation.init().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to initialize simulation: {e}"))
        })?;

        Ok(pt_physics.exit_code().unwrap_or(0))
    }
}
