//! Simulation Worker ported to Rust and exposed as a Python module via PyO3.

mod physics_worker;

use pyo3::pymodule;

/// Python bindings for the simulation runtime.
#[pymodule]
mod simulation_worker {}
