use pyo3_stub_gen::Result;
use simulation_worker::simulation_worker::stub_info;

fn main() -> Result<()> {
    let stub = stub_info()?;
    stub.generate()?;
    Ok(())
}
