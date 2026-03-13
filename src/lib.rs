//! Library for running the PolyTrack Physics (simulation_worker) and exposing it to Python via PyO3.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use pyo3::pymodule;
use wasmtime::{Caller, Engine, Extern, Instance, Linker, Memory, Module, Store};

/// HostState tracks the state of the host environment for the wasm module, including buffers for stdout/stderr and whether the module has exited.
/// It's shared across all host function calls via Arc<Mutex<>> to allow mutable access.
#[derive(Default)]
struct HostState {
    /// Buffers for stdout (fd=1) and stderr (fd=2). We buffer output until a newline so log lines appear whole.
    buffers: HashMap<i32, Vec<u8>>,
    /// Whether the wasm module has called exit() or returned from main(), which may not correspond to the process actually exiting if the JS wrapper is still alive.
    exited: bool,
    /// If exited is true, this is the code passed to exit() or the return value of main().
    exit_code: i32,
}

/// PolyTrackPhysics encapsulates the wasm instance and its host state, and provides methods to call the wasm exports.
pub struct PolyTrackPhysics {
    /// The Store holds the state for the wasm instance, including the HostState which tracks stdout/stderr buffers and exit status.
    store: Store<Arc<Mutex<HostState>>>,
    /// The wasm instance, which contains the compiled module and allows us to call its exports.
    instance: Instance,
}

impl PolyTrackPhysics {
    /// Creates a new PolyTrackPhysics instance by loading the wasm module from the specified file path.
    pub fn from_file(wasm_path: &str) -> anyhow::Result<Self> {
        let engine = Engine::default();
        let module = Module::from_file(&engine, wasm_path)?;
        Self::from_module(engine, module)
    }

    /// Creates a new PolyTrackPhysics instance from the given wasm bytes.
    /// This is useful for testing with in-memory wasm modules without needing to write them to disk.
    pub fn from_bytes(wasm_bytes: &[u8]) -> anyhow::Result<Self> {
        let engine = Engine::default();
        let module = Module::from_binary(&engine, wasm_bytes)?;
        Self::from_module(engine, module)
    }

    /// Internal helper to create a PolyTrackPhysics instance from a compiled wasm module, setting up the host functions and linking.
    fn from_module(engine: Engine, module: Module) -> anyhow::Result<Self> {
        let state = Arc::new(Mutex::new(HostState::default()));
        let mut store = Store::new(&engine, state.clone());
        let mut linker: Linker<Arc<Mutex<HostState>>> = Linker::new(&engine);

        // The wasm imports all live under module "a" with minified single-letter names.
        // See the JS wrapper for the mapping - it's the object passed to WebAssembly.instantiate.

        // Assertion failure from a C assert() macro. The four args are pointers to
        // null-terminated strings: message, filename, line number, function name.
        linker.func_wrap(
            "a",
            "i",
            |mut caller: Caller<'_, Arc<Mutex<HostState>>>,
             msg: i32,
             file: i32,
             line: i32,
             func: i32| {
                let msg_str = read_cstr(&mut caller, msg as u32);
                let file_str = read_cstr(&mut caller, file as u32);
                let func_str = read_cstr(&mut caller, func as u32);
                eprintln!("assertion failed: {msg_str}  at {file_str}:{line} ({func_str})");
                trap(&caller, 1);
            },
        )?;

        // C++ exception throw. We can't unwind across the wasm boundary so we just
        // trap - exc_ptr would normally be caught by __cxa_begin_catch.
        linker.func_wrap(
            "a",
            "a",
            |caller: Caller<'_, Arc<Mutex<HostState>>>, exc: i32, _ty: i32, _dtor: i32| {
                eprintln!("C++ exception thrown (ptr={exc})");
                trap(&caller, 1);
            },
        )?;

        linker.func_wrap("a", "e", |caller: Caller<'_, Arc<Mutex<HostState>>>| {
            eprintln!("abort()");
            trap(&caller, 1);
        })?;

        // Called by Emscripten when the runtime is shutting down cleanly,
        // not necessarily when the process is dying.
        linker.func_wrap("a", "c", |caller: Caller<'_, Arc<Mutex<HostState>>>| {
            caller.data().lock().unwrap().exited = true;
        })?;

        // setTimeout - the wasm uses this to schedule its internal timer callback
        // (export "t"). Verified via wat disassembly that export "t" has zero
        // callers inside the module and is never invoked during simulation, so
        // ignoring timer scheduling is safe here.
        linker.func_wrap(
            "a",
            "d",
            |_caller: Caller<'_, Arc<Mutex<HostState>>>, _id: i32, _ms: f64| -> i32 { 0 },
        )?;

        linker.func_wrap(
            "a",
            "h",
            |_caller: Caller<'_, Arc<Mutex<HostState>>>| -> f64 {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as f64)
                    .unwrap_or(0.0)
            },
        )?;

        // emscripten_resize_heap - called when malloc needs more memory than the
        // current wasm linear memory can provide. Returns 1 on success, 0 on failure.
        linker.func_wrap(
            "a",
            "f",
            |mut caller: Caller<'_, Arc<Mutex<HostState>>>, desired: i32| -> i32 {
                let mem = match caller.get_export("memory") {
                    Some(Extern::Memory(m)) => m,
                    _ => return 0,
                };
                let current = mem.data_size(&caller) as i64;
                let needed = desired as i64 - current;
                if needed <= 0 {
                    return 1;
                }
                let pages = ((needed + 65535) / 65536) as u64;
                match mem.grow(&mut caller, pages) {
                    Ok(_) => 1,
                    Err(_) => 0,
                }
            },
        )?;

        // WASI fd_write, used by Emscripten for printf/puts to stdout (fd=1) and
        // stderr (fd=2). Buffers output and flushes on newline so log lines appear whole.
        linker.func_wrap(
            "a",
            "g",
            |mut caller: Caller<'_, Arc<Mutex<HostState>>>,
             fd: i32,
             iov: i32,
             iovcnt: i32,
             pnum: i32|
             -> i32 {
                let mem = match caller.get_export("memory") {
                    Some(Extern::Memory(m)) => m,
                    _ => return -1,
                };

                let mut total_written: i32 = 0;
                let state_arc = caller.data().clone();
                let mut state = state_arc.lock().unwrap();
                let buf = state.buffers.entry(fd).or_default();

                for i in 0..iovcnt {
                    let base = (iov + i * 8) as usize;
                    let data = mem.data(&caller);

                    if base + 8 > data.len() {
                        break;
                    }
                    let ptr = i32::from_le_bytes(data[base..base + 4].try_into().unwrap()) as usize;
                    let len =
                        i32::from_le_bytes(data[base + 4..base + 8].try_into().unwrap()) as usize;

                    if ptr + len > data.len() {
                        break;
                    }

                    let slice = &data[ptr..ptr + len];
                    for &byte in slice {
                        if byte == b'\n' {
                            let line = String::from_utf8_lossy(buf).into_owned();
                            if fd == 1 {
                                println!("{line}");
                            } else {
                                eprintln!("{line}");
                            }
                            buf.clear();
                        } else {
                            buf.push(byte);
                        }
                    }

                    total_written += len as i32;
                }

                // Write total_written to wasm memory at pnum
                let mem_data = mem.data_mut(&mut caller);
                if (pnum as usize + 4) <= mem_data.len() {
                    mem_data[pnum as usize..pnum as usize + 4]
                        .copy_from_slice(&total_written.to_le_bytes());
                }

                0
            },
        )?;

        linker.func_wrap(
            "a",
            "b",
            |caller: Caller<'_, Arc<Mutex<HostState>>>, code: i32| {
                let mut st = caller.data().lock().unwrap();
                st.exited = true;
                st.exit_code = code;
            },
        )?;

        let instance = linker.instantiate(&mut store, &module)?;

        // "k" is __wasm_call_ctors - runs C++ static constructors and Emscripten
        // runtime init. The JS wrapper calls this immediately after instantiation.
        if let Ok(init_fn) = instance.get_typed_func::<(), ()>(&mut store, "k") {
            init_fn.call(&mut store, ())?;
        }

        Ok(Self { store, instance })
    }

    /// Helper to get the wasm linear memory export, which is needed for the various read/write/malloc/free helpers below.
    fn memory(&mut self) -> Memory {
        self.instance
            .get_memory(&mut self.store, "memory")
            .expect("wasm memory export missing")
    }

    /// Calls the wasm malloc export to allocate a buffer of the given size in wasm memory, returning the pointer.
    fn malloc(&mut self, size: i32) -> anyhow::Result<i32> {
        let f = self
            .instance
            .get_typed_func::<i32, i32>(&mut self.store, "l")?;
        Ok(f.call(&mut self.store, size)?)
    }

    /// Calls the wasm free export to deallocate a buffer previously allocated with malloc.
    fn free(&mut self, ptr: i32) -> anyhow::Result<()> {
        let f = self
            .instance
            .get_typed_func::<i32, ()>(&mut self.store, "m")?;
        Ok(f.call(&mut self.store, ptr)?)
    }

    /// Writes the given bytes into wasm memory at the specified pointer. Caller must ensure the memory is large enough.
    fn write_bytes(&mut self, ptr: i32, bytes: &[u8]) {
        let mem = self.memory();
        let data = mem.data_mut(&mut self.store);
        let start = ptr as usize;
        data[start..start + bytes.len()].copy_from_slice(bytes);
    }

    /// Reads bytes from wasm memory at the specified pointer and length, returning them as a Vec<u8>. Caller must ensure the memory is valid.
    fn read_bytes(&mut self, ptr: i32, len: usize) -> Vec<u8> {
        let mem = self.memory();
        mem.data(&self.store)[ptr as usize..ptr as usize + len].to_vec()
    }

    /// Allocates `data` in wasm memory and returns the pointer.
    /// Caller must call [`free_wasm`] when done to avoid leaking.
    pub fn alloc_bytes(&mut self, data: &[u8]) -> anyhow::Result<i32> {
        let ptr = self.malloc(data.len() as i32)?;
        self.write_bytes(ptr, data);
        Ok(ptr)
    }

    /// Frees a buffer previously allocated with [`alloc_bytes`].
    pub fn free_wasm(&mut self, ptr: i32) -> anyhow::Result<()> {
        self.free(ptr)
    }

    /// Reads a buffer from wasm memory at the specified pointer and length, returning it as a Vec<u8>.
    pub fn read_wasm(&mut self, ptr: i32, len: usize) -> Vec<u8> {
        self.read_bytes(ptr, len)
    }

    // Export names are minified - see the JS wrapper for the l/m/n/o/p/q/r/s mapping.

    /// Initializes a car collision shape from the given configuration data (pointer to wasm memory), returning a handle.
    pub fn initialize_car_collision_shape(&mut self, config_ptr: i32) -> anyhow::Result<i32> {
        let f = self
            .instance
            .get_typed_func::<i32, i32>(&mut self.store, "n")?;
        Ok(f.call(&mut self.store, config_ptr)?)
    }

    /// Initializes a track part collision shape from the given configuration data (pointer to wasm memory), returning a handle.
    pub fn add_track_part_configuration(&mut self, config_ptr: i32) -> anyhow::Result<i32> {
        let f = self
            .instance
            .get_typed_func::<i32, i32>(&mut self.store, "o")?;
        Ok(f.call(&mut self.store, config_ptr)?)
    }

    /// Returns a model handle used by the other car model functions. The config_ptr is a pointer to a wasm memory buffer containing the car configuration data.
    pub fn create_car_model(&mut self, config_ptr: i32) -> anyhow::Result<i32> {
        let f = self
            .instance
            .get_typed_func::<i32, i32>(&mut self.store, "p")?;
        Ok(f.call(&mut self.store, config_ptr)?)
    }

    /// Deletes a car model previously created with create_car_model, freeing associated resources in the wasm module.
    pub fn delete_car_model(&mut self, handle: i32) -> anyhow::Result<()> {
        let f = self
            .instance
            .get_typed_func::<i32, ()>(&mut self.store, "q")?;
        Ok(f.call(&mut self.store, handle)?)
    }

    /// Updates the car model with the given handle by advancing the simulation by dt_ms milliseconds, returning the new state of the car as a pointer to a wasm memory buffer.
    pub fn update_car_model(&mut self, handle: i32, dt_ms: f32) -> anyhow::Result<i32> {
        let f = self
            .instance
            .get_typed_func::<(i32, f32), i32>(&mut self.store, "r")?;
        Ok(f.call(&mut self.store, (handle, dt_ms))?)
    }

    /// Calls the "s" export, which runs a deterministic test of the physics simulation and returns an integer result.
    /// This is used to verify that the simulation produces consistent results across different environments.
    pub fn test_determinism(&mut self) -> anyhow::Result<i32> {
        let f = self
            .instance
            .get_typed_func::<(), i32>(&mut self.store, "s")?;
        Ok(f.call(&mut self.store, ())?)
    }
}

/// Reads a null-terminated UTF-8 string from wasm linear memory.
fn read_cstr(caller: &mut Caller<'_, Arc<Mutex<HostState>>>, ptr: u32) -> String {
    let mem = match caller.get_export("memory") {
        Some(Extern::Memory(m)) => m,
        _ => return "<no memory>".into(),
    };
    let data = mem.data(caller);
    let start = ptr as usize;
    let end = data[start..]
        .iter()
        .position(|&b| b == 0)
        .map(|i| start + i)
        .unwrap_or(data.len());
    String::from_utf8_lossy(&data[start..end]).into_owned()
}

/// Panics with an exit code, which wasmtime surfaces as a Trap on the calling side.
fn trap(caller: &Caller<'_, Arc<Mutex<HostState>>>, code: i32) {
    caller.data().lock().unwrap().exit_code = code;
    panic!("wasm abort (exit code {code})");
}

/// Example main function that loads the wasm module, creates a PolyTrackPhysics instance, and runs the determinism test, printing the result.
fn main() -> anyhow::Result<()> {
    let wasm_path = "wasm/polytrack_physics.wasm";
    let mut physics = PolyTrackPhysics::from_file(wasm_path)?;
    let r = physics.test_determinism()?;
    println!("{r}");
    Ok(())
}

/// PyO3 module definition to expose the PolyTrackPhysics functionality to Python.
#[pymodule]
mod simulation_worker {
    use pyo3::{PyResult, exceptions::PyRuntimeError, pyfunction};

    use crate::main;

    #[pyfunction]
    fn run_simulation() -> PyResult<()> {
        main().map_err(|e| PyRuntimeError::new_err(format!("simulation failed: {e}")))
    }
}
