//! PolyTrack Physics WASM Host Runtime

use std::{
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use pyo3::pymodule;
use wasmtime::{Caller, Engine, Extern, Instance, Linker, Memory, Module, Store, TypedFunc, bail};

/// Errors that can occur while executing the physics simulation.
#[derive(Debug, thiserror::Error)]
pub enum PhysicsError {
    /// The WASM module terminated with a specific exit code.
    ///
    /// This occurs when the module calls `exit()` or
    /// `__emscripten_runtime_exit`.
    #[error("wasm module exited with code {0}")]
    WasmExited(i32),

    /// Any runtime error originating from Wasmtime.
    #[error("wasm error: {0}")]
    Wasm(#[from] wasmtime::Error),
}

/// Shared runtime state used by host import functions.
///
/// The WASM module writes to file descriptors `1` (stdout) and
/// `2` (stderr). These buffers accumulate output until a newline
/// is encountered, at which point the line is flushed to the host.
#[derive(Default)]
struct HostState {
    /// Buffered stdout output.
    stdout_buf: Vec<u8>,

    /// Buffered stderr output.
    stderr_buf: Vec<u8>,

    /// Indicates whether the WASM module has exited.
    exited: bool,

    /// Exit code recorded when the module terminates.
    exit_code: i32,
}

impl HostState {
    /// Returns the exit code if the module has exited.
    fn check_exit(&self) -> Option<i32> {
        self.exited.then_some(self.exit_code)
    }

    /// Returns the buffer associated with a file descriptor.
    ///
    /// Supported descriptors:
    /// - `1` → stdout
    /// - `2` → stderr
    fn fd_buf(&mut self, fd: i32) -> Option<&mut Vec<u8>> {
        match fd {
            1 => Some(&mut self.stdout_buf),
            2 => Some(&mut self.stderr_buf),
            _ => None,
        }
    }
}

/// Arguments passed to `_initializeCarCollisionShape`.
///
/// Emscripten flattens C++ structs into raw parameters rather than passing
/// pointers, so these are the individual scalar fields of the C++ struct.
/// TODO: replace with a named struct once the C++ source is available.
type InitCarCollisionShapeArgs = (f32, i32, i32);

/// Arguments passed to `_addTrackPartConfiguration`.
type AddTrackPartConfigArgs = (
    i32,
    i32,
    i32,
    i32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    i32,
    f32,
    f32,
    f32,
);

/// Arguments passed to `_createCarModel`.
type CreateCarModelArgs = (
    i32,
    i32,
    i32,
    f32,
    f32,
    f32,
    i32,
    i32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
);

/// Arguments passed to `_updateCarModel`.
type UpdateCarModelArgs = (i32, i32, i32, i32, i32, i32, i32);

/// Cached handles to exported WASM functions.
///
/// Resolving functions once during instantiation avoids repeated
/// lookup and signature validation.
struct Exports {
    /// `_malloc`
    malloc: TypedFunc<i32, i32>,

    /// `_free`
    free: TypedFunc<i32, ()>,

    /// `_initializeCarCollisionShape`
    init_car_collision_shape: TypedFunc<InitCarCollisionShapeArgs, ()>,

    /// `_addTrackPartConfiguration`
    add_track_part_config: TypedFunc<AddTrackPartConfigArgs, ()>,

    /// `_createCarModel`
    create_car_model: TypedFunc<CreateCarModelArgs, ()>,

    /// `_deleteCarModel`
    delete_car_model: TypedFunc<i32, ()>,

    /// `_updateCarModel`
    update_car_model: TypedFunc<UpdateCarModelArgs, ()>,

    /// `_testDeterminism`
    test_determinism: TypedFunc<(), i32>,
}

/// Host runtime for the PolyTrack physics WASM module.
///
/// Each instance owns a fully isolated simulation environment.
pub struct PolyTrackPhysics {
    /// Shared host state accessible to import functions.
    store: Store<Arc<Mutex<HostState>>>,
    /// The instantiated WASM module.
    instance: Instance,
    /// Cached exports for efficient access.
    exports: Exports,
}

impl PolyTrackPhysics {
    /// Creates a runtime from a `.wasm` file on disk.
    pub fn from_file(wasm_path: &str) -> Result<Self, PhysicsError> {
        let engine = Engine::default();
        let module = Module::from_file(&engine, wasm_path)?;
        Self::from_module(engine, module)
    }

    /// Creates a runtime from raw WASM bytes.
    ///
    /// Useful when embedding the module with `include_bytes!`.
    pub fn from_bytes(wasm_bytes: &[u8]) -> Result<Self, PhysicsError> {
        let engine = Engine::default();
        let module = Module::from_binary(&engine, wasm_bytes)?;
        Self::from_module(engine, module)
    }

    /// Instantiates the module and wires up all Emscripten imports.
    fn from_module(engine: Engine, module: Module) -> Result<Self, PhysicsError> {
        let state = Arc::new(Mutex::new(HostState::default()));
        let mut store = Store::new(&engine, state.clone());
        let mut linker: Linker<Arc<Mutex<HostState>>> = Linker::new(&engine);

        // all imports live under module "a" with minified single-letter names,
        // derived by cross-referencing the JS wrapper's `Y` object:
        //   i → __assert_fail
        //   a → __cxa_throw
        //   e → abort()
        //   c → __emscripten_runtime_exit
        //   d → emscripten_set_timeout
        //   h → emscripten_date_now
        //   f → emscripten_resize_heap
        //   g → fd_write (WASI, used by printf/puts)
        //   b → exit()

        linker.func_wrap(
            "a",
            "i",
            |mut caller: Caller<'_, Arc<Mutex<HostState>>>,
             msg: i32,
             file: i32,
             line: i32,
             func: i32|
             -> Result<(), wasmtime::Error> {
                let msg_str = read_cstr(&mut caller, msg as u32);
                let file_str = read_cstr(&mut caller, file as u32);
                let func_str = read_cstr(&mut caller, func as u32);
                eprintln!("assertion failed: {msg_str}  at {file_str}:{line} ({func_str})");
                mark_exited(&caller, 134); // 134 = SIGABRT
                bail!("assertion failed")
            },
        )?;

        // c++ exceptions can't unwind across the wasm ABI boundary so we trap
        linker.func_wrap(
            "a",
            "a",
            |caller: Caller<'_, Arc<Mutex<HostState>>>,
             exc: i32,
             _ty: i32,
             _dtor: i32|
             -> Result<(), wasmtime::Error> {
                eprintln!("C++ exception thrown (ptr={exc})");
                mark_exited(&caller, 1);
                bail!("C++ exception")
            },
        )?;

        linker.func_wrap(
            "a",
            "e",
            |caller: Caller<'_, Arc<Mutex<HostState>>>| -> Result<(), wasmtime::Error> {
                eprintln!("abort()");
                mark_exited(&caller, 134);
                bail!("abort()")
            },
        )?;

        // emscripten signals clean shutdown here; we record it but don't trap
        // because the module may still flush output buffers afterward
        linker.func_wrap("a", "c", |caller: Caller<'_, Arc<Mutex<HostState>>>| {
            caller.data().lock().unwrap().exited = true;
        })?;

        // emscripten_set_timeout schedules the internal timer callback (export "t").
        // wat disassembly confirms "t" has no callers inside the module and is never
        // invoked during a simulation step, so we can safely ignore it and return a
        // dummy timer id. re-verify this if the wasm binary is ever updated.
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

        // emscripten_resize_heap is called when malloc needs more memory than the
        // current linear memory can hold. we grow by the minimum number of 64 KiB
        // pages needed and return 1 on success, 0 on failure.
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
                let pages = ((needed + 65536 - 1) / 65536) as u64;
                match mem.grow(&mut caller, pages) {
                    Ok(_) => 1,
                    Err(_) => 0,
                }
            },
        )?;

        // fd_write is the WASI syscall emscripten uses for printf/puts.
        // we line-buffer per fd so that complete log lines are printed atomically
        // rather than character by character. unknown fds are silently dropped
        // since the wasm module doesn't check the return value anyway.
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

                    if let Some(buf) = state.fd_buf(fd) {
                        for &byte in &data[ptr..ptr + len] {
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
                    }

                    total_written += len as i32;
                }

                // write the total byte count back into wasm memory at *pnum as required by WASI
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

        // __wasm_call_ctors runs C++ static constructors and emscripten runtime
        // init — the JS wrapper calls this immediately after instantiation
        if let Ok(init_fn) = instance.get_typed_func::<(), ()>(&mut store, "k") {
            init_fn.call(&mut store, ())?;
        }

        // export name mapping (minified → semantic, from the JS wrapper):
        //   l → _malloc
        //   m → _free
        //   n → _initializeCarCollisionShape
        //   o → _addTrackPartConfiguration
        //   p → _createCarModel
        //   q → _deleteCarModel
        //   r → _updateCarModel
        //   s → _testDeterminism
        //   t → timer callback (never invoked; see emscripten_set_timeout above)
        //   k → __wasm_call_ctors (called above)
        //
        // signatures were verified with wasm-objdump and wasm2wat — a type mismatch
        // here surfaces immediately at instantiation rather than buried in a call
        let exports = Exports {
            malloc: instance.get_typed_func::<i32, i32>(&mut store, "l")?,
            free: instance.get_typed_func::<i32, ()>(&mut store, "m")?,
            init_car_collision_shape: instance
                .get_typed_func::<InitCarCollisionShapeArgs, ()>(&mut store, "n")?,
            add_track_part_config: instance
                .get_typed_func::<AddTrackPartConfigArgs, ()>(&mut store, "o")?,
            create_car_model: instance.get_typed_func::<CreateCarModelArgs, ()>(&mut store, "p")?,
            delete_car_model: instance.get_typed_func::<i32, ()>(&mut store, "q")?,
            update_car_model: instance.get_typed_func::<UpdateCarModelArgs, ()>(&mut store, "r")?,
            test_determinism: instance.get_typed_func::<(), i32>(&mut store, "s")?,
        };

        Ok(Self {
            store,
            instance,
            exports,
        })
    }

    fn check_exited(&self) -> Result<(), PhysicsError> {
        if let Some(code) = self.store.data().lock().unwrap().check_exit() {
            return Err(PhysicsError::WasmExited(code));
        }
        Ok(())
    }

    /// Returns the module's linear memory.
    fn memory(&mut self) -> Memory {
        self.instance
            .get_memory(&mut self.store, "memory")
            .expect("wasm memory export missing")
    }

    /// Allocates a buffer inside the WASM heap and copies `data` into it.
    ///
    /// The returned pointer must later be freed with [`free_wasm`].
    pub fn alloc_bytes(&mut self, data: &[u8]) -> Result<i32, PhysicsError> {
        self.check_exited()?;
        let ptr = self
            .exports
            .malloc
            .call(&mut self.store, data.len() as i32)?;
        // re-fetch memory after malloc since the call may have grown the heap
        let mem = self.memory();
        mem.data_mut(&mut self.store)[ptr as usize..ptr as usize + data.len()]
            .copy_from_slice(data);
        Ok(ptr)
    }

    /// Frees a previously allocated WASM buffer.
    pub fn free_wasm(&mut self, ptr: i32) -> Result<(), PhysicsError> {
        self.check_exited()?;
        self.exports.free.call(&mut self.store, ptr)?;
        self.check_exited()
    }

    /// Reads `len` bytes from wasm linear memory starting at `ptr`.
    pub fn read_wasm(&mut self, ptr: i32, len: usize) -> Vec<u8> {
        let mem = self.memory();
        mem.data(&self.store)[ptr as usize..ptr as usize + len].to_vec()
    }

    /// Initializes a car collision shape with the given parameters.
    pub fn initialize_car_collision_shape(
        &mut self,
        args: InitCarCollisionShapeArgs,
    ) -> Result<(), PhysicsError> {
        self.check_exited()?;
        self.exports
            .init_car_collision_shape
            .call(&mut self.store, args)?;
        self.check_exited()
    }

    /// Adds a track part configuration with the given parameters.
    pub fn add_track_part_configuration(
        &mut self,
        args: AddTrackPartConfigArgs,
    ) -> Result<(), PhysicsError> {
        self.check_exited()?;
        self.exports
            .add_track_part_config
            .call(&mut self.store, args)?;
        self.check_exited()
    }

    /// Creates a car model with the given parameters.
    pub fn create_car_model(&mut self, args: CreateCarModelArgs) -> Result<(), PhysicsError> {
        self.check_exited()?;
        self.exports.create_car_model.call(&mut self.store, args)?;
        self.check_exited()
    }

    /// Deletes a car model by handle.
    pub fn delete_car_model(&mut self, handle: i32) -> Result<(), PhysicsError> {
        self.check_exited()?;
        self.exports
            .delete_car_model
            .call(&mut self.store, handle)?;
        self.check_exited()
    }

    /// Updates a car model's state with the given parameters.
    pub fn update_car_model(&mut self, args: UpdateCarModelArgs) -> Result<(), PhysicsError> {
        self.check_exited()?;
        self.exports.update_car_model.call(&mut self.store, args)?;
        self.check_exited()
    }

    /// Runs the determinism self-test built into the simulation.
    ///
    /// Returns `1` if deterministic.
    pub fn test_determinism(&mut self) -> Result<i32, PhysicsError> {
        self.check_exited()?;
        let result = self.exports.test_determinism.call(&mut self.store, ())?;
        self.check_exited()?;
        Ok(result)
    }

    /// Checks if the WASM module has exited.
    pub fn has_exited(&self) -> bool {
        self.store.data().lock().unwrap().exited
    }

    /// If the module has exited, returns the recorded exit code.
    pub fn exit_code(&self) -> Option<i32> {
        self.store.data().lock().unwrap().check_exit()
    }
}

/// Reads a null-terminated C string from WASM memory.
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

/// Marks the module as exited with the given code.
fn mark_exited(caller: &Caller<'_, Arc<Mutex<HostState>>>, code: i32) {
    let mut st = caller.data().lock().unwrap();
    st.exited = true;
    st.exit_code = code;
}

/// Standalone test entry point.
fn main() -> Result<(), PhysicsError> {
    let wasm_path = "wasm/polytrack_physics.wasm";
    let mut physics = PolyTrackPhysics::from_file(wasm_path)?;
    let r = physics.test_determinism()?;
    println!("determinism result: {r}");
    Ok(())
}

/// Python bindings for the simulation runtime.
#[pymodule]
mod simulation_worker {
    use pyo3::{PyResult, exceptions::PyRuntimeError, pyfunction};

    use crate::main;

    /// Runs the physics determinism test from Python.
    #[pyfunction]
    fn run_simulation() -> PyResult<()> {
        main().map_err(|e| PyRuntimeError::new_err(format!("simulation failed: {e}")))
    }
}
