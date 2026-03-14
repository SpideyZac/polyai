//! PolyTrack Physics WASM Host Runtime

use std::time::{SystemTime, UNIX_EPOCH};

use pyo3::pymodule;
use wasmtime::{
    Caller, Engine, Extern, Instance, Linker, Memory, Module, Store, TypedFunc, bail, format_err,
};

/// Size of the pre-allocated scratch buffer inside the WASM heap (bytes).
///
/// Any `alloc_bytes` call whose payload fits within this size reuses the
/// single persistent allocation instead of calling `malloc`/`free`.
const SCRATCH_BUF_SIZE: usize = 64 * 1024; // 64 KiB

/// Errors that can occur while executing the physics simulation.
#[derive(Debug, thiserror::Error)]
pub enum PhysicsError {
    /// The WASM module terminated with a specific exit code.
    ///
    /// This occurs when the module calls `exit()` or
    /// `__emscripten_runtime_exit`.
    #[error("wasm module exited with code {0}")]
    WasmExited(i32),

    /// A host-side bounds check on WASM linear memory failed.
    #[error("out-of-bounds wasm memory access: offset {offset} + len {len} > heap size {heap}")]
    OutOfBounds {
        offset: usize,
        len: usize,
        heap: usize,
    },

    /// Any runtime error originating from Wasmtime.
    #[error("wasm error: {0}")]
    Wasm(#[from] wasmtime::Error),
}

/// Shared runtime state used by host import functions.
///
/// WASM execution is inherently single-threaded — one `Store` is never
/// shared across threads — so no `Arc<Mutex<>>` wrapper is needed here.
#[derive(Default)]
struct HostState {
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
    /// Host state (exit flags, etc.) stored directly in the `Store`.
    ///
    /// WASM execution via Wasmtime is single-threaded, so no `Arc<Mutex<>>`
    /// wrapper is needed.
    store: Store<HostState>,

    /// The instantiated WASM module.
    instance: Instance,

    /// Cached exports for efficient access.
    exports: Exports,

    /// Pointer to a persistent `SCRATCH_BUF_SIZE`-byte allocation inside the
    /// WASM heap.  Reused by `alloc_bytes` when the payload fits, avoiding a
    /// `malloc`/`free` round-trip on every call.
    scratch_ptr: i32,
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
        let mut store = Store::new(&engine, HostState::default());
        let mut linker: Linker<HostState> = Linker::new(&engine);

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
            |mut caller: Caller<'_, HostState>,
             msg: i32,
             file: i32,
             line: i32,
             func: i32|
             -> Result<(), wasmtime::Error> {
                let msg_str = read_cstr(&mut caller, msg as u32);
                let file_str = read_cstr(&mut caller, file as u32);
                let func_str = read_cstr(&mut caller, func as u32);
                eprintln!("assertion failed: {msg_str}  at {file_str}:{line} ({func_str})");
                mark_exited(&mut caller, 134); // 134 = SIGABRT
                bail!("assertion failed")
            },
        )?;

        // c++ exceptions can't unwind across the wasm ABI boundary so we trap
        linker.func_wrap(
            "a",
            "a",
            |mut caller: Caller<'_, HostState>,
             exc: i32,
             _ty: i32,
             _dtor: i32|
             -> Result<(), wasmtime::Error> {
                eprintln!("C++ exception thrown (ptr={exc})");
                mark_exited(&mut caller, 1);
                bail!("C++ exception")
            },
        )?;

        linker.func_wrap(
            "a",
            "e",
            |mut caller: Caller<'_, HostState>| -> Result<(), wasmtime::Error> {
                eprintln!("abort()");
                mark_exited(&mut caller, 134);
                bail!("abort()")
            },
        )?;

        // never called.
        linker.func_wrap("a", "c", |_caller: Caller<'_, HostState>| {})?;

        // never called.
        linker.func_wrap(
            "a",
            "d",
            |_caller: Caller<'_, HostState>, _id: i32, _ms: f64| -> i32 { 0 },
        )?;

        linker.func_wrap("a", "h", |_caller: Caller<'_, HostState>| -> f64 {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as f64)
                .unwrap_or(0.0)
        })?;

        // emscripten_resize_heap is called when malloc needs more memory than the
        // current linear memory can hold. we grow by the minimum number of 64 KiB
        // pages needed and return 1 on success, 0 on failure.
        linker.func_wrap(
            "a",
            "f",
            |mut caller: Caller<'_, HostState>, desired: i32| -> i32 {
                let mem = match caller.get_export("j") {
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

        // never called.
        linker.func_wrap(
            "a",
            "g",
            |mut _caller: Caller<'_, HostState>, _fd: i32, _iov: i32, _iovcnt: i32, _pnum: i32| 1,
        )?;

        // never called.
        linker.func_wrap("a", "b", |_caller: Caller<'_, HostState>, _code: i32| {})?;

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
        //   u → stackRestore (never called; not needed for interface)
        //   v → stackAlloc (never called; not needed for interface)
        //   w → stackSave (never called; not needed for interface)
        //   j → memory (never used; not needed for interface)
        //   __indirect_function_table → functionTable (never used; not needed for interface)
        //   k → __wasm_call_ctors (called above)
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

        // Pre-allocate a scratch buffer. Any alloc_bytes call that fits within
        // SCRATCH_BUF_SIZE will write directly into this region rather than
        // calling malloc/free.
        let scratch_ptr = exports.malloc.call(&mut store, SCRATCH_BUF_SIZE as i32)?;
        if scratch_ptr == 0 {
            return Err(format_err!("failed to pre-allocate scratch buffer").into());
        }

        Ok(Self {
            store,
            instance,
            exports,
            scratch_ptr,
        })
    }

    fn check_exited(&self) -> Result<(), PhysicsError> {
        if let Some(code) = self.store.data().check_exit() {
            return Err(PhysicsError::WasmExited(code));
        }
        Ok(())
    }

    /// Returns the module's linear memory.
    fn memory(&mut self) -> Memory {
        self.instance
            .get_memory(&mut self.store, "j")
            .expect("wasm memory export missing")
    }

    /// Copies `data` into the WASM heap and returns a pointer to it.
    ///
    /// If `data` fits within `SCRATCH_BUF_SIZE`, the pre-allocated scratch
    /// buffer is reused and `is_scratch` is `true`; the caller must **not**
    /// free the pointer in that case.  For larger payloads a fresh `malloc`
    /// allocation is made (`is_scratch = false`) and the caller is responsible
    /// for freeing it with [`free_wasm`].
    ///
    /// Returns `(ptr, is_scratch)`.
    pub fn alloc_bytes(&mut self, data: &[u8]) -> Result<(i32, bool), PhysicsError> {
        self.check_exited()?;

        let (ptr, is_scratch) = if data.len() <= SCRATCH_BUF_SIZE {
            (self.scratch_ptr, true)
        } else {
            let p = self
                .exports
                .malloc
                .call(&mut self.store, data.len() as i32)?;
            (p, false)
        };

        // Bounds-check before writing: re-fetch memory since malloc may have
        // grown the heap.
        let mem = self.memory();
        let heap_size = mem.data_size(&self.store);
        let offset = ptr as usize;
        if offset
            .checked_add(data.len())
            .is_none_or(|end| end > heap_size)
        {
            return Err(PhysicsError::OutOfBounds {
                offset,
                len: data.len(),
                heap: heap_size,
            });
        }

        mem.data_mut(&mut self.store)[offset..offset + data.len()].copy_from_slice(data);
        Ok((ptr, is_scratch))
    }

    /// Frees a previously allocated WASM buffer.
    ///
    /// Do **not** call this for pointers obtained from `alloc_bytes` when
    /// `is_scratch` was `true` — the scratch buffer is owned by the runtime
    /// and freed in `Drop`.
    pub fn free_wasm(&mut self, ptr: i32) -> Result<(), PhysicsError> {
        self.check_exited()?;
        self.exports.free.call(&mut self.store, ptr)?;
        self.check_exited()
    }

    /// Reads `len` bytes from wasm linear memory starting at `ptr`.
    pub fn read_wasm(&mut self, ptr: i32, len: usize) -> Result<Vec<u8>, PhysicsError> {
        let mem = self.memory();
        let heap_size = mem.data_size(&self.store);
        let offset = ptr as usize;

        if offset.checked_add(len).is_none_or(|end| end > heap_size) {
            return Err(PhysicsError::OutOfBounds {
                offset,
                len,
                heap: heap_size,
            });
        }

        Ok(mem.data(&self.store)[offset..offset + len].to_vec())
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
        self.store.data().exited
    }

    /// If the module has exited, returns the recorded exit code.
    pub fn exit_code(&self) -> Option<i32> {
        self.store.data().check_exit()
    }
}

impl Drop for PolyTrackPhysics {
    /// Releases the pre-allocated scratch buffer back to the WASM heap.
    ///
    /// Skipped if the module has already exited, since the heap is gone.
    fn drop(&mut self) {
        if !self.has_exited() {
            let _ = self.exports.free.call(&mut self.store, self.scratch_ptr);
        }
    }
}

/// Reads a null-terminated C string from WASM memory.
fn read_cstr(caller: &mut Caller<'_, HostState>, ptr: u32) -> String {
    let mem = match caller.get_export("j") {
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
fn mark_exited(caller: &mut Caller<'_, HostState>, code: i32) {
    let st = caller.data_mut();
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
