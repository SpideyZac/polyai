//! Glue code between the host and the PolyTrack physics WASM module.
//!
//! The WASM module was originally compiled by Emscripten from C++, so its
//! import/export names are minified single-letter identifiers. This file
//! translates between those minified names and human-readable Rust types,
//! wires up all host imports the module expects (assert, abort, heap growth,
//! etc.), and exposes a safe [`PolyTrackPhysics`] wrapper that owns the entire
//! simulation lifecycle.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────┐
//! │  PolyTrackPhysics                        │
//! │  ┌──────────────┐   ┌──────────────────┐ │
//! │  │ Store<Host>  │   │ Exports (cached) │ │
//! │  │  · exited    │   │  · malloc / free │ │
//! │  │  · exit_code │   │  · physics fns   │ │
//! │  └──────────────┘   └──────────────────┘ │
//! │  ┌──────────────────────────────────────┐ │
//! │  │ Linear memory ("j")                  │ │
//! │  │  [scratch_ptr ... scratch_ptr+64KiB] │ │
//! │  │  [malloc'd regions as needed]        │ │
//! │  └──────────────────────────────────────┘ │
//! └──────────────────────────────────────────┘
//! ```
//!
//! # Fuel
//!
//! Wasmtime's *fuel* mechanism lets us bound how long the module runs.  Each
//! WASM instruction burns one unit of fuel; when the tank hits zero Wasmtime
//! traps.  We start with 10 million units (see [`PolyTrackPhysics::from_module`])
//! and surface that trap as [`PhysicsError::OutOfFuel`].

use std::num::NonZeroI32;

use wasmtime::{
    Caller, Config, Engine, Extern, Linker, Memory, Module, OptLevel, Store, StoreLimits,
    StoreLimitsBuilder, TypedFunc, WasmParams, WasmResults, bail, format_err,
};

/// Size of the pre-allocated scratch buffer inside the WASM heap (bytes).
///
/// Allocating through `malloc`/`free` on every call is expensive because it
/// crosses the host–guest boundary twice and triggers the allocator's internal
/// bookkeeping.  Instead, we keep a single persistent region of this size and
/// reuse it for any write that fits.  64 KiB covers the vast majority of
/// per-frame payloads without wasting significant memory.
///
/// See [`PolyTrackPhysics::alloc_bytes`] for how this is used.
const SCRATCH_BUF_SIZE: usize = 64 * 1024;

/// Extra bytes requested when allocating the scratch buffer so we can align
/// the usable start to a 16-byte boundary.
///
/// SIMD load/store instructions inside the WASM module require 16-byte
/// alignment.  We over-allocate by this amount and round the raw pointer up,
/// intentionally wasting the few bytes before the aligned address.
const SCRATCH_BUF_ALIGN_PADDING: usize = 16;

/// Maximum WASM linear memory size in bytes.
///
/// Enforced via Wasmtime's [`StoreLimits`] so every `memory.grow` instruction
/// is checked against this cap.  The physics simulation should never need more
/// than a few MiB; 64 MiB is generous headroom without risking runaway growth.
const MAX_MEMORY_BYTES: usize = 64 * 1024 * 1024;

/// Initial fuel budget for a single [`PolyTrackPhysics`] instance.
///
/// Wasmtime burns one unit per WASM instruction.  10 million units is
/// enough for a full simulation step with headroom to spare; hitting this
/// limit almost certainly indicates an infinite loop or a logic bug rather
/// than a legitimately expensive computation.
///
/// See [`PolyTrackPhysics::reset_fuel`] if you need to replenish mid-run.
pub const INITIAL_FUEL: u64 = 10_000_000;

/// Exit code used when the WASM module calls `abort()` or fails an assertion.
///
/// Mirrors the Unix convention where SIGABRT (signal 6) produces exit status
/// 128 + 6 = 134 when the process is killed by the signal handler.
const EXIT_CODE_ABORT: i32 = 134;

/// Creates a Wasmtime [`Engine`] configured for the physics module.
///
/// Settings chosen here:
/// - **Fuel metering** — lets callers cap execution time by limiting the number
///   of WASM instructions executed.  Without this an infinite loop in the
///   physics code would hang the process.
/// - **Cranelift `Speed` optimisation** — the physics hot loop is CPU-bound, so
///   we pay the slightly longer compile time upfront for faster steady-state
///   throughput.
/// - **SIMD** — the Emscripten-compiled module uses 128-bit SIMD intrinsics for
///   vector math; the engine must be told to enable them explicitly.
///
/// Panics if the engine cannot be constructed (practically, this only happens
/// if the host CPU lacks SSE2/NEON when SIMD is enabled).
pub fn create_engine() -> Engine {
    let mut config = Config::new();

    config.consume_fuel(true);
    config.cranelift_opt_level(OptLevel::Speed);
    config.wasm_simd(true);

    Engine::new(&config).unwrap()
}

/// A region of WASM linear memory that holds a serialised payload.
///
/// The distinction between the two variants determines whether the backing
/// memory needs to be freed after use.  Callers should always go through
/// [`PolyTrackPhysics::free_wasm`] rather than calling `free` manually — it
/// no-ops on scratch buffers automatically.
pub enum WasmBuffer {
    /// The payload was written into the pre-allocated scratch region.
    ///
    /// The inner value is the byte offset within WASM linear memory.  This
    /// buffer is owned by [`PolyTrackPhysics`] and must **not** be freed by the
    /// caller.
    Scratch(i32),

    /// The payload required a fresh `malloc` call because it exceeded the
    /// scratch size.
    ///
    /// The inner value is the pointer returned by `malloc`.  The caller is
    /// responsible for calling [`PolyTrackPhysics::free_wasm`] when done.
    Owned(i32),
}

impl WasmBuffer {
    /// Constructs a [`WasmBuffer`] from a raw pointer and a flag that says
    /// whether the pointer points into the scratch region.
    pub fn new(ptr: i32, is_scratch: bool) -> Self {
        if is_scratch {
            WasmBuffer::Scratch(ptr)
        } else {
            WasmBuffer::Owned(ptr)
        }
    }

    /// Returns the raw byte offset (pointer) into WASM linear memory.
    pub fn ptr(&self) -> i32 {
        match self {
            WasmBuffer::Scratch(offset) => *offset,
            WasmBuffer::Owned(ptr) => *ptr,
        }
    }

    /// Returns `true` if this buffer lives in the scratch region and therefore
    /// does not need to be freed.
    pub fn is_scratch(&self) -> bool {
        matches!(self, WasmBuffer::Scratch(_))
    }
}

/// Errors that can occur while driving the physics simulation.
#[derive(Debug, thiserror::Error)]
pub enum PhysicsError {
    /// The module called `exit()` or `__emscripten_runtime_exit`.
    ///
    /// Once this happens, the WASM linear memory and internal state are
    /// considered invalid.  The [`PolyTrackPhysics`] instance must be
    /// discarded; continued use will return this same error.
    #[error("wasm module exited with code {0}")]
    WasmExited(i32),

    /// A host-side bounds check on WASM linear memory failed.
    ///
    /// This should never occur under normal operation; it indicates a bug in
    /// the argument marshalling code or a corrupt pointer from the WASM side.
    #[error("out-of-bounds wasm memory access: offset {offset} + len {len} > heap size {heap}")]
    OutOfBounds {
        offset: usize,
        len: usize,
        heap: usize,
    },

    /// The module executed more WASM instructions than the configured fuel
    /// budget allows.
    ///
    /// The caller can recover by calling [`PolyTrackPhysics::reset_fuel`] if
    /// partial progress is acceptable, or treat this as a fatal error if
    /// determinism is required.
    #[error("wasm ran out of fuel")]
    OutOfFuel,

    /// Any other runtime error from Wasmtime (trap, link error, etc.).
    #[error("wasm error: {0}")]
    Wasm(#[from] wasmtime::Error),
}

/// Host-side data stored inside the Wasmtime [`Store`].
///
/// Wasmtime host import callbacks receive a `Caller<HostState>` that grants
/// mutable access to this struct, letting import handlers record exit events
/// without needing any shared state or synchronisation primitives.
#[derive(Default)]
struct HostState {
    /// Set to `true` by [`mark_exited`] the first time the module calls any
    /// exit path (assert failure, C++ exception, `abort`, or `exit`).
    exited: bool,

    /// The exit code recorded by the last exit path that fired.
    ///
    /// Meaningful only when `exited == true`.  Follows Unix conventions:
    /// non-zero means abnormal termination (e.g. 134 = SIGABRT).
    exit_code: i32,

    /// Wasmtime resource limiter that caps total linear memory growth.
    ///
    /// Passed to [`Store::limiter`] during construction so Wasmtime checks it
    /// on every `memory.grow` instruction, preventing unbounded heap expansion.
    limits: StoreLimits,
}

impl HostState {
    /// Returns the exit code if the module has terminated, or `None` if it is
    /// still running.
    fn check_exit(&self) -> Option<i32> {
        self.exited.then_some(self.exit_code)
    }
}

/// Raw scalar arguments for `_initializeCarCollisionShape`.
///
/// Emscripten flattens C++ value structs into individual parameters rather than
/// passing a pointer, so the WASM function signature is a flat list of scalars
/// corresponding to the fields of the original C++ struct.
///
/// TODO: replace with a named struct once the C++ source is available.
pub type InitCarCollisionShapeArgs = (f32, i32, i32);

/// Raw scalar arguments for `_addTrackPartConfiguration`.
pub type AddTrackPartConfigArgs = (
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

/// Raw scalar arguments for `_createCarModel`.
pub type CreateCarModelArgs = (
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

/// Raw scalar arguments for `_updateCarModel`.
pub type UpdateCarModelArgs = (i32, i32, i32, i32, i32, i32, i32);

/// Resolved handles to every exported WASM function we call from the host.
///
/// [`TypedFunc`] is both type-checked and pre-validated at instantiation time,
/// so calling through these handles has no runtime overhead from name lookups or
/// signature verification.
///
/// Export name mapping (minified JS wrapper name → semantic name):
/// ```text
/// l → _malloc
/// m → _free
/// n → _initializeCarCollisionShape
/// o → _addTrackPartConfiguration
/// p → _createCarModel
/// q → _deleteCarModel
/// r → _updateCarModel
/// s → _testDeterminism
/// j → memory  (linear memory, not a function)
/// ```
#[derive(Clone)]
pub struct Exports {
    pub malloc: TypedFunc<i32, i32>,
    pub free: TypedFunc<i32, ()>,
    pub init_car_collision_shape: TypedFunc<InitCarCollisionShapeArgs, ()>,
    pub add_track_part_config: TypedFunc<AddTrackPartConfigArgs, ()>,
    pub _create_car_model: TypedFunc<CreateCarModelArgs, ()>,
    pub _delete_car_model: TypedFunc<i32, ()>,
    pub _update_car_model: TypedFunc<UpdateCarModelArgs, ()>,
    pub _test_determinism: TypedFunc<(), i32>,
}

/// An isolated instance of the PolyTrack physics simulation.
///
/// Each `PolyTrackPhysics` owns a complete Wasmtime execution context: a
/// [`Store`] (which holds the instance's linear memory, globals, and tables),
/// cached function handles, and a scratch allocation for efficient data
/// transfer.
///
/// # Thread safety
///
/// [`Store`] is `!Send`, so this type cannot be sent across threads.  If you
/// need parallel simulations, create one instance per thread.
///
/// # Lifecycle
///
/// 1. Construct with [`PolyTrackPhysics::from_file`] or [`from_module`].
/// 2. Call physics functions via the typed methods.
/// 3. On drop the scratch buffer is freed; if the module has already exited
///    the free is skipped since the heap is gone.
pub struct PolyTrackPhysics {
    store: Store<HostState>,

    /// Handle to the module's linear memory export (`"j"`), cached to avoid
    /// a hash-map lookup on every read/write.
    memory: Memory,

    exports: Exports,

    /// Pointer to a persistent [`SCRATCH_BUF_SIZE`]-byte region inside the
    /// WASM heap.
    ///
    /// Aligned to 16 bytes at allocation time so SIMD loads/stores inside
    /// the WASM module never fault on unaligned access.  The raw malloc'd
    /// pointer might be slightly lower; `scratch_ptr` is the aligned address
    /// we actually write to.
    ///
    /// `NonZeroI32` ensures the pointer can be stored without an extra
    /// "is-null" check and rules out the zero address, which WASM treats as
    /// null.
    scratch_ptr: NonZeroI32,
}

impl PolyTrackPhysics {
    /// Instantiates the physics module from a `.wasm` file on disk.
    ///
    /// Returns both the ready-to-use [`PolyTrackPhysics`] instance *and* the
    /// compiled [`Module`].  Holding onto the `Module` lets callers create
    /// additional instances cheaply via [`from_module`] without re-parsing and
    /// re-compiling the bytecode.
    pub fn from_file(engine: &Engine, wasm_path: &str) -> Result<(Self, Module), PhysicsError> {
        let module = Module::from_file(engine, wasm_path)?;
        Ok((Self::from_module(engine, module.clone())?, module))
    }

    /// Instantiates the module and wires up all Emscripten host imports.
    ///
    /// # Import wiring
    ///
    /// The Emscripten runtime expects a set of host functions under the module
    /// name `"a"` with minified single-letter export names.  The JS wrapper's
    /// `Y` object documents the mapping:
    ///
    /// | Minified | Semantic              | Notes                                       |
    /// |----------|-----------------------|---------------------------------------------|
    /// | `i`      | `__assert_fail`       | Logs the failure and traps                  |
    /// | `a`      | `__cxa_throw`         | C++ exceptions can't cross WASM ABI; traps  |
    /// | `e`      | `abort()`             | Marks exit code 134 (SIGABRT) and traps     |
    /// | `c`      | `__emscripten_runtime_exit` | Never actually called at runtime      |
    /// | `d`      | `emscripten_set_timeout` | Never called; physics doesn't use timers |
    /// | `h`      | `emscripten_date_now` | Returns 0; the C++ code never reads the clock |
    /// | `f`      | `emscripten_resize_heap` | Grows linear memory on `malloc` pressure |
    /// | `g`      | `fd_write` (WASI)     | Never called; `printf` paths are dead code  |
    /// | `b`      | `exit()`              | Never called at runtime                     |
    ///
    /// After linking, `__wasm_call_ctors` (`"k"`) is invoked to run C++ static
    /// constructors and complete Emscripten runtime initialisation — equivalent
    /// to what the JS wrapper does immediately after `WebAssembly.instantiate`.
    pub fn from_module(engine: &Engine, module: Module) -> Result<Self, PhysicsError> {
        let state = HostState {
            exited: false,
            exit_code: 0,
            limits: StoreLimitsBuilder::new()
                .memory_size(MAX_MEMORY_BYTES)
                .build(),
        };
        let mut store = Store::new(engine, state);
        store.limiter(|state| &mut state.limits);
        store.set_fuel(INITIAL_FUEL)?;
        let mut linker: Linker<HostState> = Linker::new(engine);

        // `__assert_fail(msg, file, line, func)` — print the assertion message
        // and terminate; we map line to an i32 because WASM doesn't have a
        // native unsigned type in the Wasmtime typed API.
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
                mark_exited(&mut caller, EXIT_CODE_ABORT);
                bail!("assertion failed")
            },
        )?;

        // `__cxa_throw` — C++ exception unwinding cannot cross the WASM ABI
        // boundary, so we trap immediately.  The three parameters are the
        // exception object pointer, the type-info pointer, and the destructor;
        // we don't need any of them.
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

        // `abort()` — unconditional abnormal termination.
        linker.func_wrap(
            "a",
            "e",
            |mut caller: Caller<'_, HostState>| -> Result<(), wasmtime::Error> {
                eprintln!("abort()");
                mark_exited(&mut caller, EXIT_CODE_ABORT);
                bail!("abort()")
            },
        )?;

        // `__emscripten_runtime_exit` — the Emscripten runtime calls this
        // internally, but it never fires in practice for this module.
        linker.func_wrap("a", "c", |_caller: Caller<'_, HostState>| {})?;

        // `emscripten_set_timeout` — the physics code doesn't schedule any
        // async work, so this is a no-op stub returning a dummy timer ID.
        linker.func_wrap(
            "a",
            "d",
            |_caller: Caller<'_, HostState>, _id: i32, _ms: f64| -> i32 { 0 },
        )?;

        // `emscripten_date_now` — the original C++ never queries wall-clock
        // time; Emscripten calls this internally for its own bookkeeping but
        // the returned value doesn't affect simulation output.
        linker.func_wrap("a", "h", |_caller: Caller<'_, HostState>| -> f64 { 0.0 })?;

        // `emscripten_resize_heap(desired_bytes)` — called by `malloc` when the
        // current linear memory can't satisfy an allocation.  We compute the
        // minimum number of 64 KiB pages needed to reach `desired` and attempt
        // to grow.  Returns 1 on success, 0 on failure (which causes malloc to
        // return NULL, which the caller must check).
        linker.func_wrap(
            "a",
            "f",
            |mut caller: Caller<'_, HostState>, desired: i32| -> i32 {
                let mem = match caller.get_export("j") {
                    Some(Extern::Memory(m)) => m,
                    _ => return 0,
                };

                let desired = desired as usize;
                let current = mem.data_size(&caller);

                if desired <= current {
                    // Memory is already large enough; nothing to do.
                    return 1;
                }

                let grow = desired - current;
                // WASM pages are always 64 KiB; div_ceil avoids under-growing.
                let pages = grow.div_ceil(65536) as u64;

                match mem.grow(&mut caller, pages) {
                    Ok(_) => 1,
                    Err(_) => 0,
                }
            },
        )?;

        // `fd_write` (WASI) — wired up because Emscripten links against WASI's
        // I/O primitives even when the C++ source doesn't call printf/puts.
        // The physics module never actually writes anything to stdout/stderr,
        // so we return 1 (success) without doing anything.
        linker.func_wrap(
            "a",
            "g",
            |mut _caller: Caller<'_, HostState>, _fd: i32, _iov: i32, _iovcnt: i32, _pnum: i32| 1,
        )?;

        // `exit(code)` — likewise never called at runtime for this module.
        linker.func_wrap("a", "b", |_caller: Caller<'_, HostState>, _code: i32| {})?;

        let instance = linker.instantiate(&mut store, &module)?;

        // Run C++ static constructors and Emscripten internal initialisation.
        // The JS wrapper does this immediately after instantiation; skipping it
        // leaves global objects uninitialised and will cause subtle corruption.
        if let Ok(init_fn) = instance.get_typed_func::<(), ()>(&mut store, "k") {
            init_fn.call(&mut store, ())?;
        }

        let exports = Exports {
            malloc: instance.get_typed_func::<i32, i32>(&mut store, "l")?,
            free: instance.get_typed_func::<i32, ()>(&mut store, "m")?,
            init_car_collision_shape: instance
                .get_typed_func::<InitCarCollisionShapeArgs, ()>(&mut store, "n")?,
            add_track_part_config: instance
                .get_typed_func::<AddTrackPartConfigArgs, ()>(&mut store, "o")?,
            _create_car_model: instance
                .get_typed_func::<CreateCarModelArgs, ()>(&mut store, "p")?,
            _delete_car_model: instance.get_typed_func::<i32, ()>(&mut store, "q")?,
            _update_car_model: instance
                .get_typed_func::<UpdateCarModelArgs, ()>(&mut store, "r")?,
            _test_determinism: instance.get_typed_func::<(), i32>(&mut store, "s")?,
        };

        // Pre-allocate the scratch buffer.  We request SCRATCH_BUF_SIZE + 16
        // extra bytes so we can align the usable start to a 16-byte boundary
        // without risking an overrun — SIMD load/store instructions in the WASM
        // module require 16-byte alignment.
        let raw = exports.malloc.call(
            &mut store,
            (SCRATCH_BUF_SIZE + SCRATCH_BUF_ALIGN_PADDING) as i32,
        )?;

        if raw == 0 {
            return Err(format_err!("failed to allocate scratch buffer").into());
        }

        // Round up to the next 16-byte boundary.  The few bytes before this
        // address (if any) are intentionally wasted.
        let scratch_ptr = NonZeroI32::new(
            (raw + SCRATCH_BUF_ALIGN_PADDING as i32 - 1) & !(SCRATCH_BUF_ALIGN_PADDING as i32 - 1),
        )
        .ok_or_else(|| format_err!("failed to pre-allocate scratch buffer"))?;

        let memory = instance
            .get_memory(&mut store, "j")
            .ok_or_else(|| format_err!("wasm memory export missing"))?;

        Ok(Self {
            store,
            exports,
            memory,
            scratch_ptr,
        })
    }

    /// Returns `Err(PhysicsError::WasmExited)` if the module has terminated.
    ///
    /// Called at the top of every public method so callers get a clear error
    /// instead of a confusing Wasmtime trap when they accidentally continue
    /// using an instance after it has exited.
    fn check_exited(&self) -> Result<(), PhysicsError> {
        if let Some(code) = self.store.data().check_exit() {
            return Err(PhysicsError::WasmExited(code));
        }
        Ok(())
    }

    /// Copies `data` into the WASM heap and returns a [`WasmBuffer`] pointing
    /// to it.
    ///
    /// If `data` fits within [`SCRATCH_BUF_SIZE`], the write goes into the
    /// pre-allocated scratch region and no `malloc`/`free` round-trip occurs.
    /// For larger payloads a fresh allocation is made; the caller must free it
    /// with [`free_wasm`] when done.
    ///
    /// Zero-length slices are handled as a special case: they return a pointer
    /// to the scratch region without writing anything, which avoids handing the
    /// caller a null pointer they might accidentally pass to `free`.
    pub fn alloc_bytes(&mut self, data: &[u8]) -> Result<WasmBuffer, PhysicsError> {
        self.check_exited()?;

        if data.is_empty() {
            return Ok(WasmBuffer::new(self.scratch_ptr.get(), true));
        }

        let (ptr, is_scratch) = if data.len() <= SCRATCH_BUF_SIZE {
            (self.scratch_ptr.get(), true)
        } else {
            let p = self
                .exports
                .malloc
                .call(&mut self.store, data.len() as i32)?;
            if p == 0 {
                return Err(format_err!("wasm malloc failed").into());
            }
            (p, false)
        };

        // Re-check heap size after malloc — the allocation may have triggered
        // `emscripten_resize_heap`, which grows linear memory.  Using a stale
        // size here would produce a false-positive bounds error.
        let heap_size = self.memory.data(&self.store).len();
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

        self.memory.data_mut(&mut self.store)[offset..offset + data.len()].copy_from_slice(data);
        Ok(WasmBuffer::new(ptr, is_scratch))
    }

    /// Frees a WASM heap buffer previously returned by [`alloc_bytes`].
    ///
    /// Scratch buffers are silently ignored because they are owned by
    /// [`PolyTrackPhysics`] and will be freed on drop.
    pub fn free_wasm(&mut self, buffer: WasmBuffer) -> Result<(), PhysicsError> {
        self.check_exited()?;
        if !buffer.is_scratch() {
            self.exports.free.call(&mut self.store, buffer.ptr())?;
        }
        // Check again after the call — `free` might have triggered exit somehow.
        self.check_exited()
    }

    /// Returns an immutable view into WASM linear memory at `[ptr, ptr+len)`.
    ///
    /// Performs a host-side bounds check before returning the slice.  The check
    /// is against the current heap size, which may have grown since the pointer
    /// was allocated.
    pub fn _wasm_slice(&self, ptr: i32, len: usize) -> Result<&[u8], PhysicsError> {
        let heap_size = self.memory.data(&self.store).len();
        let offset = ptr as usize;

        if offset.checked_add(len).is_none_or(|end| end > heap_size) {
            return Err(PhysicsError::OutOfBounds {
                offset,
                len,
                heap: heap_size,
            });
        }

        Ok(&self.memory.data(&self.store)[offset..offset + len])
    }

    /// Returns a mutable view into WASM linear memory at `[ptr, ptr+len)`.
    ///
    /// See [`wasm_slice`] for bounds-checking behaviour.
    pub fn _wasm_slice_mut(&mut self, ptr: i32, len: usize) -> Result<&mut [u8], PhysicsError> {
        let heap_size = self.memory.data(&self.store).len();
        let offset = ptr as usize;

        if offset.checked_add(len).is_none_or(|end| end > heap_size) {
            return Err(PhysicsError::OutOfBounds {
                offset,
                len,
                heap: heap_size,
            });
        }

        Ok(&mut self.memory.data_mut(&mut self.store)[offset..offset + len])
    }

    /// Reads `len` bytes starting at `ptr` from WASM linear memory and returns
    /// them as an owned `Vec<u8>`.
    ///
    /// Prefer [`wasm_slice`] when a borrow is sufficient; this method allocates
    /// and copies.
    pub fn _read_wasm(&mut self, ptr: i32, len: usize) -> Result<Vec<u8>, PhysicsError> {
        let heap_size = self.memory.data(&self.store).len();
        let offset = ptr as usize;

        if offset.checked_add(len).is_none_or(|end| end > heap_size) {
            return Err(PhysicsError::OutOfBounds {
                offset,
                len,
                heap: heap_size,
            });
        }

        Ok(self.memory.data(&self.store)[offset..offset + len].to_vec())
    }

    /// Returns a reference to the cached WASM export handles.
    ///
    /// Clones the [`Exports`] struct so that callers can hold onto the function handles without needing to borrow the entire [`PolyTrackPhysics`] instance.
    /// The result should be a long-lived struct to reduce the overhead of cloning, but cloning is cheap since it only contains [`TypedFunc`] handles which are internally reference-counted.
    pub fn exports(&self) -> Exports {
        self.exports.clone()
    }

    /// Calls a typed WASM function, translating Wasmtime errors into
    /// [`PhysicsError`] variants.
    ///
    /// Checks for module exit both before and after the call.  This handles
    /// the case where an import (e.g. `abort`) records an exit and then the
    /// Wasmtime trap unwinds out of the guest — without the post-call check we
    /// would return `Err(Wasm(...))` instead of the more informative
    /// `Err(WasmExited(...))`.
    pub fn call<T, R>(&mut self, f: &TypedFunc<T, R>, args: T) -> Result<R, PhysicsError>
    where
        T: WasmParams,
        R: WasmResults,
    {
        self.check_exited()?;
        let result = match f.call(&mut self.store, args) {
            Ok(v) => v,
            Err(e) => {
                // Wasmtime surfaces fuel exhaustion as a generic error message
                // rather than a dedicated type, so we match on the string.
                if e.to_string().contains("all fuel consumed") {
                    return Err(PhysicsError::OutOfFuel);
                }
                return Err(e.into());
            }
        };
        self.check_exited()?;
        Ok(result)
    }

    /// Returns `true` if the module has called `exit()`, `abort()`, or hit an
    /// assertion failure.
    ///
    /// After this returns `true`, every method that executes WASM will return
    /// [`PhysicsError::WasmExited`].
    pub fn has_exited(&self) -> bool {
        self.store.data().exited
    }

    /// Returns the exit code recorded when the module terminated, or `None` if
    /// it is still running.
    pub fn exit_code(&self) -> Option<i32> {
        self.store.data().check_exit()
    }

    /// Replenishes the fuel tank to `fuel` units, allowing execution to
    /// continue after a [`PhysicsError::OutOfFuel`] error.
    ///
    /// Use this only when partial progress is acceptable.  If the simulation
    /// must be deterministic, hitting the fuel limit likely indicates a bug —
    /// reset the instance entirely rather than re-fueling.
    pub fn reset_fuel(&mut self, fuel: u64) -> Result<(), PhysicsError> {
        self.store.set_fuel(fuel)?;
        Ok(())
    }
}

impl Drop for PolyTrackPhysics {
    /// Frees the pre-allocated scratch buffer back into the WASM heap.
    ///
    /// Skipped when the module has already exited — in that case the heap is
    /// effectively gone and calling `free` would either be a no-op or cause a
    /// confusing secondary trap.
    fn drop(&mut self) {
        if !self.has_exited() {
            let _ = self
                .exports
                .free
                .call(&mut self.store, self.scratch_ptr.get());
        }
    }
}

/// Reads a null-terminated C string from WASM linear memory starting at `ptr`.
///
/// Used by the `__assert_fail` host import to turn WASM string pointers into
/// Rust `String`s for logging.  Returns a human-readable placeholder if the
/// memory export is missing or the pointer is out of range.
fn read_cstr(caller: &mut Caller<'_, HostState>, ptr: u32) -> String {
    let mem = match caller.get_export("j") {
        Some(Extern::Memory(m)) => m,
        _ => return "<no memory>".into(),
    };

    let data = mem.data(caller);
    let start = ptr as usize;

    if start >= data.len() {
        return "<invalid ptr>".into();
    }

    // Scan forward for the null terminator; if none is found (truncated heap?)
    // treat the rest of the heap as the string, which will at least give us
    // something useful in the error log.
    let end = data[start..]
        .iter()
        .position(|&b| b == 0)
        .map(|i| start + i)
        .unwrap_or(data.len());

    String::from_utf8_lossy(&data[start..end]).into_owned()
}

/// Records that the WASM module has terminated with `code`.
///
/// Idempotent — if the module has already exited, the exit code is
/// overwritten, but in practice only one exit path fires per execution.
fn mark_exited(caller: &mut Caller<'_, HostState>, code: i32) {
    let st = caller.data_mut();
    st.exited = true;
    st.exit_code = code;
}
