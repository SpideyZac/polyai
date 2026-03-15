use wasmtime::{
    Caller, Config, Engine, Extern, Linker, Memory, Module, OptLevel, Store, TypedFunc, WasmParams,
    WasmResults, bail, format_err,
};

const EXIT_CODE_ABORT: i32 = 134;

pub fn create_engine() -> Engine {
    let mut config = Config::new();

    config.cranelift_opt_level(OptLevel::Speed);
    config.wasm_simd(true);

    Engine::new(&config).unwrap()
}

#[derive(Debug, thiserror::Error)]
pub enum PhysicsError {
    #[error("wasm module exited with code {0}")]
    WasmExited(i32),

    #[error("out-of-bounds wasm memory access: offset {offset} + len {len} > heap size {heap}")]
    OutOfBounds {
        offset: usize,
        len: usize,
        heap: usize,
    },

    #[error("wasm error: {0}")]
    Wasm(#[from] wasmtime::Error),
}

#[derive(Default)]
struct HostState {
    exited: bool,
    exit_code: i32,
}

impl HostState {
    fn check_exit(&self) -> Option<i32> {
        self.exited.then_some(self.exit_code)
    }
}

pub type InitCarCollisionShapeArgs = (f32, i32, i32);

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

pub type UpdateCarModelArgs = (i32, i32, i32, i32, i32, i32, i32);

#[derive(Clone)]
pub struct Exports {
    pub malloc: TypedFunc<i32, i32>,
    pub free: TypedFunc<i32, ()>,
    pub init_car_collision_shape: TypedFunc<InitCarCollisionShapeArgs, ()>,
    pub add_track_part_config: TypedFunc<AddTrackPartConfigArgs, ()>,
    pub create_car_model: TypedFunc<CreateCarModelArgs, ()>,
    pub delete_car_model: TypedFunc<i32, ()>,
    pub update_car_model: TypedFunc<UpdateCarModelArgs, ()>,
    pub test_determinism: TypedFunc<(), i32>,
}

pub struct PolyTrackPhysics {
    store: Store<HostState>,
    memory: Memory,
    exports: Exports,
}

impl PolyTrackPhysics {
    pub fn from_file(engine: &Engine, wasm_path: &str) -> Result<(Self, Module), PhysicsError> {
        let module = Module::from_file(engine, wasm_path)?;
        Ok((Self::from_module(engine, &module)?, module))
    }

    pub fn from_module(engine: &Engine, module: &Module) -> Result<Self, PhysicsError> {
        let state = HostState {
            exited: false,
            exit_code: 0,
        };
        let mut store = Store::new(engine, state);
        let mut linker: Linker<HostState> = Linker::new(engine);

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
                mark_exited(&mut caller, EXIT_CODE_ABORT);
                bail!("abort()")
            },
        )?;

        linker.func_wrap("a", "c", |_caller: Caller<'_, HostState>| {})?;

        linker.func_wrap(
            "a",
            "d",
            |_caller: Caller<'_, HostState>, _id: i32, _ms: f64| -> i32 { 0 },
        )?;

        linker.func_wrap("a", "h", |_caller: Caller<'_, HostState>| -> f64 { 0.0 })?;

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
                    return 1;
                }

                let grow = desired - current;
                let pages = grow.div_ceil(65536) as u64;

                match mem.grow(&mut caller, pages) {
                    Ok(_) => 1,
                    Err(_) => 0,
                }
            },
        )?;

        linker.func_wrap(
            "a",
            "g",
            |mut _caller: Caller<'_, HostState>, _fd: i32, _iov: i32, _iovcnt: i32, _pnum: i32| 1,
        )?;

        linker.func_wrap("a", "b", |_caller: Caller<'_, HostState>, _code: i32| {})?;

        let instance = linker.instantiate(&mut store, module)?;

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
            create_car_model: instance.get_typed_func::<CreateCarModelArgs, ()>(&mut store, "p")?,
            delete_car_model: instance.get_typed_func::<i32, ()>(&mut store, "q")?,
            update_car_model: instance.get_typed_func::<UpdateCarModelArgs, ()>(&mut store, "r")?,
            test_determinism: instance.get_typed_func::<(), i32>(&mut store, "s")?,
        };

        let memory = instance
            .get_memory(&mut store, "j")
            .ok_or_else(|| format_err!("wasm memory export missing"))?;

        Ok(Self {
            store,
            exports,
            memory,
        })
    }

    fn check_exited(&self) -> Result<(), PhysicsError> {
        if let Some(code) = self.store.data().check_exit() {
            return Err(PhysicsError::WasmExited(code));
        }
        Ok(())
    }

    pub fn alloc_bytes(&mut self, data: &[u8]) -> Result<i32, PhysicsError> {
        self.check_exited()?;

        let ptr = self
            .exports
            .malloc
            .call(&mut self.store, data.len() as i32)?;
        if ptr == 0 {
            return Err(format_err!("wasm malloc failed").into());
        }

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
        Ok(ptr)
    }

    pub fn free_wasm(&mut self, buffer: i32) -> Result<(), PhysicsError> {
        self.check_exited()?;
        if buffer != 0 {
            self.exports.free.call(&mut self.store, buffer)?;
        }

        self.check_exited()
    }

    pub fn wasm_slice(&self, ptr: i32, len: usize) -> Result<&[u8], PhysicsError> {
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

    pub fn exports(&self) -> Exports {
        self.exports.clone()
    }

    pub fn call<T, R>(&mut self, f: &TypedFunc<T, R>, args: T) -> Result<R, PhysicsError>
    where
        T: WasmParams,
        R: WasmResults,
    {
        self.check_exited()?;
        let result = f.call(&mut self.store, args)?;
        self.check_exited()?;
        Ok(result)
    }
}

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

    let end = data[start..]
        .iter()
        .position(|&b| b == 0)
        .map(|i| start + i)
        .unwrap_or(data.len());

    String::from_utf8_lossy(&data[start..end]).into_owned()
}

fn mark_exited(caller: &mut Caller<'_, HostState>, code: i32) {
    let st = caller.data_mut();
    st.exited = true;
    st.exit_code = code;
}
