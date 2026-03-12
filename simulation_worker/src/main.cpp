#include <wasmtime.h>
#include <wasm.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>

// ── Globals mirroring the JS state ────────────────────────────────────────────
static wasmtime_memory_t g_memory;   // the exported "j" (heap memory)
static wasmtime_store_t* g_store = nullptr;

// Helper: read a null-terminated UTF-8 string from wasm memory
std::string read_wasm_str(const uint8_t* mem_data, uint32_t ptr) {
    if (ptr == 0) return "";
    return std::string(reinterpret_cast<const char*>(mem_data + ptr));
}

// ── Import implementations ─────────────────────────────────────────────────────
// Y.i → assertion failed handler
// signature: (i32 msg, i32 filename, i32 line, i32 funcname) -> void
static wasm_trap_t* import_assert_fail(
    void* env, wasmtime_caller_t* caller,
    const wasmtime_val_t* args, size_t nargs,
    wasmtime_val_t* results, size_t nresults)
{
    wasmtime_context_t* ctx = wasmtime_caller_context(caller);
    uint8_t* mem = wasmtime_memory_data(ctx, &g_memory);

    std::string msg      = read_wasm_str(mem, args[0].of.i32);
    std::string filename = args[1].of.i32 ? read_wasm_str(mem, args[1].of.i32) : "unknown";
    int32_t     line     = args[2].of.i32;
    std::string func     = args[3].of.i32 ? read_wasm_str(mem, args[3].of.i32) : "unknown";

    fprintf(stderr, "Assertion failed: %s, at: %s:%d (%s)\n",
            msg.c_str(), filename.c_str(), line, func.c_str());

    // Return a trap to unwind wasm
    return wasmtime_trap_new("assertion failed", 16);
}

// Y.a → throw exception: (i32 excPtr, i32 type, i32 destructor) -> void
static wasm_trap_t* import_throw(
    void* env, wasmtime_caller_t* caller,
    const wasmtime_val_t* args, size_t nargs,
    wasmtime_val_t* results, size_t nresults)
{
    fprintf(stderr, "Wasm threw exception at ptr=%d type=%d\n",
            args[0].of.i32, args[1].of.i32);
    return wasmtime_trap_new("wasm exception", 14);
}

// Y.e → abort: () -> void
static wasm_trap_t* import_abort(
    void* env, wasmtime_caller_t* caller,
    const wasmtime_val_t* args, size_t nargs,
    wasmtime_val_t* results, size_t nresults)
{
    fprintf(stderr, "Wasm called abort()\n");
    return wasmtime_trap_new("abort", 5);
}

// Y.c → set noExitRuntime / reset keepalive: () -> void
static wasm_trap_t* import_no_exit_runtime(
    void* env, wasmtime_caller_t* caller,
    const wasmtime_val_t* args, size_t nargs,
    wasmtime_val_t* results, size_t nresults)
{
    // JS does: E = false; H = 0  (runtime keepalive counter reset)
    // Nothing meaningful to do in a native host
    return nullptr;
}

// Y.d → setTimeout equivalent: (i32 id, f64 timeout_ms) -> i32
static wasm_trap_t* import_set_timeout(
    void* env, wasmtime_caller_t* caller,
    const wasmtime_val_t* args, size_t nargs,
    wasmtime_val_t* results, size_t nresults)
{
    // In native you'd use a timer thread or event loop.
    // For a synchronous/offline physics engine, you can stub this out:
    results[0].kind    = WASMTIME_I32;
    results[0].of.i32  = 0;
    return nullptr;
}

// Y.h → Date.now(): () -> f64
static wasm_trap_t* import_date_now(
    void* env, wasmtime_caller_t* caller,
    const wasmtime_val_t* args, size_t nargs,
    wasmtime_val_t* results, size_t nresults)
{
    using namespace std::chrono;

    auto now = system_clock::now();
    auto duration = now.time_since_epoch();

    double ms = duration_cast<milliseconds>(duration).count();

    results[0].kind   = WASMTIME_F64;
    results[0].of.f64 = ms;
    return nullptr;
}

// Y.f → memory grow request: (i32 requested_bytes) -> i32
static wasm_trap_t* import_grow_memory(
    void* env, wasmtime_caller_t* caller,
    const wasmtime_val_t* args, size_t nargs,
    wasmtime_val_t* results, size_t nresults)
{
    wasmtime_context_t* ctx = wasmtime_caller_context(caller);
    uint32_t requested = (uint32_t)args[0].of.i32;

    // Calculate pages needed (matches JS logic: ceil to 64KB pages)
    uint64_t cur_bytes = wasmtime_memory_data_size(ctx, &g_memory);
    uint32_t pages_needed = (requested + 65535) / 65536;
    uint64_t new_size = 0;

    wasmtime_error_t* err = wasmtime_memory_grow(ctx, &g_memory, pages_needed, &new_size);
    if (err) {
        wasmtime_error_delete(err);
        results[0].kind   = WASMTIME_I32;
        results[0].of.i32 = 0;  // false
    } else {
        results[0].kind   = WASMTIME_I32;
        results[0].of.i32 = 1;  // true
    }
    return nullptr;
}

// Y.g → fd_write / console output: (i32 fd, i32 iov, i32 iovcnt, i32 pnum) -> i32
// This is a WASI-style writev. JS routes fd=1 to console.log, fd=2 to console.error
static wasm_trap_t* import_fd_write(
    void* env, wasmtime_caller_t* caller,
    const wasmtime_val_t* args, size_t nargs,
    wasmtime_val_t* results, size_t nresults)
{
    wasmtime_context_t* ctx = wasmtime_caller_context(caller);
    uint8_t*  mem    = wasmtime_memory_data(ctx, &g_memory);
    uint32_t* mem32  = reinterpret_cast<uint32_t*>(mem);

    int32_t  fd     = args[0].of.i32;
    uint32_t iov    = args[1].of.i32;
    uint32_t iovcnt = args[2].of.i32;
    uint32_t pnum   = args[3].of.i32;

    uint32_t written = 0;
    FILE* out = (fd == 1) ? stdout : stderr;

    for (uint32_t i = 0; i < iovcnt; i++) {
        uint32_t buf_ptr = mem32[(iov + i * 8)     / 4];
        uint32_t buf_len = mem32[(iov + i * 8 + 4) / 4];
        fwrite(mem + buf_ptr, 1, buf_len, out);
        written += buf_len;
    }

    mem32[pnum / 4] = written;
    results[0].kind   = WASMTIME_I32;
    results[0].of.i32 = 0;  // success
    return nullptr;
}

// Y.b → exit(code): (i32) -> void
static wasm_trap_t* import_exit(
    void* env, wasmtime_caller_t* caller,
    const wasmtime_val_t* args, size_t nargs,
    wasmtime_val_t* results, size_t nresults)
{
    int code = args[0].of.i32;
    fprintf(stderr, "Wasm exit(%d)\n", code);
    return wasmtime_trap_new("exit", 4);
}

// ── Helper: register one import function ──────────────────────────────────────
static wasmtime_func_t make_import(
    wasmtime_store_t* store,
    wasmtime_func_callback_t cb,
    const wasm_valtype_vec_t* params,
    const wasm_valtype_vec_t* results)
{
    wasm_functype_t* ft = wasm_functype_new(
        const_cast<wasm_valtype_vec_t*>(params),
        const_cast<wasm_valtype_vec_t*>(results));
    wasmtime_func_t func;
    wasmtime_func_new(wasmtime_store_context(store), ft, cb, nullptr, nullptr, &func);
    wasm_functype_delete(ft);
    return func;
}

// ── Main ───────────────────────────────────────────────────────────────────────
int main() {
    wasm_engine_t*   engine = wasm_engine_new();
    g_store                 = wasmtime_store_new(engine, nullptr, nullptr);
    wasmtime_context_t* ctx = wasmtime_store_context(g_store);

    // 1. Load the .wasm file
    FILE* f = fopen("polytrack_physics.wasm", "rb");
    fseek(f, 0, SEEK_END);
    size_t wasm_size = ftell(f);
    rewind(f);
    std::vector<uint8_t> wasm_bytes(wasm_size);
    fread(wasm_bytes.data(), 1, wasm_size, f);
    fclose(f);

    // 2. Compile
    wasmtime_module_t* module = nullptr;
    wasmtime_error_t* err = wasmtime_module_new(engine, wasm_bytes.data(), wasm_size, &module);
    if (err) { fprintf(stderr, "compile failed\n"); return 1; }

    // 3. Build the import list matching { a: { i,a,e,c,d,h,f,g,b } }
    //    Order must match the module's import section exactly.
    //    Check with: wasm-objdump -x polytrack_physics.wasm | grep import
    //
    //    Expected imports (from the JS Y object, namespace "a"):
    //      a.i  (i32,i32,i32,i32)->()     assert_fail
    //      a.a  (i32,i32,i32)->()          throw
    //      a.e  ()->()                     abort
    //      a.c  ()->()                     no_exit_runtime
    //      a.d  (i32,f64)->i32             set_timeout
    //      a.h  ()->f64                    date_now
    //      a.f  (i32)->i32                 grow_memory
    //      a.g  (i32,i32,i32,i32)->i32     fd_write
    //      a.b  (i32)->()                  exit

    // Build type helpers
    auto make_type = [](std::vector<wasm_valkind_t> ps, std::vector<wasm_valkind_t> rs) {
        wasm_valtype_vec_t params, results;
        wasm_valtype_vec_new_uninitialized(&params, ps.size());
        for (size_t i = 0; i < ps.size(); i++) params.data[i] = wasm_valtype_new(ps[i]);
        wasm_valtype_vec_new_uninitialized(&results, rs.size());
        for (size_t i = 0; i < rs.size(); i++) results.data[i] = wasm_valtype_new(rs[i]);
        wasm_functype_t* ft = wasm_functype_new(&params, &results);
        return ft;
    };

    struct FuncDef {
        const char* module_name;
        const char* field_name;
        wasmtime_func_callback_t callback;
        std::vector<wasm_valkind_t> params;
        std::vector<wasm_valkind_t> results;
    };

    std::vector<FuncDef> import_defs = {
        {"a","i", import_assert_fail,    {WASM_I32,WASM_I32,WASM_I32,WASM_I32}, {}},
        {"a","a", import_throw,          {WASM_I32,WASM_I32,WASM_I32},          {}},
        {"a","e", import_abort,          {},                                     {}},
        {"a","c", import_no_exit_runtime,{},                                     {}},
        {"a","d", import_set_timeout,    {WASM_I32,WASM_F64},                   {WASM_I32}},
        {"a","h", import_date_now,       {},                                     {WASM_F64}},
        {"a","f", import_grow_memory,    {WASM_I32},                            {WASM_I32}},
        {"a","g", import_fd_write,       {WASM_I32,WASM_I32,WASM_I32,WASM_I32},{WASM_I32}},
        {"a","b", import_exit,           {WASM_I32},                            {}},
    };

    std::vector<wasmtime_extern_t> imports;
    for (auto& def : import_defs) {
        wasm_functype_t* ft = make_type(def.params, def.results);
        wasmtime_func_t func;
        wasmtime_func_new(ctx, ft, def.callback, nullptr, nullptr, &func);
        wasm_functype_delete(ft);
        wasmtime_extern_t ext;
        ext.kind     = WASMTIME_EXTERN_FUNC;
        ext.of.func  = func;
        imports.push_back(ext);
    }

    // 4. Instantiate
    wasmtime_instance_t instance;
    wasm_trap_t* trap = nullptr;
    err = wasmtime_instance_new(ctx, module, imports.data(), imports.size(), &instance, &trap);
    if (err || trap) { fprintf(stderr, "instantiation failed\n"); return 1; }

    // 5. Grab the memory export ("j" in JS, typically named "memory" in the wasm)
    wasmtime_extern_t mem_ext;
    if (wasmtime_instance_export_get(ctx, &instance, "memory", 6, &mem_ext))
        g_memory = mem_ext.of.memory;

    // 6. Call the runtime init — equivalent to V.k() in JS
    wasmtime_extern_t init_ext;
    if (wasmtime_instance_export_get(ctx, &instance, "k", 1, &init_ext)) {
        wasmtime_func_call(ctx, &init_ext.of.func, nullptr, 0, nullptr, 0, &trap);
    }

    // 7. Now grab and call the actual physics exports, e.g.:
    //    _initializeCarCollisionShape  → "n"
    //    _createCarModel               → "p"
    //    _updateCarModel               → "r"

    wasmtime_extern_t update_ext;
    if (wasmtime_instance_export_get(ctx, &instance, "r", 1, &update_ext)) {
        // Call _updateCarModel with your args...
        wasmtime_val_t args[1];
        args[0].kind   = WASMTIME_I32;
        args[0].of.i32 = 0; // example: model handle
        wasmtime_func_call(ctx, &update_ext.of.func, args, 1, nullptr, 0, &trap);
    }

    // Cleanup
    wasmtime_module_delete(module);
    wasmtime_store_delete(g_store);
    wasm_engine_delete(engine);
    return 0;
}
