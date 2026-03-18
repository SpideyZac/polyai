use std::{collections::HashSet, sync::Arc, time::Instant};

use bytemuck::{Pod, Zeroable};
use parry3d::glamx::{Mat4, Vec3};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use crate::simulation::World;

pub struct ExplorerConfig {
    pub title: &'static str,
    pub width: u32,
    pub height: u32,
    pub start_pos: [f32; 3],
    pub move_speed: f32,
    pub look_speed: f32,
    pub sprint_mult: f32,
    pub max_ray_dist: f32,
}

impl Default for ExplorerConfig {
    fn default() -> Self {
        Self {
            title: "World Explorer - Mesh & Raycast Debug",
            width: 1280,
            height: 720,
            start_pos: [0.0, 15.0, 0.0],
            move_speed: 25.0,
            look_speed: 1.6,
            sprint_mult: 4.0,
            max_ray_dist: 500.0,
        }
    }
}

pub fn run_explorer(world: &World, cfg: ExplorerConfig) {
    let mesh_data = extract_mesh_data(world);
    pollster::block_on(run_async(mesh_data, world, cfg));
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        use std::mem::size_of;
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 3]>() as u64,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

struct MeshData {
    world_verts: Vec<Vertex>,
}

fn extract_mesh_data(world: &World) -> MeshData {
    let mut world_verts: Vec<Vertex> = Vec::new();

    for inst in &world.mesh_instances {
        let Some(asset) = world.assets.get(&inst.id) else {
            continue;
        };

        let color = id_to_color(inst.id);
        let transform = Mat4::from_rotation_translation(inst.quaternion, inst.position);

        for tri in asset.mesh.indices() {
            for &vi in tri.iter() {
                let p = asset.mesh.vertices()[vi as usize];
                let local = Vec3::new(p.x, p.y, p.z);
                let world_pt = transform.transform_point3(local);
                world_verts.push(Vertex {
                    position: world_pt.into(),
                    color,
                });
            }
        }
    }

    eprintln!(
        "[explorer] extracted {} triangles from {} instances",
        world_verts.len() / 3,
        world.mesh_instances.len()
    );

    MeshData { world_verts }
}

/// Deterministic per-id hue via multiplicative hash → HSV → RGB.
fn id_to_color(id: u32) -> [f32; 3] {
    let h = (id.wrapping_mul(2_654_435_761) >> 8) as f32 / 16_777_216.0;
    hsv_to_rgb(h, 0.65, 0.88)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0) as u32 % 6;
    let f = h * 6.0 - (h * 6.0).floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

struct Camera {
    pos: Vec3,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        )
        .normalize()
    }

    fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_to_rh(self.pos, self.forward(), Vec3::Y);
        let proj = Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_2, // 90° fov
            aspect,
            0.1,
            8000.0,
        );
        proj * view
    }
}

const SHADER: &str = r#"
struct Uni { view_proj: mat4x4<f32> }
@group(0) @binding(0) var<uniform> uni: Uni;

struct VIn  { @location(0) pos: vec3<f32>, @location(1) col: vec3<f32> }
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) col: vec3<f32> }

@vertex fn vs(v: VIn) -> VOut {
    return VOut(uni.view_proj * vec4<f32>(v.pos, 1.0), v.col);
}
@fragment fn fs(v: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(v.col, 1.0);
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
}

struct GpuBuf {
    buf: wgpu::Buffer,
    count: u32,
}

struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline_tri: wgpu::RenderPipeline,
    pipeline_line: wgpu::RenderPipeline,
    uni_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth_view: wgpu::TextureView,
    world_buf: GpuBuf,
    window: Arc<Window>,
}

impl Gpu {
    async fn new(window: Arc<Window>, verts: &[Vertex]) -> Self {
        let size = window.inner_size();
        let inst = wgpu::Instance::default();
        let surface = inst.create_surface(Arc::clone(&window)).unwrap();

        let adapter = inst
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No GPU adapter found");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::POLYGON_MODE_LINE,
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .expect("Failed to open GPU device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let uni_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uni_buf.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let depth_fmt = wgpu::TextureFormat::Depth32Float;
        let pipeline_tri = make_pipeline(
            &device,
            &pl_layout,
            &shader,
            format,
            depth_fmt,
            wgpu::PrimitiveTopology::TriangleList,
            wgpu::PolygonMode::Fill,
        );
        let pipeline_line = make_pipeline(
            &device,
            &pl_layout,
            &shader,
            format,
            depth_fmt,
            wgpu::PrimitiveTopology::LineList,
            wgpu::PolygonMode::Fill,
        );

        let depth_view = make_depth(&device, config.width, config.height, depth_fmt);
        let world_buf = upload(&device, verts);

        Self {
            surface,
            device,
            queue,
            config,
            pipeline_tri,
            pipeline_line,
            uni_buf,
            bind_group,
            depth_view,
            world_buf,
            window,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = make_depth(&self.device, w, h, wgpu::TextureFormat::Depth32Float);
    }

    fn draw(&mut self, cam: &Camera, ray: Option<&GpuBuf>, wireframe: bool) {
        let aspect = self.config.width as f32 / self.config.height as f32;
        let uni = Uniforms {
            view_proj: cam.view_proj(aspect).to_cols_array_2d(),
        };
        self.queue
            .write_buffer(&self.uni_buf, 0, bytemuck::bytes_of(&uni));

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            Err(e) => {
                eprintln!("[explorer] surface error: {e}");
                return;
            }
        };

        let view = frame.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());

        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.07,
                            b: 0.12,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // World geometry — fill or wireframe
            rp.set_pipeline(if wireframe {
                &self.pipeline_line
            } else {
                &self.pipeline_tri
            });
            rp.set_bind_group(0, &self.bind_group, &[]);
            rp.set_vertex_buffer(0, self.world_buf.buf.slice(..));
            rp.draw(0..self.world_buf.count, 0..1);

            // Debug ray line
            if let Some(r) = ray {
                rp.set_pipeline(&self.pipeline_line);
                rp.set_bind_group(0, &self.bind_group, &[]);
                rp.set_vertex_buffer(0, r.buf.slice(..));
                rp.draw(0..r.count, 0..1);
            }
        }

        self.queue.submit(std::iter::once(enc.finish()));
        frame.present();
        self.window.request_redraw();
    }
}

struct App<'w> {
    world: &'w World,
    cfg: ExplorerConfig,
    verts: Vec<Vertex>,
    gpu: Option<Gpu>,
    keys: HashSet<KeyCode>,
    cam: Camera,
    last_tick: Instant,
    wireframe: bool,
    ray_buf: Option<GpuBuf>,
}

impl<'w> ApplicationHandler for App<'w> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }

        let win = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title(self.cfg.title)
                        .with_inner_size(winit::dpi::LogicalSize::new(
                            self.cfg.width,
                            self.cfg.height,
                        )),
                )
                .expect("create window"),
        );

        let gpu = pollster::block_on(Gpu::new(Arc::clone(&win), &self.verts));
        self.gpu = Some(gpu);

        eprintln!(
            "[explorer] ready — WASD/QE move, arrows look, \
             Space raycast, F wireframe, R reset, Esc quit"
        );
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(s) => {
                if let Some(g) = &mut self.gpu {
                    g.resize(s.width, s.height);
                }
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => {
                match state {
                    ElementState::Pressed => {
                        self.keys.insert(key);
                    }
                    ElementState::Released => {
                        self.keys.remove(&key);
                    }
                }
                if state == ElementState::Pressed {
                    match key {
                        KeyCode::Escape => event_loop.exit(),
                        KeyCode::KeyF => {
                            self.wireframe = !self.wireframe;
                            eprintln!("[explorer] wireframe={}", self.wireframe);
                        }
                        KeyCode::KeyR => {
                            self.cam = Camera {
                                pos: Vec3::from_array(self.cfg.start_pos),
                                yaw: 0.0,
                                pitch: 0.0,
                            };
                            eprintln!("[explorer] camera reset");
                        }
                        KeyCode::Space => self.do_raycast(),
                        _ => {}
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                self.tick();
                if let Some(g) = &mut self.gpu {
                    g.draw(&self.cam, self.ray_buf.as_ref(), self.wireframe);
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(g) = &self.gpu {
            g.window.request_redraw();
        }
    }
}

impl<'w> App<'w> {
    fn tick(&mut self) {
        let dt = self.last_tick.elapsed().as_secs_f32().min(0.1);
        self.last_tick = Instant::now();

        let sprint = if self.keys.contains(&KeyCode::ShiftLeft)
            || self.keys.contains(&KeyCode::ShiftRight)
        {
            self.cfg.sprint_mult
        } else {
            1.0
        };

        let ms = self.cfg.move_speed * sprint * dt;
        let ls = self.cfg.look_speed * dt;

        let fwd = self.cam.forward();
        let right = self.cam.right();

        if self.keys.contains(&KeyCode::KeyW) {
            self.cam.pos += fwd * ms;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            self.cam.pos -= fwd * ms;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            self.cam.pos += right * ms;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            self.cam.pos -= right * ms;
        }
        if self.keys.contains(&KeyCode::KeyE) {
            self.cam.pos += Vec3::Y * ms;
        }
        if self.keys.contains(&KeyCode::KeyQ) {
            self.cam.pos -= Vec3::Y * ms;
        }

        if self.keys.contains(&KeyCode::ArrowLeft) {
            self.cam.yaw -= ls;
        }
        if self.keys.contains(&KeyCode::ArrowRight) {
            self.cam.yaw += ls;
        }
        if self.keys.contains(&KeyCode::ArrowUp) {
            self.cam.pitch = (self.cam.pitch + ls).min(1.55);
        }
        if self.keys.contains(&KeyCode::ArrowDown) {
            self.cam.pitch = (self.cam.pitch - ls).max(-1.55);
        }
    }

    fn do_raycast(&mut self) {
        let origin = self.cam.pos;
        let dir = self.cam.forward();
        let max_d = self.cfg.max_ray_dist;

        let toi = match self.world.raycast(origin.into(), dir.into(), max_d) {
            Some((id, toi)) => {
                let hit = origin + dir * toi;
                eprintln!(
                    "[raycast] HIT  asset_id={id}  toi={toi:.4}  \
                    hit=({:.3},{:.3},{:.3})",
                    hit.x, hit.y, hit.z
                );
                Some(toi)
            }
            None => {
                eprintln!(
                    "[raycast] MISS  origin=({:.2},{:.2},{:.2})  \
                    dir=({:.3},{:.3},{:.3})",
                    origin.x, origin.y, origin.z, dir.x, dir.y, dir.z,
                );
                None
            }
        };

        if let Some(g) = &self.gpu {
            let end = origin + dir * toi.unwrap_or(max_d);
            let color: [f32; 3] = if toi.is_some() {
                [0.1, 1.0, 0.4]
            } else {
                [1.0, 0.4, 0.1]
            };
            let verts = vec![
                Vertex {
                    position: origin.into(),
                    color,
                },
                Vertex {
                    position: end.into(),
                    color,
                },
            ];
            self.ray_buf = Some(upload(&g.device, &verts));
        }
    }
}

async fn run_async(mesh_data: MeshData, world: &World, cfg: ExplorerConfig) {
    #[cfg(target_os = "windows")]
    use winit::platform::windows::EventLoopBuilderExtWindows;

    let event_loop = {
        #[cfg(target_os = "windows")]
        {
            EventLoop::builder().with_any_thread(true).build()
        }
        #[cfg(not(target_os = "windows"))]
        {
            EventLoop::builder().build()
        }
    }
    .expect("Failed to create event loop");

    let start_pos = cfg.start_pos;

    let mut app = App {
        world,
        cfg,
        verts: mesh_data.world_verts,
        gpu: None,
        keys: HashSet::new(),
        cam: Camera {
            pos: Vec3::from_array(start_pos),
            yaw: 0.0,
            pitch: 0.0,
        },
        last_tick: Instant::now(),
        wireframe: false,
        ray_buf: None,
    };

    event_loop.run_app(&mut app).expect("event loop error");
}

fn upload(device: &wgpu::Device, verts: &[Vertex]) -> GpuBuf {
    use wgpu::util::DeviceExt;
    GpuBuf {
        buf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vbuf"),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        }),
        count: verts.len() as u32,
    }
}

fn make_depth(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    fmt: wgpu::TextureFormat,
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn make_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    surf_fmt: wgpu::TextureFormat,
    dep_fmt: wgpu::TextureFormat,
    topology: wgpu::PrimitiveTopology,
    poly: wgpu::PolygonMode,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs"),
            buffers: &[Vertex::layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surf_fmt,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology,
            polygon_mode: poly,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: dep_fmt,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}
