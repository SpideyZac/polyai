use std::{
    collections::HashMap,
    f32::consts::{PI, SQRT_2},
};

use anyhow::{Context, anyhow, bail};
use bytemuck::cast_slice;
use parry3d::{
    bounding_volume::Aabb,
    glamx::{EulerRot, Quat},
    math::{Pose, Vec2, Vec3},
    partitioning::{Bvh, BvhBuildStrategy},
    query::{Ray, RayCast},
    shape::TriMesh,
};
use polytrack_codes::v6::{Block, Direction, TrackInfo, decode_track_code, decode_track_data};

use crate::{
    data_reader::{Detector, assets},
    physics_worker::{Exports, PolyTrackPhysics},
};

const PART_SIZE: f32 = 5.0;
const HEIGHT_SCALE: f32 = 100.0;
const MOUNTAIN_RING_STEP: f32 = 100.0;
const MOUNTAIN_RING_SEGMENTS: usize = 8;
const MOUNTAIN_MIN_RADIUS: f32 = 200.0;
const MOUNTAIN_RADIUS_BASE: f32 = 160.0;
const MOUNTAIN_MAX_RADIUS: f32 = 4500.0;
const FINISH_LINE_IDS: [u8; 4] = [74, 6, 78, 76];

#[allow(clippy::excessive_precision, clippy::approx_constant)]
const FACE_ROTATION_QUATS: [[Quat; 4]; 6] = [
    [
        Quat::from_xyzw(0.0, 0.0, 0.0, 1.0),
        Quat::from_xyzw(0.0, 0.7071067811865475, 0.0, 0.7071067811865476),
        Quat::from_xyzw(0.0, 1.0, 0.0, 0.0),
        Quat::from_xyzw(0.0, 0.7071067811865476, 0.0, -0.7071067811865475),
    ],
    [
        Quat::from_xyzw(0.0, 0.0, 1.0, 0.0),
        Quat::from_xyzw(0.7071067811865475, 0.0, 0.7071067811865476, 0.0),
        Quat::from_xyzw(1.0, 0.0, 0.0, 0.0),
        Quat::from_xyzw(0.7071067811865476, 0.0, -0.7071067811865475, 0.0),
    ],
    [
        Quat::from_xyzw(0.0, 0.0, -0.7071067811865477, 0.7071067811865475),
        Quat::from_xyzw(0.5, 0.5, -0.5, 0.5),
        Quat::from_xyzw(0.7071067811865475, 0.7071067811865477, 0.0, 0.0),
        Quat::from_xyzw(0.5, 0.5, 0.5, -0.5),
    ],
    [
        Quat::from_xyzw(0.0, 0.0, 0.7071067811865475, 0.7071067811865476),
        Quat::from_xyzw(0.5, -0.5, 0.5, 0.5),
        Quat::from_xyzw(0.7071067811865476, -0.7071067811865475, 0.0, 0.0),
        Quat::from_xyzw(0.5, -0.5, -0.5, -0.5),
    ],
    [
        Quat::from_xyzw(0.7071067811865475, 0.0, 0.0, 0.7071067811865476),
        Quat::from_xyzw(0.5, 0.5, 0.5, 0.5),
        Quat::from_xyzw(0.0, 0.7071067811865476, 0.7071067811865475, 0.0),
        Quat::from_xyzw(-0.5, 0.5, 0.5, -0.5),
    ],
    [
        Quat::from_xyzw(-0.7071067811865477, 0.0, 0.0, 0.7071067811865475),
        Quat::from_xyzw(-0.5, -0.5, 0.5, 0.5),
        Quat::from_xyzw(0.0, -0.7071067811865475, 0.7071067811865477, 0.0),
        Quat::from_xyzw(0.5, -0.5, 0.5, -0.5),
    ],
];

#[allow(clippy::excessive_precision)]
const RNG_TABLE: &[f32] = &[
    0.12047764760664692,
    0.19645762332790628,
    0.5525629082262744,
    0.41272626379209965,
    0.7795036003541387,
    0.13367266027110114,
    0.7999601557377349,
    0.9519714253374205,
    0.1735048382917752,
    0.7513367084489158,
    0.6531386724839523,
    0.9026427867068505,
    0.8543272738216994,
    0.11176849958868162,
    0.6705698284858437,
    0.26628732081296946,
    0.31140322993719605,
    0.45170300835470933,
    0.12615515120247944,
    0.0610638094525735,
    0.291990923385425,
    0.4613983868623317,
    0.6615759832726253,
    0.4373182881232056,
    0.7432890501246443,
    0.39316710322388837,
    0.49444122821563297,
    0.5994296685114344,
    0.060050119050233386,
    0.4165885432422003,
    0.43974364800990084,
    0.1628314496954224,
    0.05787972729968116,
    0.225388541259955,
    0.6075775236386991,
    0.8908354370882479,
    0.47072983115144584,
    0.7662003453186828,
    0.20651036895645647,
    0.03724062137286044,
    0.17110277274376795,
    0.7626426077793496,
    0.8372112804261309,
    0.8761690804447455,
    0.13887024930406633,
    0.8287513367412203,
    0.9794446290917873,
    0.807658524448803,
    0.8465629116398186,
    0.5187285629536083,
    0.33962953580139277,
    0.9798419666114342,
    0.6777071959103609,
    0.5388899884934379,
    0.7863389168762325,
    0.4274591420924474,
    0.25631366937500566,
    0.5695289062505289,
    0.026841382754547727,
    0.18267938207996903,
    0.9853642975717878,
    0.24428485895234409,
    0.5322028747608949,
    0.9655065842019517,
    0.043810183244384016,
    0.541216190236913,
    0.05897981610006209,
    0.2849168541804703,
    0.5349823008832073,
    0.9655676144971486,
    0.22831812764497283,
    0.7698701658704175,
    0.4103995069939841,
    0.25782763124411856,
    0.8490222628872495,
    0.39280879489916987,
    0.31999467883347554,
    0.2860820872456349,
    0.9684928577493004,
    0.9973831481899462,
    0.2930912094664657,
    0.4847128131859766,
    0.7218400909709828,
    0.40407009594106236,
    0.7059298060123587,
    0.45362146566562744,
    0.4640974655488792,
    0.16076769483252273,
    0.5989453525750241,
    0.585759299589679,
    0.9417035568973537,
    0.20117930667657413,
    0.5777873180244959,
    0.1991854396549344,
    0.8743781441651348,
    0.624666386634513,
    0.38720573630932886,
    0.9967931526923675,
    0.49817894572849486,
    0.24585267823751833,
    0.8639168275132305,
    0.2865624029759799,
    0.6163605496913385,
    0.5864748073339972,
    0.8781049154377354,
    0.7497547608938613,
    0.7864098057445887,
    0.0334170452332867,
    0.4875588105294657,
    0.6737395339380896,
    0.21851121231639659,
    0.2923739650597854,
    0.6073797612662293,
    0.41823228947229896,
    0.8531029420136382,
    0.3260916332061783,
    0.6306262204574675,
    0.5268576689601923,
    0.3516570914484707,
    0.8659366375222706,
    0.8447448461834428,
    0.3794548980890986,
    0.9832775904115916,
    0.8442256760399809,
    0.3006550591973338,
    0.9718660619781394,
    0.5103245035851833,
    0.794319831388071,
];

fn face_rotation(dir: Direction, rotation: u8) -> Quat {
    FACE_ROTATION_QUATS[dir as usize][rotation as usize]
}

pub struct MeshAsset {
    pub mesh: TriMesh,
}

pub struct MeshInstance {
    pub id: u32,
    pub position: Vec3,
    pub quaternion: Quat,
}

pub struct World {
    pub assets: HashMap<u32, MeshAsset>,
    pub mesh_instances: Vec<MeshInstance>,
    pub bvh: Option<Bvh>,
    pub aabbs: Vec<Aabb>,
}

impl World {
    fn from_track_info(track_info: &TrackInfo, assets: &HashMap<u32, TriMesh>) -> Self {
        let mut world = World {
            assets: HashMap::new(),
            mesh_instances: Vec::new(),
            bvh: None,
            aabbs: Vec::new(),
        };

        for (&id, mesh) in assets.iter() {
            world.add_asset(id, mesh.clone());
        }

        for part in &track_info.parts {
            for block in &part.blocks {
                let position = Vec3::new(
                    (block.x as i32 + track_info.min_x) as f32 * PART_SIZE,
                    (block.y as i32 + track_info.min_y) as f32 * PART_SIZE,
                    (block.z as i32 + track_info.min_z) as f32 * PART_SIZE,
                );
                let quaternion = face_rotation(block.dir, block.rotation);

                if assets.contains_key(&(part.id as u32)) {
                    world
                        .add_instance(part.id as u32, position, quaternion)
                        .unwrap();
                }
            }
        }

        world.build_bvh();
        world
    }

    fn add_asset(&mut self, id: u32, mesh: TriMesh) {
        self.assets.insert(id, MeshAsset { mesh });
    }

    fn add_instance(
        &mut self,
        asset_id: u32,
        position: Vec3,
        quaternion: Quat,
    ) -> anyhow::Result<()> {
        self.mesh_instances.push(MeshInstance {
            id: asset_id,
            position,
            quaternion,
        });

        Ok(())
    }

    fn build_bvh(&mut self) {
        self.aabbs.clear();

        for inst in &self.mesh_instances {
            let asset = &self.assets[&inst.id];

            let local = asset.mesh.local_aabb();
            let world = local.transform_by(&Pose::from_parts(inst.position, inst.quaternion));

            self.aabbs.push(world);
        }

        self.bvh = Some(Bvh::from_leaves(BvhBuildStrategy::default(), &self.aabbs));
    }

    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_toi: f32) -> (u32, f32) {
        let bvh = self.bvh.as_ref().expect("BVH not built");
        let ray = Ray::new(origin, dir.normalize());

        bvh.cast_ray(&ray, max_toi, |leaf_id, best| {
            let inst = &self.mesh_instances[leaf_id as usize];
            let asset = &self.assets[&inst.id];
            let iso = Pose::from_parts(inst.position, inst.quaternion);
            let local_ray = ray.inverse_transform_by(&iso);

            asset.mesh.cast_local_ray(&local_ray, best, true)
        })
        .map(|(leaf_id, toi)| {
            let inst = &self.mesh_instances[leaf_id as usize];
            (inst.id, toi)
        })
        .expect("Raycast failed")
    }
}

struct TableRng {
    index: usize,
}

impl TableRng {
    fn new(seed: Option<i64>) -> Self {
        let index = match seed {
            Some(s) => s.rem_euclid(RNG_TABLE.len() as i64) as usize,
            None => 0,
        };
        Self { index }
    }

    fn next(&mut self) -> f32 {
        self.index = (self.index + 1) % RNG_TABLE.len();
        RNG_TABLE[self.index]
    }
}

pub struct MountainMesh {
    pub vertices: Vec<f32>,
    pub offset: Vec3,
}

fn track_bounds(track_info: &TrackInfo) -> (i32, i32) {
    let mut max_x = i32::MIN;
    let mut max_z = i32::MIN;
    for part in &track_info.parts {
        for block in &part.blocks {
            max_x = max_x.max(block.x as i32 + track_info.min_x);
            max_z = max_z.max(block.z as i32 + track_info.min_z);
        }
    }
    (max_x, max_z)
}

fn mountain_radius(track_info: &TrackInfo, max_x: i32, max_z: i32) -> f32 {
    let width = (max_x - track_info.min_x).abs() as f32 * PART_SIZE / 2.0;
    let height = (max_z - track_info.min_z).abs() as f32 * PART_SIZE / 2.0;
    f32::max(
        MOUNTAIN_MIN_RADIUS,
        MOUNTAIN_RADIUS_BASE + f32::max(width, height) * SQRT_2,
    )
}

fn mountain_center(track_info: &TrackInfo, max_x: i32, max_z: i32) -> Vec2 {
    Vec2 {
        x: (track_info.min_x as f32 + (max_x - track_info.min_x) as f32 / 2.0) * PART_SIZE,
        y: (track_info.min_z as f32 + (max_z - track_info.min_z) as f32 / 2.0) * PART_SIZE,
    }
}

fn generate_rings(ring_count: usize, rng: &mut TableRng) -> Vec<Vec<f32>> {
    (0..ring_count)
        .map(|_| {
            (0..MOUNTAIN_RING_SEGMENTS)
                .map(|n| {
                    if n == 0 || n == 7 || (n == 1 && rng.next() < 0.5) {
                        0.0
                    } else {
                        rng.next()
                    }
                })
                .collect()
        })
        .collect()
}

fn triangulate_rings(rings: &[Vec<f32>], radius: f32) -> Vec<f32> {
    let mut vertices = Vec::new();
    let ring_count = rings.len();

    for e in 0..ring_count {
        let t = (e as f32 / ring_count as f32) * PI * 2.0;
        let i = ((e + 1) as f32 / ring_count as f32) * PI * 2.0;

        let current = &rings[e];
        let next = if e + 1 < ring_count {
            &rings[e + 1]
        } else {
            &rings[0]
        };

        for seg in 0..current.len() - 1 {
            let inner = radius + MOUNTAIN_RING_STEP * seg as f32;
            let outer = radius + MOUNTAIN_RING_STEP * (seg + 1) as f32;

            vertices.extend_from_slice(&[
                t.cos() * inner,
                current[seg] * HEIGHT_SCALE,
                t.sin() * inner,
                i.cos() * inner,
                next[seg] * HEIGHT_SCALE,
                i.sin() * inner,
                i.cos() * outer,
                next[seg + 1] * HEIGHT_SCALE,
                i.sin() * outer,
            ]);
            vertices.extend_from_slice(&[
                t.cos() * inner,
                current[seg] * HEIGHT_SCALE,
                t.sin() * inner,
                i.cos() * outer,
                next[seg + 1] * HEIGHT_SCALE,
                i.sin() * outer,
                t.cos() * outer,
                current[seg + 1] * HEIGHT_SCALE,
                t.sin() * outer,
            ]);
        }
    }

    vertices
}

fn build_mountain_mesh(track_info: &TrackInfo) -> MountainMesh {
    let (max_x, max_z) = track_bounds(track_info);
    let radius = mountain_radius(track_info, max_x, max_z);
    let center = mountain_center(track_info, max_x, max_z);

    if radius > MOUNTAIN_MAX_RADIUS {
        return MountainMesh {
            vertices: vec![],
            offset: Vec3::ZERO,
        };
    }

    let ring_count = (radius / 10.0).floor() as usize;
    let mut rng = TableRng::new(None);
    let rings = generate_rings(ring_count, &mut rng);
    let vertices = triangulate_rings(&rings, radius);

    MountainMesh {
        vertices,
        offset: Vec3::new(center.x, 0.0, center.y),
    }
}

pub struct StartTransform {
    pub position: Vec3,
    pub quaternion: Quat,
}

pub struct PlayerController {
    pub up: bool,
    pub right: bool,
    pub down: bool,
    pub left: bool,
    pub reset: bool,
}

struct Car {
    id: u32,
    controls: PlayerController,
}

pub struct SimulationWorker {
    physics: PolyTrackPhysics,
    exports: Exports,
    cars: Vec<Car>,
    car_state_buffer_ptr: i32,
    track_info: TrackInfo,
    max_checkpoint: u16,
    start: StartTransform,
    mountain_ptr: i32,
    track_ptr: i32,
    part_count: i32,
    mountain_vertices_len: i32,
    mountain_offset: Vec3,
    world: World,
}

impl SimulationWorker {
    pub fn new(mut physics: PolyTrackPhysics, export_string: &str) -> Self {
        let exports = physics.exports();
        let car_state_buffer_ptr = physics
            .alloc_bytes(&vec![0u8; 227])
            .expect("Failed to allocate car state buffer");

        let track_info = decode_track(export_string).expect("Failed to decode track");

        let max_checkpoint = track_info
            .parts
            .iter()
            .flat_map(|part| part.blocks.iter())
            .map(|block| block.cp_order.unwrap_or(0))
            .max()
            .unwrap_or(0);

        let start_block = find_start_block(&track_info).expect("Failed to find start block");
        let start = calculate_start_transform(start_block, &track_info);
        let mountain = build_mountain_mesh(&track_info);

        let mountain_ptr = physics
            .alloc_bytes(cast_slice(&mountain.vertices))
            .expect("Failed to allocate mountain mesh");
        let track_bytes = pack_track_data(&track_info);
        let track_ptr = physics
            .alloc_bytes(&track_bytes)
            .expect("Failed to allocate track data");
        let part_count = (track_bytes.len() / 19) as i32;
        let mountain_vertices_len = mountain.vertices.len() as i32;
        let mountain_offset = mountain.offset;

        let assets = assets();
        let mut assets_map = HashMap::new();
        for part in assets.track_parts.iter() {
            let vertices: Vec<Vec3> = part.vertices.chunks(3).map(Vec3::from_slice).collect();

            let indices: Vec<[u32; 3]> = (0..vertices.len() / 3)
                .map(|i| [i as u32 * 3, i as u32 * 3 + 1, i as u32 * 3 + 2])
                .collect();

            let mesh = TriMesh::new(vertices, indices).expect("Failed to create mesh");
            assets_map.insert(part.id, mesh);
        }
        let world = World::from_track_info(&track_info, &assets_map);

        Self {
            physics,
            exports,
            cars: vec![],
            car_state_buffer_ptr,
            track_info,
            max_checkpoint,
            start,
            mountain_ptr,
            track_ptr,
            part_count,
            mountain_vertices_len,
            mountain_offset,
            world,
        }
    }

    pub fn init(&mut self) -> anyhow::Result<()> {
        let assets = assets();

        let vertices_ptr = self
            .physics
            .alloc_bytes(cast_slice(&assets.car_collision_vertices))?;

        self.physics.call(
            &self.exports.init_car_collision_shape,
            (
                assets.car_mass_offset,
                vertices_ptr,
                assets.car_collision_vertices.len() as i32,
            ),
        )?;
        self.physics.free_wasm(vertices_ptr)?;

        for part in assets.track_parts.iter() {
            let part_vertices_ptr = self.physics.alloc_bytes(cast_slice(&part.vertices))?;

            let detector_default = Detector::default();
            let detector = part.detector.as_ref().unwrap_or(&detector_default);
            let start_offset = part.start_offset.unwrap_or([0.0; 3]);

            self.physics.call(
                &self.exports.add_track_part_config,
                (
                    part.id as i32,
                    part_vertices_ptr,
                    part.vertices.len() as i32,
                    detector.detector_type,
                    detector.center[0],
                    detector.center[1],
                    detector.center[2],
                    detector.size[0],
                    detector.size[1],
                    detector.size[2],
                    part.start_offset.is_some() as i32,
                    start_offset[0],
                    start_offset[1],
                    start_offset[2],
                ),
            )?;

            self.physics.free_wasm(part_vertices_ptr)?;
        }

        Ok(())
    }

    pub fn determinism_test(&mut self) -> anyhow::Result<bool> {
        let result = self.physics.call(&self.exports.test_determinism, ())?;
        Ok(result != 0)
    }

    pub fn create_car(&mut self, car_id: u32) -> anyhow::Result<()> {
        self.physics.call(
            &self.exports.create_car_model,
            (
                car_id as i32,
                self.mountain_ptr,
                self.mountain_vertices_len,
                self.mountain_offset.x,
                self.mountain_offset.y,
                self.mountain_offset.z,
                self.track_ptr,
                self.part_count,
                self.start.position.x,
                self.start.position.y,
                self.start.position.z,
                self.start.quaternion.x,
                self.start.quaternion.y,
                self.start.quaternion.z,
                self.start.quaternion.w,
            ),
        )?;

        self.cars.push(Car {
            id: car_id,
            controls: PlayerController {
                up: false,
                right: false,
                down: false,
                left: false,
                reset: false,
            },
        });

        Ok(())
    }

    pub fn delete_car(&mut self, car_id: u32) -> anyhow::Result<()> {
        self.physics
            .call(&self.exports.delete_car_model, car_id as i32)?;
        self.cars.retain(|c| c.id != car_id);
        Ok(())
    }

    pub fn set_car_controls(
        &mut self,
        car_id: u32,
        controls: PlayerController,
    ) -> anyhow::Result<()> {
        let car = self
            .cars
            .iter_mut()
            .find(|c| c.id == car_id)
            .ok_or_else(|| anyhow!("Car with id {car_id} not found"))?;

        car.controls = controls;
        Ok(())
    }

    pub fn update_car(&mut self, car_id: u32) -> anyhow::Result<CarState> {
        let car = match self.cars.iter().find(|c| c.id == car_id) {
            Some(car) => car,
            None => {
                eprintln!("Car with id {car_id} not found");
                return Err(anyhow!("Car with id {car_id} not found"));
            }
        };

        self.physics.call(
            &self.exports.update_car_model,
            (
                car_id as i32,
                car.controls.up as i32,
                car.controls.right as i32,
                car.controls.down as i32,
                car.controls.left as i32,
                car.controls.reset as i32,
                self.car_state_buffer_ptr,
            ),
        )?;

        let car_state_buffer = self.physics.wasm_slice(self.car_state_buffer_ptr, 227)?;
        CarState::deserialize(
            car_state_buffer[4..].try_into()?,
            self.max_checkpoint,
            &self.track_info,
        )
        .context("Failed to deserialize car state")
    }

    pub fn raycast(&self, origin: [f32; 3], dir: [f32; 3], max_distance: f32) -> (u32, f32) {
        self.world.raycast(
            Vec3::from_array(origin),
            Vec3::from_array(dir),
            max_distance,
        )
    }
}

fn decode_track(export_string: &str) -> anyhow::Result<TrackInfo> {
    let track =
        decode_track_code(export_string).ok_or_else(|| anyhow!("Failed to decode track code"))?;
    decode_track_data(&track.track_data).ok_or_else(|| anyhow!("Failed to decode track data"))
}

fn pack_track_data(track_info: &TrackInfo) -> Vec<u8> {
    let part_count: usize = track_info.parts.iter().map(|p| p.blocks.len()).sum();
    let mut buf = vec![0u8; 19 * part_count];
    let mut offset = 0;

    for part in &track_info.parts {
        for block in &part.blocks {
            buf[offset] = part.id;
            offset += 1;

            buf[offset..offset + 4]
                .copy_from_slice(&(block.x as i32 + track_info.min_x).to_le_bytes());
            offset += 4;

            buf[offset..offset + 4]
                .copy_from_slice(&(block.y as i32 + track_info.min_y).to_le_bytes());
            offset += 4;

            buf[offset..offset + 4]
                .copy_from_slice(&(block.z as i32 + track_info.min_z).to_le_bytes());
            offset += 4;

            buf[offset] = block.rotation;
            offset += 1;

            buf[offset] = block.dir as u8;
            offset += 1;

            let cp_order = block.cp_order.map(|x| x as i32).unwrap_or(-1);
            buf[offset..offset + 4].copy_from_slice(&cp_order.to_le_bytes());
            offset += 4;
        }
    }

    buf
}

struct StartBlock<'a> {
    block: &'a Block,
    start_offset: [f32; 3],
}

fn find_start_block(track_info: &TrackInfo) -> anyhow::Result<StartBlock<'_>> {
    let assets = assets();
    let mut best: Option<(u32, StartBlock)> = None;

    for part in &track_info.parts {
        let part_data = assets
            .track_parts
            .iter()
            .find(|p| p.id as u8 == part.id)
            .ok_or_else(|| anyhow!("Unknown track part id {}", part.id))?;

        let Some(start_offset) = part_data.start_offset else {
            continue;
        };

        for block in &part.blocks {
            let start_order = block
                .start_order
                .ok_or_else(|| anyhow!("Start part block is missing start_order"))?;

            if best
                .as_ref()
                .is_none_or(|(best_order, _)| start_order >= *best_order)
            {
                best = Some((
                    start_order,
                    StartBlock {
                        block,
                        start_offset,
                    },
                ));
            }
        }
    }

    best.map(|(_, sb)| sb)
        .ok_or_else(|| anyhow!("Track has no start position"))
}

fn calculate_start_transform(start: StartBlock, track_info: &TrackInfo) -> StartTransform {
    let block = start.block;
    let block_quat = face_rotation(block.dir, block.rotation);
    let y_flip = Quat::from_euler(EulerRot::XYZ, 0.0, PI, 0.0);
    let quaternion = block_quat * y_flip;

    let offset = quaternion * Vec3::from_array(start.start_offset);

    let position = Vec3::new(
        (block.x as i32 + track_info.min_x) as f32 * PART_SIZE + offset.x,
        (block.y as i32 + track_info.min_y) as f32 * PART_SIZE + offset.y,
        (block.z as i32 + track_info.min_z) as f32 * PART_SIZE + offset.z,
    );

    StartTransform {
        position,
        quaternion,
    }
}

pub struct WheelContact {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

pub struct CarState {
    pub frames: u32,
    pub speed_kmh: f32,
    pub has_started: bool,
    pub finish_frames: Option<u32>,
    pub next_checkpoint_index: u16,
    pub has_checkpoint_to_respawn_at: bool,
    pub position: [f32; 3],
    pub quaternion: [f32; 4],
    pub collision_impulses: Vec<f32>,
    pub wheel_contacts: [Option<WheelContact>; 4],
    pub wheel_suspension_lengths: [f32; 4],
    pub wheel_suspension_velocities: [f32; 4],
    pub wheel_delta_rotations: [f32; 4],
    pub wheel_skid_info: [f32; 4],
    pub steering: f32,
    pub brake_light_enabled: bool,
    pub controls: PlayerController,

    pub is_finishline_cp: bool,
    pub next_checkpoint_position: [f32; 3],
}

impl CarState {
    pub fn deserialize(
        buf: &[u8],
        max_checkpoint: u16,
        track_info: &TrackInfo,
    ) -> anyhow::Result<Self> {
        let mut r = Reader::new(buf);

        let frames = r.read_u24_le().context("frames")?;
        let speed_kmh = r.read_f32_le().context("speed_kmh")?;

        let flag_byte = r.read_u8().context("flag byte")?;
        let has_started = flag_byte & 1 != 0;
        let has_finish_frames = flag_byte & 2 != 0;
        let has_checkpoint = flag_byte & 4 != 0;
        let wheel_contact_present = [
            flag_byte & 8 != 0,
            flag_byte & 16 != 0,
            flag_byte & 32 != 0,
            flag_byte & 64 != 0,
        ];

        let finish_frames = if has_finish_frames {
            Some(r.read_u24_le().context("finish_frames")?)
        } else {
            None
        };

        let next_checkpoint_index = r.read_u16_le().context("next_checkpoint_index")?;
        let position = r.read_vec3().context("position")?;
        let quaternion = r.read_vec4().context("quaternion")?;

        let impulse_count = r.read_u8().context("impulse count")?;
        if impulse_count > 4 {
            bail!("Number of collision impulses exceeds maximum allowed");
        }
        let mut collision_impulses = Vec::with_capacity(impulse_count as usize);
        for i in 0..impulse_count {
            collision_impulses.push(
                r.read_f32_le()
                    .with_context(|| format!("collision_impulse[{i}]"))?,
            );
        }

        let wheel_contacts = {
            let mut contacts = [None, None, None, None];
            for i in 0..4 {
                if wheel_contact_present[i] {
                    contacts[i] = Some(WheelContact {
                        position: r
                            .read_vec3()
                            .with_context(|| format!("wheel_contact[{i}].position"))?,
                        normal: r
                            .read_vec3()
                            .with_context(|| format!("wheel_contact[{i}].normal"))?,
                    });
                }
            }
            contacts
        };

        let wheel_suspension_lengths = r.read_vec4().context("wheel_suspension_lengths")?;
        let wheel_suspension_velocities = r.read_vec4().context("wheel_suspension_velocities")?;
        let wheel_delta_rotations = r.read_vec4().context("wheel_delta_rotations")?;
        let wheel_skid_info = r.read_vec4().context("wheel_skid_info")?;

        let steering = r.read_f32_le().context("steering")?;

        let control_byte = r.read_u8().context("control byte")?;
        let controls = PlayerController {
            up: control_byte & 1 != 0,
            right: control_byte & 2 != 0,
            down: control_byte & 4 != 0,
            left: control_byte & 8 != 0,
            reset: control_byte & 16 != 0,
        };
        let brake_light_enabled = control_byte & 32 != 0;

        let is_finishline_cp = next_checkpoint_index == max_checkpoint + 1 || max_checkpoint == 0;
        let next_cp = if is_finishline_cp {
            let finish_parts = track_info
                .parts
                .iter()
                .filter(|part| FINISH_LINE_IDS.contains(&part.id))
                .collect::<Vec<_>>();

            finish_parts
                .iter()
                .flat_map(|part| part.blocks.iter())
                .map(|block| {
                    let block_pos = Vec3::new(
                        (block.x as i32 + track_info.min_x) as f32 * PART_SIZE,
                        (block.y as i32 + track_info.min_y) as f32 * PART_SIZE,
                        (block.z as i32 + track_info.min_z) as f32 * PART_SIZE,
                    );
                    let dist = Vec3::from_array(position).distance(block_pos);
                    (dist, block_pos)
                })
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .expect("No finish line blocks found")
                .1
                .to_array()
        } else {
            track_info
                .parts
                .iter()
                .flat_map(|part| part.blocks.iter())
                .filter(|block| block.cp_order == Some(next_checkpoint_index))
                .map(|block| {
                    [
                        (block.x as i32 + track_info.min_x) as f32 * PART_SIZE,
                        (block.y as i32 + track_info.min_y) as f32 * PART_SIZE,
                        (block.z as i32 + track_info.min_z) as f32 * PART_SIZE,
                    ]
                })
                .next()
                .expect("Next checkpoint block not found")
        };

        Ok(Self {
            frames,
            speed_kmh,
            has_started,
            finish_frames,
            next_checkpoint_index,
            has_checkpoint_to_respawn_at: has_checkpoint,
            position,
            quaternion,
            collision_impulses,
            wheel_contacts,
            wheel_suspension_lengths,
            wheel_suspension_velocities,
            wheel_delta_rotations,
            wheel_skid_info,
            steering,
            brake_light_enabled,
            controls,
            is_finishline_cp,
            next_checkpoint_position: next_cp,
        })
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.offset)
    }

    fn require(&self, n: usize) -> anyhow::Result<()> {
        if self.remaining() < n {
            bail!("CarState data is too short");
        }
        Ok(())
    }

    fn read_u8(&mut self) -> anyhow::Result<u8> {
        self.require(1)?;
        let v = self.buf[self.offset];
        self.offset += 1;
        Ok(v)
    }

    fn read_u16_le(&mut self) -> anyhow::Result<u16> {
        self.require(2)?;
        let v = u16::from_le_bytes(self.buf[self.offset..self.offset + 2].try_into()?);
        self.offset += 2;
        Ok(v)
    }

    fn read_u24_le(&mut self) -> anyhow::Result<u32> {
        self.require(3)?;
        let v = self.buf[self.offset] as u32
            | (self.buf[self.offset + 1] as u32) << 8
            | (self.buf[self.offset + 2] as u32) << 16;
        self.offset += 3;
        Ok(v)
    }

    fn read_f32_le(&mut self) -> anyhow::Result<f32> {
        self.require(4)?;
        let v = f32::from_le_bytes(self.buf[self.offset..self.offset + 4].try_into()?);
        self.offset += 4;
        Ok(v)
    }

    fn read_vec3(&mut self) -> anyhow::Result<[f32; 3]> {
        Ok([
            self.read_f32_le()?,
            self.read_f32_le()?,
            self.read_f32_le()?,
        ])
    }

    fn read_vec4(&mut self) -> anyhow::Result<[f32; 4]> {
        Ok([
            self.read_f32_le()?,
            self.read_f32_le()?,
            self.read_f32_le()?,
            self.read_f32_le()?,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_explorer;

    #[test]
    fn explore_world() {
        let track_info = decode_track("PolyTrack24pdrVdskciFE8XCb3wRMNeeewNMvHNe2G3XftlmdkmRaPuRSkZdjoSysIWv4pl4S1STwDRzKiHJBJ44eVK0lkDs4GjdNUOdkl0FmEorcHZ02TdYD1uJUV6qgwRXKwjucnL0y81hxFFHx9yNOANRKFKNfS627RCOiwqLSqIoFlBca72GYWC0GAph3jAGQtbwLcAkYThEeuF9olyBEAFh0XtTDTYWzuHpCktWNeOnLnZuqGFmva4dRuBGoHED6dZNjEIIRZnQi3pt03tqWqgE5dojDp2nTLy0KJWBW7PRI69Ol7TOmESCvFQeAOFbrxfGDd3AbV9Au1jAqVdYY3W7YRJGbqEAj05u8aNQRzmVnPkJj1iljU51pCgrLeqSY2pUtaXYcKJNeIKQYs9T2LfGeQrsUYVLT25i1IU2N3Bl8l1leqWa9Ahmk9D3UmwcQNnSBlynlISO23aJaNFjeS5gW8IMsZRsfpcE4qbRabrXIDJBNlQy7t1BCfEjwGyXJt5vZKi4LMYIM4xTlsZZHXuz8k9QEeJs9gJpIv9MVynPlZt002SGkTjkEk3U8NkhGeJuPFiv0jNS91k0LET77Mr8ImufhDgY6cfjVPkwhV7MSaMR7Z1IQcesNq4wmIYcObELODkoehQuEC9mcr6ovy3p6aXdqvDXi2dtnRcbPAfYsve3RkAYafM3mK1vkMh37PuK1XIBQezZPwtmoeUfkigd4vXUuVvhifBxn4el3ZxffRWOLmHX06B9wQBrhr8Mr3xadI2F1aPEDngAqfOgzbbHRTneH6mRhxgIldZEyi8Wc5jd2gygS0y0desgpuAyAeNMmDcLcLie5B2lzyQ1zIHQQzWF1vJjmtnfbC6UIXy6yI5zGyGyPxa9gw8E3BU0sVT9VCSCdJXhtVQcW0SRbtBeI8nKHHXfKgve2NTrheuJ4wTUmGci0Na49RGxeHZE3fMjY6k9EjbtFPLjZdKvZm6BhORyuwuWOEM1EQqSISU7ext7lwBY4JdtTVjUhtxALC37eRG3HgcqviUavOESJ2eV2ZvkpRw8y6wejYq0wHN1xCErYWzOucVFPqVFYj4ATHrW5aZPzdm7sSMHv8WyeEVdJ8fN1lsmEm19fWvLhLf9lpyecfsJBbIsISLTt5HEQYfe9keEbFMVCy1L7lLywRopov0q76br3oM0WaQqp2Yf380jZHzEeGnkMqBqBivxoHWXHSSm38jeRzvt53NWecbE51qtjqeC6m0bvmRjnO8rHDfzXtNmd38iRhPGSn5Ib19VPzBeF5SFqB5gb6T2v0KsxReMyCc7YWaDQgWfzZd1kuuE7OWSkjYGPp4bfJZnu9KpkteZjpuqweFhY2FBqHxSyYxe7ZEYAepeBIIkMetGNhgeOxaNLjve8nNlcfGHD7FsGXhIkGHBJ0HvfAUfojDoO4Hne5oQn51oeLkEz2Efdke7OyE3SoCbk50cfrzvYsY8mMmcmsVeScqj2rOfS9epl5fpQFGqjhGHVXY83C4WEB6WTHjwueP3tYLCGCZiHyc0WeEcupH1Cps0mmikCQnnfvfKG12peXnT1XNIfZEoaEy9aAeQ6ciSb4Gm0pSyR1UFqclSEBfKq5MeIySoPFFiLRcK8mztlSDeujKhYkksZ28j57oG8nmFx2siyrewpyIEvczxnW2Unyqnlt7VqvNY4zoVCfKtrrBCb7w8zjheyXMAeqo6HTRwTOhjSBSoyaf9fQCgsPOdIuMfPgaGstZ")
            .expect("Failed to decode track");

        let asset_data = assets();
        let mut assets_map = std::collections::HashMap::new();
        for part in asset_data.track_parts.iter() {
            let vertices: Vec<Vec3> = part.vertices.chunks(3).map(Vec3::from_slice).collect();
            let indices: Vec<[u32; 3]> = (0..vertices.len() / 3)
                .map(|i| [i as u32 * 3, i as u32 * 3 + 1, i as u32 * 3 + 2])
                .collect();
            let mesh = TriMesh::new(vertices, indices).expect("Failed to create mesh");
            assets_map.insert(part.id, mesh);
        }

        let world = World::from_track_info(&track_info, &assets_map);
        let start = find_start_block(&track_info).expect("No start block");
        let start_tf = calculate_start_transform(start, &track_info);

        world_explorer::run_explorer(
            &world,
            world_explorer::ExplorerConfig {
                start_pos: [
                    start_tf.position.x,
                    start_tf.position.y + 10.0,
                    start_tf.position.z,
                ],
                ..Default::default()
            },
        );
    }
}
