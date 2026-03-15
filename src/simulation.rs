use std::f32::consts::{PI, SQRT_2};

use anyhow::anyhow;
use bytemuck::cast_slice;
use glam::{EulerRot, Quat, Vec2, Vec3};
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

struct PlayerController {
    up: bool,
    right: bool,
    down: bool,
    left: bool,
}

struct Car {
    id: u32,
    controls: PlayerController,
    has_started: bool,
    frames: u64,
    is_paused: bool,
}

pub struct SimulationWorker {
    physics: PolyTrackPhysics,
    exports: Exports,
    cars: Vec<Car>,
}

impl SimulationWorker {
    pub fn new(physics: PolyTrackPhysics) -> Self {
        let exports = physics.exports();
        Self {
            physics,
            exports,
            cars: vec![],
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

    pub fn create_car(&mut self, export_string: &str, car_id: u32) -> anyhow::Result<()> {
        let track_info = decode_track(export_string)?;
        let start_block = find_start_block(&track_info)?;
        let start = calculate_start_transform(start_block, &track_info);
        let mountain = build_mountain_mesh(&track_info);

        let mountain_ptr = self.physics.alloc_bytes(cast_slice(&mountain.vertices))?;
        let track_bytes = pack_track_data(&track_info);
        let track_ptr = self.physics.alloc_bytes(&track_bytes)?;
        let part_count = track_bytes.len() / 19;

        self.physics.call(
            &self.exports.create_car_model,
            (
                car_id as i32,
                mountain_ptr,
                mountain.vertices.len() as i32,
                mountain.offset.x,
                mountain.offset.y,
                mountain.offset.z,
                track_ptr,
                part_count as i32,
                start.position.x,
                start.position.y,
                start.position.z,
                start.quaternion.x,
                start.quaternion.y,
                start.quaternion.z,
                start.quaternion.w,
            ),
        )?;

        self.physics.free_wasm(mountain_ptr)?;
        self.physics.free_wasm(track_ptr)?;

        self.cars.push(Car {
            id: car_id,
            controls: PlayerController {
                up: false,
                right: false,
                down: false,
                left: false,
            },
            has_started: false,
            frames: 0,
            is_paused: false,
        });

        Ok(())
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

fn calculate_start_transform(start: StartBlock<'_>, track_info: &TrackInfo) -> StartTransform {
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
