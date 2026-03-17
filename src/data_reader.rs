use std::{
    io::{Cursor, Read},
    sync::OnceLock,
};

pub const SIM_DATA: &[u8] = include_bytes!("../simulation_assets.bin");

static ASSETS: OnceLock<SimulationAssets> = OnceLock::new();

pub fn assets() -> &'static SimulationAssets {
    ASSETS.get_or_init(|| load_assets(SIM_DATA))
}

struct Reader<'a> {
    cur: Cursor<&'a [u8]>,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            cur: Cursor::new(data),
        }
    }

    fn u8(&mut self) -> u8 {
        let mut b = [0u8; 1];
        self.cur.read_exact(&mut b).unwrap();
        b[0]
    }

    fn u32(&mut self) -> u32 {
        let mut b = [0u8; 4];
        self.cur.read_exact(&mut b).unwrap();
        u32::from_le_bytes(b)
    }

    fn i32(&mut self) -> i32 {
        let mut b = [0u8; 4];
        self.cur.read_exact(&mut b).unwrap();
        i32::from_le_bytes(b)
    }

    fn f32(&mut self) -> f32 {
        let mut b = [0u8; 4];
        self.cur.read_exact(&mut b).unwrap();
        f32::from_le_bytes(b)
    }

    fn vec_f32(&mut self) -> Vec<f32> {
        let len = self.u32() as usize;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.f32());
        }
        out
    }

    fn vec3(&mut self) -> [f32; 3] {
        [self.f32(), self.f32(), self.f32()]
    }
}

pub struct Detector {
    pub detector_type: i32,
    pub center: [f32; 3],
    pub size: [f32; 3],
}

impl Default for Detector {
    fn default() -> Self {
        Self {
            detector_type: -1,
            center: [0.0; 3],
            size: [0.0; 3],
        }
    }
}

pub struct TrackPart {
    pub id: u32,
    pub vertices: Vec<f32>,
    pub detector: Option<Detector>,
    pub start_offset: Option<[f32; 3]>,
}

pub struct SimulationAssets {
    pub car_collision_vertices: Vec<f32>,
    pub car_mass_offset: f32,
    pub track_parts: Vec<TrackPart>,
}

pub fn load_assets(data: &[u8]) -> SimulationAssets {
    let mut r = Reader::new(data);

    let car_collision_vertices = r.vec_f32();
    let car_mass_offset = r.f32();

    let part_count = r.u32() as usize;
    let mut track_parts = Vec::with_capacity(part_count);

    for _ in 0..part_count {
        let id = r.u32();
        let vertices = r.vec_f32();

        let detector = if r.u8() == 1 {
            Some(Detector {
                detector_type: r.i32(),
                center: r.vec3(),
                size: r.vec3(),
            })
        } else {
            None
        };

        let start_offset = if r.u8() == 1 { Some(r.vec3()) } else { None };

        track_parts.push(TrackPart {
            id,
            vertices,
            detector,
            start_offset,
        });
    }

    SimulationAssets {
        car_collision_vertices,
        car_mass_offset,
        track_parts,
    }
}
