//! Deserialisation of the compiled simulation asset bundle.
//!
//! [`SIM_DATA`] is baked into the binary at compile time via `include_bytes!`.
//! [`assets`] parses it on first call and returns a shared reference to the
//! result on every subsequent call, so the binary blob is only walked once
//! regardless of how many times the rest of the codebase calls in.
//!
//! # Binary format
//!
//! The `.bin` file is a tightly-packed little-endian stream with no padding or
//! alignment:
//!
//! ```text
//! [car_collision_vertices: vec_f32]
//! [car_mass_offset: f32]
//! [part_count: u32]
//! for each part:
//!   [id: u32]
//!   [vertices: vec_f32]
//!   [has_detector: u8]  -- 1 = present, 0 = absent
//!   if has_detector:
//!     [detector_type: i32]
//!     [center: vec3]
//!     [size: vec3]
//!   [has_start_offset: u8]
//!   if has_start_offset:
//!     [start_offset: vec3]
//! ```
//!
//! `vec_f32` is a `u32` length followed by that many `f32` values.
//! `vec3` is three consecutive `f32` values (x, y, z).

use std::{
    io::{Cursor, Read},
    sync::OnceLock,
};

/// The raw asset bundle, embedded at compile time.
///
/// Parsed lazily on first access via [`assets`]; direct use of this slice is
/// only needed if you want to inspect the raw bytes.
pub const SIM_DATA: &[u8] = include_bytes!("../simulation_assets.bin");

/// Global cache for the parsed assets.
///
/// [`OnceLock`] guarantees that [`load_assets`] is called at most once even
/// under concurrent access, and that every caller after the first gets a
/// reference to the same allocation.
static ASSETS: OnceLock<SimulationAssets> = OnceLock::new();

/// Returns a reference to the parsed simulation assets, parsing [`SIM_DATA`]
/// on the first call and returning the cached result on every subsequent call.
///
/// This is the preferred entry point — callers should not need to call
/// [`load_assets`] directly unless they are working with a non-standard data
/// source (e.g. in tests).
pub fn assets() -> &'static SimulationAssets {
    ASSETS.get_or_init(|| load_assets(SIM_DATA))
}

/// Cursor-based reader for the little-endian binary asset format.
///
/// Wraps a [`Cursor`] and exposes typed read methods that advance the position
/// on each call.  All methods panic on unexpected EOF — the asset bundle is
/// baked into the binary, so a truncated read indicates a build-time error
/// rather than a recoverable runtime condition.
struct Reader<'a> {
    cur: Cursor<&'a [u8]>,
}

impl<'a> Reader<'a> {
    /// Creates a new reader for the given byte slice.
    fn new(data: &'a [u8]) -> Self {
        Self {
            cur: Cursor::new(data),
        }
    }

    /// Reads a single byte as a `u8`.
    fn u8(&mut self) -> u8 {
        let mut b = [0u8; 1];
        self.cur.read_exact(&mut b).unwrap();
        b[0]
    }

    /// Reads four bytes as a `u32` in little-endian order.
    fn u32(&mut self) -> u32 {
        let mut b = [0u8; 4];
        self.cur.read_exact(&mut b).unwrap();
        u32::from_le_bytes(b)
    }

    /// Reads four bytes as an `i32` in little-endian order.
    fn i32(&mut self) -> i32 {
        let mut b = [0u8; 4];
        self.cur.read_exact(&mut b).unwrap();
        i32::from_le_bytes(b)
    }

    /// Reads four bytes as an `f32` in little-endian order.
    fn f32(&mut self) -> f32 {
        let mut b = [0u8; 4];
        self.cur.read_exact(&mut b).unwrap();
        f32::from_le_bytes(b)
    }

    /// Reads a length-prefixed array of `f32` values.
    ///
    /// The length is a `u32` stored immediately before the values.
    fn vec_f32(&mut self) -> Vec<f32> {
        let len = self.u32() as usize;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.f32());
        }
        out
    }

    /// Reads three consecutive `f32` values as an XYZ vector.
    fn vec3(&mut self) -> [f32; 3] {
        [self.f32(), self.f32(), self.f32()]
    }
}

/// An axis-aligned bounding box that triggers a game event when a car enters
/// it.
pub struct Detector {
    /// Identifies what kind of event this detector fires (e.g. checkpoint,
    /// finish line).  Interpreted by the physics module.
    pub detector_type: i32,

    /// Centre of the detector volume in world space.
    pub center: [f32; 3],

    /// Half-extents of the detector volume on each axis.
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

/// A single piece of track geometry with its associated physics metadata.
pub struct TrackPart {
    /// Unique identifier used to look this part up in the physics module.
    pub id: u32,

    /// Flat list of XYZ vertex positions for the collision mesh.
    ///
    /// Length is always a multiple of 3; vertices are not indexed.
    pub vertices: Vec<f32>,

    /// Event detector attached to this part, if any.
    ///
    /// Most track parts are plain geometry and have no detector.
    pub detector: Option<Detector>,

    /// World-space offset that defines the car's starting position when this
    /// part is the spawn point, if applicable.
    pub start_offset: Option<[f32; 3]>,
}

/// All simulation data needed to initialise a physics run.
pub struct SimulationAssets {
    /// Flat list of XYZ vertex positions for the car's collision hull.
    ///
    /// Length is always a multiple of 3; vertices are not indexed.
    pub car_collision_vertices: Vec<f32>,

    /// Vertical offset of the car's centre of mass from its collision hull
    /// origin, in world units.
    pub car_mass_offset: f32,

    /// All track pieces that make up the current level.
    pub track_parts: Vec<TrackPart>,
}

/// Parses a simulation asset bundle from `data` and returns the result.
///
/// In normal use you should call [`assets`] instead, which caches the result
/// of parsing [`SIM_DATA`].  Call this directly only when you need to parse a
/// different byte slice (e.g. in unit tests with synthetic data).
///
/// # Panics
///
/// Panics if `data` is truncated or otherwise does not conform to the expected
/// binary layout.  Since the canonical input is `include_bytes!`'d at compile
/// time, a panic here means the asset bundle was generated incorrectly.
pub fn load_assets(data: &[u8]) -> SimulationAssets {
    let mut r = Reader::new(data);

    let car_collision_vertices = r.vec_f32();
    let car_mass_offset = r.f32();

    let part_count = r.u32() as usize;
    let mut track_parts = Vec::with_capacity(part_count);

    for _ in 0..part_count {
        let id = r.u32();
        let vertices = r.vec_f32();

        // A single byte acts as a boolean tag indicating whether the optional
        // field that follows is present in the stream.
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
