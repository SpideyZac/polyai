use bytemuck::cast_slice;

use crate::{
    data_reader::{Detector, assets},
    physics_worker::{Exports, INITIAL_FUEL, PolyTrackPhysics},
};

pub struct SimulationWorker<'a> {
    physics: &'a mut PolyTrackPhysics,
    exports: Exports,
}

impl<'a> SimulationWorker<'a> {
    pub fn new(physics: &'a mut PolyTrackPhysics) -> Self {
        let exports = physics.exports();
        Self { physics, exports }
    }

    pub fn init(&mut self) -> anyhow::Result<()> {
        self.physics.reset_fuel(INITIAL_FUEL)?;
        let assets = assets();
        let vertices_ptr = self
            .physics
            .alloc_bytes(cast_slice(&assets.car_collision_vertices))?;

        self.physics.call(
            &self.exports.init_car_collision_shape,
            (
                assets.car_mass_offset,
                vertices_ptr.ptr(),
                assets.car_collision_vertices.len() as i32,
            ),
        )?;

        self.physics.free_wasm(vertices_ptr)?;

        for part in assets.track_parts.iter() {
            self.physics.reset_fuel(INITIAL_FUEL * 4)?;
            let part_verticies_ptr = self.physics.alloc_bytes(cast_slice(&part.vertices))?;

            let detector_default = Detector::default();
            let detector = part.detector.as_ref().unwrap_or(&detector_default);
            let start_offset = part.start_offset.unwrap_or([0.0; 3]);

            self.physics.call(
                &self.exports.add_track_part_config,
                (
                    part.id as i32,
                    part_verticies_ptr.ptr(),
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

            self.physics.free_wasm(part_verticies_ptr)?;
        }

        Ok(())
    }
}
