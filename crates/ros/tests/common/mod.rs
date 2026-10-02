//! Shared by the test binaries.

use autonomousim_sim::record::{Recorder, RecorderConfig, Recording};
use autonomousim_sim::{BatchSim, Scenario};
use std::path::Path;
use std::sync::Arc;

/// Records the street scenario (camera at 160×120, LiDAR scans, camera frames at 10 Hz) for
/// 50 policy steps, then a second episode of 25, into `path`; returns the recording.
pub fn record_street(path: &Path) -> Recording {
    autonomousim_sim::camera::use_adapter(&autonomousim_render::AdapterChoice::Software);
    let toml =
        include_str!("../street_scenario.toml").replace("width = 640, height = 480", "width = 160, height = 120");
    let sc = Arc::new(Scenario::from_toml(&toml).unwrap().compile().unwrap());
    let mut b = BatchSim::from_compiled(sc.clone(), 1, 7, 1).unwrap();
    let config = RecorderConfig { lidar: true, camera_hz: 10, ..Default::default() };
    b.attach_recorder(0, Recorder::create(path, config).unwrap());
    let ego = vec![0.3f32; sc.groups[0].act_dim()];
    for k in 0..75 {
        if k == 50 {
            b.reset(None, None);
        }
        b.step(&[&ego, &[], &[]]);
    }
    b.detach_recorder(0).unwrap().finish().unwrap();
    Recording::read(path).unwrap()
}
