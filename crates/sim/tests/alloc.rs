//! Stepping a world does not touch the heap (allocations would serialise threads on the
//! allocator and cost time per tick).

// A counting global allocator needs `unsafe`; it only forwards to the system allocator.
#![allow(unsafe_code)]

use autonomousim_core::rng::Seed;
use autonomousim_sim::{Scenario, WorldInstance};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn stepping_does_not_allocate() {
    let sc = Scenario::from_toml(
        r#"
        map = { type = "testworld", kind = "forest_patch", size = 120.0, density = 150.0, seed = 5 }
        [randomize_environment]
        wind_speed = [2.0, 2.0]
        turbulence_w20 = [7.7, 7.7]
        [[groups]]
        count = 4
        action_mode = "velocity"
        sensors = [
            { name = "imu", type = "imu" }, { name = "gps", type = "gps" }, { name = "baro", type = "baro" },
            { name = "mag", type = "mag" }, { name = "range", type = "rangefinder" }, { name = "lidar", type = "lidar" },
        ]
        obs = [ { term = "goal_rel_body" }, { term = "imu", sensor = "imu" }, { term = "lidar_log", sensor = "lidar" },
                { term = "clearance" } ]
        "#,
    )
    .unwrap();
    let mut w = WorldInstance::new(Arc::new(sc.compile().unwrap()), Seed::from_u64(0));
    let n = w.scenario().groups[0].obs_dim() * 4;
    let mut obs = vec![0.0; n];
    let actions = vec![0.1; 16];
    // Warm up: buffers reach their working size.
    for _ in 0..50 {
        w.set_actions(0, &actions);
        w.step();
        w.observe(0, &mut obs);
    }
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..50 {
        w.set_actions(0, &actions);
        w.step();
        w.observe(0, &mut obs);
    }
    let count = ALLOCATIONS.load(Ordering::Relaxed) - before;
    assert_eq!(count, 0, "{count} allocations in 50 policy steps");
}
