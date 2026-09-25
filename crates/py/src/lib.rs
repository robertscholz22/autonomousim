//! Python extension module `autonomousim._native`: batched simulation with numpy outputs.
//!
//! [`BatchSim`] wraps [`autonomousim_sim::BatchSim`]. Its outputs are numpy arrays created
//! once and overwritten in place after every `step` and `reset`, one set per agent group:
//! observations `float32 [num_envs, count, obs_dim]`, state rows `float64 [num_envs, count,
//! STATE_DIM]` and event bits `uint32 [num_envs, count]`. Stepping, resetting and building
//! the scenario release the GIL.

use autonomousim_sim::record::{Recorder, RecorderConfig};
use autonomousim_sim::{BatchSim as Batch, CompiledScenario, Events, STATE_DIM, STATE_FIELDS, Scenario, SimError};
use numpy::{PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyIndexError, PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

fn sim_err(e: SimError) -> PyErr {
    match e {
        SimError::Io(e) => PyIOError::new_err(e.to_string()),
        SimError::Record(m) => PyRuntimeError::new_err(m),
        e => PyValueError::new_err(e.to_string()),
    }
}

/// Version of the native extension (matches the Cargo package version).
#[pyfunction]
fn native_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Parse a scenario (TOML if `toml`, else JSON) and return it as JSON with every default
/// filled in. Raises `ValueError` on unknown fields or bad values (maps are not built).
#[pyfunction]
#[pyo3(signature = (text, toml = false))]
fn normalize_scenario(text: &str, toml: bool) -> PyResult<String> {
    let s = if toml { Scenario::from_toml(text) } else { Scenario::from_json(text) };
    Ok(s.map_err(sim_err)?.to_json())
}

/// The default scenario as JSON.
#[pyfunction]
fn default_scenario() -> String {
    Scenario::default().to_json()
}

/// Names of the built-in vehicle presets.
#[pyfunction]
fn vehicle_presets() -> Vec<&'static str> {
    autonomousim_vehicles::presets::names().collect()
}

/// Names of the built-in trailers (for a group's `trailers`).
#[pyfunction]
fn trailer_presets() -> Vec<&'static str> {
    autonomousim_vehicles::presets::trailer_names().collect()
}

/// A group given by index or name.
#[derive(FromPyObject)]
enum GroupRef {
    Index(usize),
    Name(String),
}

/// Output arrays of one group.
struct GroupArrays {
    obs: Py<PyArray3<f32>>,
    state: Py<PyArray3<f64>>,
    events: Py<PyArray2<u32>>,
}

/// Copy the batch outputs into the numpy arrays.
fn publish(py: Python<'_>, arrays: &[GroupArrays], sim: &Batch) -> PyResult<()> {
    for (g, a) in arrays.iter().enumerate() {
        a.obs.bind(py).readwrite().as_slice_mut()?.copy_from_slice(sim.obs(g));
        a.state.bind(py).readwrite().as_slice_mut()?.copy_from_slice(sim.state(g));
        a.events.bind(py).readwrite().as_slice_mut()?.copy_from_slice(sim.events(g));
    }
    Ok(())
}

/// Copy an action array (float32 or float64, any layout) into `out`.
fn stage_actions(a: &Bound<'_, PyAny>, num_envs: usize, out: &mut [f32], group: &str) -> PyResult<()> {
    let check = |shape: &[usize], len: usize| {
        if len != out.len() || shape.first() != Some(&num_envs) {
            Err(PyValueError::new_err(format!(
                "actions of group {group:?}: expected {} values as [num_envs = {num_envs}, ...], got shape {shape:?}",
                out.len()
            )))
        } else {
            Ok(())
        }
    };
    if let Ok(arr) = a.extract::<PyReadonlyArrayDyn<'_, f32>>() {
        check(arr.shape(), arr.len())?;
        out.iter_mut().zip(arr.as_array().iter()).for_each(|(o, &v)| *o = v);
    } else if let Ok(arr) = a.extract::<PyReadonlyArrayDyn<'_, f64>>() {
        check(arr.shape(), arr.len())?;
        out.iter_mut().zip(arr.as_array().iter()).for_each(|(o, &v)| *o = v as f32);
    } else {
        return Err(PyTypeError::new_err(format!(
            "actions of group {group:?} must be a float32 or float64 numpy array"
        )));
    }
    Ok(())
}

/// Booleans from a numpy bool array or a sequence.
fn bools(obj: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
    match obj.extract::<PyReadonlyArray1<'_, bool>>() {
        Ok(a) => Ok(a.as_array().to_vec()),
        Err(_) => obj.extract(),
    }
}

/// Non-negative integers from a numpy integer array or a sequence.
fn seeds(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    if let Ok(a) = obj.extract::<PyReadonlyArray1<'_, u64>>() {
        return Ok(a.as_array().to_vec());
    }
    if let Ok(a) = obj.extract::<PyReadonlyArray1<'_, i64>>() {
        return a
            .as_array()
            .iter()
            .map(|&s| u64::try_from(s).map_err(|_| PyValueError::new_err("seeds must be non-negative")))
            .collect();
    }
    obj.extract()
}

/// `num_envs` independent worlds of one scenario, stepped in parallel on a dedicated thread
/// pool. World `i` starts with the base seed `seed/env/i`.
#[pyclass(module = "autonomousim._native")]
struct BatchSim {
    /// Mutated only through `&mut self` (`Mutex::get_mut`, no locking); the mutex makes the
    /// class `Sync` for PyO3 and serialises the short `&self` reads.
    sim: Mutex<Batch>,
    scenario: Arc<CompiledScenario>,
    num_envs: usize,
    arrays: Vec<GroupArrays>,
    /// Actions of each group staged for the GIL-free step.
    actions: Vec<Vec<f32>>,
}

impl BatchSim {
    fn locked(&self) -> MutexGuard<'_, Batch> {
        self.sim.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn group(&self, g: &GroupRef) -> PyResult<usize> {
        let n = self.scenario.groups.len();
        match g {
            GroupRef::Index(i) if *i < n => Ok(*i),
            GroupRef::Index(i) => Err(PyIndexError::new_err(format!("group {i} out of range ({n} groups)"))),
            GroupRef::Name(s) => {
                self.scenario.group_index(s).ok_or_else(|| PyKeyError::new_err(format!("no group named {s:?}")))
            }
        }
    }

    fn env(&self, i: usize) -> PyResult<usize> {
        let n = self.num_envs;
        if i < n { Ok(i) } else { Err(PyIndexError::new_err(format!("env {i} out of range ({n} envs)"))) }
    }
}

#[pymethods]
impl BatchSim {
    /// Build the scenario (JSON text; see `normalize_scenario`) and `num_envs` worlds, each
    /// reset to its first episode. `num_threads = 0` uses one thread per logical CPU.
    #[new]
    #[pyo3(signature = (scenario, num_envs, seed = 0, num_threads = 0))]
    fn new(py: Python<'_>, scenario: &str, num_envs: usize, seed: u64, num_threads: usize) -> PyResult<Self> {
        let spec = Scenario::from_json(scenario).map_err(sim_err)?;
        let sim = py.detach(move || Batch::new(spec, num_envs, seed, num_threads)).map_err(sim_err)?;
        let scenario = sim.scenario().clone();
        let arrays = scenario
            .groups
            .iter()
            .map(|g| {
                let c = g.spec.count;
                GroupArrays {
                    obs: PyArray3::zeros(py, [num_envs, c, g.obs_dim()], false).unbind(),
                    state: PyArray3::zeros(py, [num_envs, c, STATE_DIM], false).unbind(),
                    events: PyArray2::zeros(py, [num_envs, c], false).unbind(),
                }
            })
            .collect::<Vec<_>>();
        let actions = scenario.groups.iter().map(|g| vec![0.0; num_envs * g.spec.count * g.act_dim()]).collect();
        publish(py, &arrays, &sim)?;
        Ok(Self { sim: Mutex::new(sim), scenario, num_envs, arrays, actions })
    }

    /// One policy step of every world. `actions` is one array `[num_envs, count, act_dim]`
    /// (any shape with `num_envs` rows and that many values; float32 or float64) per group,
    /// as a list or tuple, or a single array when there is one group. Actions are clipped to
    /// [−1, 1]; non-finite values read as 0.
    fn step(&mut self, py: Python<'_>, actions: &Bound<'_, PyAny>) -> PyResult<()> {
        let groups = &self.scenario.groups;
        let num_envs = self.num_envs;
        if actions.is_instance_of::<PyList>() || actions.is_instance_of::<PyTuple>() {
            let items: Vec<Bound<'_, PyAny>> = actions.try_iter()?.collect::<PyResult<_>>()?;
            if items.len() != groups.len() {
                return Err(PyValueError::new_err(format!(
                    "expected {} action arrays (one per group), got {}",
                    groups.len(),
                    items.len()
                )));
            }
            for ((a, out), g) in items.iter().zip(&mut self.actions).zip(groups) {
                stage_actions(a, num_envs, out, &g.spec.name)?;
            }
        } else if groups.len() == 1 {
            stage_actions(actions, num_envs, &mut self.actions[0], &groups[0].spec.name)?;
        } else {
            return Err(PyTypeError::new_err("pass a list with one action array per group"));
        }
        let sim = self.sim.get_mut().unwrap_or_else(PoisonError::into_inner);
        let staged: Vec<&[f32]> = self.actions.iter().map(Vec::as_slice).collect();
        py.detach(|| sim.step(&staged));
        publish(py, &self.arrays, sim)
    }

    /// Start new episodes in the worlds of `mask` (bool per world; all if `None`). With
    /// `seeds` (one per world), world `i` starts the first episode of `seeds[i]`; otherwise
    /// its next episode.
    #[pyo3(signature = (mask = None, seeds = None))]
    fn reset(
        &mut self,
        py: Python<'_>,
        mask: Option<&Bound<'_, PyAny>>,
        seeds: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let mask = mask.map(bools).transpose()?;
        let seeds = seeds.map(self::seeds).transpose()?;
        let n = self.num_envs;
        if mask.as_ref().is_some_and(|m| m.len() != n) || seeds.as_ref().is_some_and(|s| s.len() != n) {
            return Err(PyValueError::new_err(format!("mask and seeds need one value per world ({n})")));
        }
        let sim = self.sim.get_mut().unwrap_or_else(PoisonError::into_inner);
        py.detach(|| sim.reset(mask.as_deref(), seeds.as_deref()));
        publish(py, &self.arrays, sim)
    }

    /// Stop the agents of a group where `mask` (bool, `[num_envs, count]`) is true, for the
    /// rest of their episodes: they freeze and drop out of contacts and sensors, as after a
    /// terminal event.
    fn disable(&mut self, group: GroupRef, mask: PyReadonlyArrayDyn<'_, bool>) -> PyResult<()> {
        let g = self.group(&group)?;
        let count = self.scenario.groups[g].spec.count;
        if mask.len() != self.num_envs * count || mask.shape().first() != Some(&self.num_envs) {
            return Err(PyValueError::new_err(format!(
                "disable mask: expected [num_envs = {}, count = {count}], got shape {:?}",
                self.num_envs,
                mask.shape()
            )));
        }
        let mask: Vec<bool> = mask.as_array().iter().copied().collect();
        self.sim.get_mut().unwrap_or_else(PoisonError::into_inner).disable(g, &mask);
        Ok(())
    }

    /// Observations of a group, `float32 [num_envs, count, obs_dim]`, overwritten in place by
    /// every step and reset.
    #[pyo3(signature = (group = GroupRef::Index(0)))]
    fn obs<'py>(&self, py: Python<'py>, group: GroupRef) -> PyResult<Bound<'py, PyArray3<f32>>> {
        Ok(self.arrays[self.group(&group)?].obs.bind(py).clone())
    }

    /// State rows (`STATE_FIELDS`) of a group, `float64 [num_envs, count, STATE_DIM]`,
    /// overwritten in place.
    #[pyo3(signature = (group = GroupRef::Index(0)))]
    fn state<'py>(&self, py: Python<'py>, group: GroupRef) -> PyResult<Bound<'py, PyArray3<f64>>> {
        Ok(self.arrays[self.group(&group)?].state.bind(py).clone())
    }

    /// Event bits (`EVENTS`) of a group during the last policy step, `uint32 [num_envs,
    /// count]`, overwritten in place.
    #[pyo3(signature = (group = GroupRef::Index(0)))]
    fn events<'py>(&self, py: Python<'py>, group: GroupRef) -> PyResult<Bound<'py, PyArray2<u32>>> {
        Ok(self.arrays[self.group(&group)?].events.bind(py).clone())
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.num_envs
    }

    #[getter]
    fn num_threads(&self) -> usize {
        self.locked().num_threads()
    }

    #[getter]
    fn num_groups(&self) -> usize {
        self.scenario.groups.len()
    }

    #[getter]
    fn group_names(&self) -> Vec<String> {
        self.scenario.groups.iter().map(|g| g.spec.name.clone()).collect()
    }

    /// Physics time step (s).
    #[getter]
    fn dt(&self) -> f64 {
        self.scenario.dt()
    }

    /// Duration of a policy step (s).
    #[getter]
    fn policy_dt(&self) -> f64 {
        self.scenario.policy_dt()
    }

    /// Physics ticks per policy step.
    #[getter]
    fn decimation(&self) -> u32 {
        self.scenario.decimation
    }

    /// The scenario as JSON, with every default filled in.
    #[getter]
    fn scenario_json(&self) -> String {
        self.scenario.spec.to_json()
    }

    /// Content hashes (hex) of the maps in the pool.
    #[getter]
    fn map_hashes(&self) -> Vec<String> {
        self.scenario.map_hashes.iter().map(|h| h.hex()).collect()
    }

    /// Layout of a group: name, count, vehicle, family, action mode, `obs_dim`, `act_dim` and the
    /// observation terms as `(name, offset, length)`.
    #[pyo3(signature = (group = GroupRef::Index(0)))]
    fn group_info<'py>(&self, py: Python<'py>, group: GroupRef) -> PyResult<Bound<'py, PyDict>> {
        let g = &self.scenario.groups[self.group(&group)?];
        let d = PyDict::new(py);
        d.set_item("name", &g.spec.name)?;
        d.set_item("count", g.spec.count)?;
        d.set_item("vehicle", g.def.name())?;
        d.set_item("family", g.family().name())?;
        d.set_item("action_mode", g.action_mode().name())?;
        d.set_item("obs_dim", g.obs_dim())?;
        d.set_item("act_dim", g.act_dim())?;
        d.set_item("num_rotors", g.def.as_multirotor().map_or(0, |d| d.rotors.len()))?;
        d.set_item("mass", g.def.mass())?;
        d.set_item("obs_layout", g.obs.layout())?;
        Ok(d)
    }

    /// Simulated time of world `env` since its last reset (s).
    fn time(&self, env: usize) -> PyResult<f64> {
        let i = self.env(env)?;
        Ok(self.locked().world(i).time())
    }

    /// Index of the map (in the pool) of world `env`'s current episode.
    fn map_index(&self, env: usize) -> PyResult<usize> {
        let i = self.env(env)?;
        Ok(self.locked().world(i).map_index())
    }

    /// BLAKE3 hash of world `env`'s complete dynamic state (for determinism checks).
    fn state_hash<'py>(&self, py: Python<'py>, env: usize) -> PyResult<Bound<'py, PyBytes>> {
        let i = self.env(env)?;
        let h = self.locked().world(i).state_hash();
        Ok(PyBytes::new(py, &h))
    }

    /// Record world `env` to an MCAP file from now on (state and pose at `state_hz`, actions,
    /// events, episodes; LiDAR scans if `lidar`). A previous recording of the world is
    /// finished first.
    #[pyo3(signature = (env, path, state_hz = 50, lidar = false))]
    fn attach_recorder(&mut self, env: usize, path: std::path::PathBuf, state_hz: u32, lidar: bool) -> PyResult<()> {
        let i = self.env(env)?;
        let recorder = Recorder::create(&path, RecorderConfig { state_hz, lidar }).map_err(sim_err)?;
        let sim = self.sim.get_mut().unwrap_or_else(PoisonError::into_inner);
        match sim.attach_recorder(i, recorder) {
            Some(old) => old.finish().map_err(sim_err),
            None => Ok(()),
        }
    }

    /// Stop recording world `env` and finish the file; returns whether it was recording.
    fn detach_recorder(&mut self, env: usize) -> PyResult<bool> {
        let i = self.env(env)?;
        let sim = self.sim.get_mut().unwrap_or_else(PoisonError::into_inner);
        match sim.detach_recorder(i) {
            Some(r) => r.finish().map(|_| true).map_err(sim_err),
            None => Ok(false),
        }
    }

    /// Finish all recordings (also done when the object is dropped, ignoring errors).
    fn close(&mut self) -> PyResult<()> {
        let sim = self.sim.get_mut().unwrap_or_else(PoisonError::into_inner);
        let mut first = Ok(());
        for i in 0..sim.num_envs() {
            if let Some(r) = sim.detach_recorder(i)
                && let Err(e) = r.finish()
                && first.is_ok()
            {
                first = Err(sim_err(e));
            }
        }
        first
    }

    fn __repr__(&self) -> String {
        let s = self.locked();
        format!(
            "BatchSim(scenario={:?}, num_envs={}, groups={:?}, num_threads={})",
            self.scenario.spec.name,
            self.num_envs,
            self.group_names(),
            s.num_threads()
        )
    }
}

impl Drop for BatchSim {
    fn drop(&mut self) {
        let sim = self.sim.get_mut().unwrap_or_else(PoisonError::into_inner);
        for i in 0..sim.num_envs() {
            if let Some(r) = sim.detach_recorder(i) {
                let _ = r.finish();
            }
        }
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_scenario, m)?)?;
    m.add_function(wrap_pyfunction!(default_scenario, m)?)?;
    m.add_function(wrap_pyfunction!(vehicle_presets, m)?)?;
    m.add_function(wrap_pyfunction!(trailer_presets, m)?)?;
    m.add_class::<BatchSim>()?;
    m.add("STATE_DIM", STATE_DIM)?;
    m.add("STATE_FIELDS", STATE_FIELDS.to_vec())?;
    m.add("EVENTS", Events::NAMES.iter().map(|(n, e)| (*n, e.0)).collect::<Vec<_>>())?;
    m.add("TERMINAL_EVENTS", Events::TERMINAL.0)?;
    Ok(())
}
