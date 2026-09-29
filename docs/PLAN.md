# Plan: autonomousim, a 3D autonomous-systems simulator (Rust core + Python training API)

## Context

Greenfield project. Goal: a 3D simulator
for training control policies (RL/IL) for aerial and ground vehicles. It has procedurally generated wild, rural
and urban maps, 6DOF aerial dynamics, vehicle-dynamics-grade multi-body ground vehicles, and a clean control and
observation interface. Visuals are 3D but not photorealistic. The full scope is large, so this plan fixes an
architecture that already supports all of it. It then details **Milestone 1** (quadrotor in a wild map, end to
end) and summarizes M2–M9.

### Decisions made with the user
| Topic | Decision |
|---|---|
| Stack | Rust Cargo workspace (headless core) + Python bindings (PyO3/maturin, `uv`) |
| Physics | Custom reduced-coordinate multibody (Featherstone ABA) + parry3d-f64 collision queries + penalty contacts; deterministic; f64 |
| Ground fidelity | Vehicle-dynamics grade: Pacejka MF tires with relaxation length, suspension K&C, steering, powertrain, brakes, load transfer. Tire contact goes through a `Terrain` query trait, not the collision engine |
| Ground scope | Diff-drive/skid-steer robots, Ackermann cars, multi-axle trucks + trailers, tracked vehicles (added 2026-09-24, M4), motorcycles/bicycles |
| Aerial scope | Multirotors, fixed-wing, helicopters, VTOL/tiltrotor |
| World | Seeded procedural generation, **separate map per character** (wild / rural / urban) |
| Agents | Multiple learning agents, mixed air+ground teams, scripted NPC traffic/pedestrians, swarms (100+) |
| Observations | State-first (GT state, IMU, GPS, baro, mag, rangefinder, raycast LiDAR); cameras later via headless wgpu |
| Compute | Laptop: i7-1365U 12T, 14 GB, Iris Xe (no CUDA). Optional desktop: RX 7900 XT (Linux, Vulkan/ROCm). CPU physics, rayon across envs; GPU only via wgpu |
| Viewer | Native Bevy app (interactive control, cameras, egui HUD, replay) |
| Interfaces | Python API (Gymnasium / PettingZoo / batched native VecEnv) + ROS 2 bridge (later) |
| Platforms | Linux only |
| RL tooling | PyTorch, CleanRL-style single-file PPO/SAC against the batched native env |
| Name | `autonomousim`: crates `autonomousim-*` (dirs `crates/<short>`), Python package `autonomousim` |
| Milestone 1 | **Quadrotor in a procedurally generated wild map**, end to end through PPO training and viewer replay |

## Architecture

### Crates (the core never depends on Bevy, PyO3 or ROS)
| Crate | Contents |
|---|---|
| `core` | `math` (spatial algebra, quaternions, frames); `dynamics` (joints, FK, ABA/RNEA/CRBA, integrators, energy); `contact` (colliders, penalty model, bristle friction, contact cache, broadphase); `Terrain` / `StaticGeometry` / material traits; `rng` (seed tree); integer-tick `time` |
| `world` | `HeightGrid` (implements `Terrain`, with a min/max mip pyramid), `StaticWorld` (terrain + obstacle BVH + water + material table), environment models (ISA atmosphere, wind + Dryden turbulence + gusts, magnetic field), map file I/O, cache, hashing, hand-made test worlds |
| `procgen` | Own f64 noise, terrain, erosion, hydrology, materials, scatter; `wild` generator (rural and urban later); map cache |
| `vehicles` | `VehicleDef` enum loaded from TOML; `multirotor` (M1); later `ground`, `fixedwing`, `rotorcraft` |
| `sensors` | IMU, GPS, baro, mag, rangefinder, LiDAR, ground truth; noise and latency primitives |
| `control` | Action modes, PID, multirotor cascade and control allocation (ground modes in M2) |
| `sim` | `Scenario` (TOML/JSON: map, environment and its randomization, events, agent groups with vehicle, controller, action mode, sensors, observations, spawn, goals, parameter randomization), `WorldInstance`, `Agent`, events, observation terms, agent–agent contacts, `BatchSim` (rayon), MCAP recorder |
| `scene` | Renderer-independent f32 meshes: primitives, terrain and water chunks at any LOD stride, obstacles merged per chunk, multirotor visuals (step 10a). Shared by the viewer now and headless cameras later |
| `py` | PyO3 cdylib `autonomousim._native` (abi3-py312): `BatchSim` with in-place numpy outputs (step 11) |
| `viewer` | Bevy binary (excluded from `default-members`, so `cargo test` never builds Bevy) |
| `cli` | `autonomousim mapgen / map-hash / map-info` (step 9); later `bench / run / inspect-rec` |

Dependency order: `core ← world ← procgen`; `core ← vehicles ← control`; `core,world ← sensors`; all of these `← sim ← {py, cli, viewer}`; `{world, vehicles} ← scene ← viewer`.

### Conventions and determinism
- **Units and frames**: SI units, radians. World frame is ENU (Z up), body frame is FLU (ROS REP-103). Quaternions are Hamilton, stored as glam `DQuat` (x, y, z, w), rotating body to world.
- **Math library**: glam f64 (`DVec3`, `DQuat`) throughout the core, matching parry3d-f64 0.31, which uses glam via glamx. Bevy uses glam 0.32 in f32, so the scene/viewer boundary converts through `[f32; N]`.
- **Frame conversions**, all in `core::math::frames` with round-trip tests:
  - ENU→Bevy: `(x, y, z) → (x, z, −y)`.
  - ENU→parry heightfield: parry's heightfield is Y-up, so it gets an `R_x(+90°)` pose and reversed rows. A test checks that parry ray casts agree with `Terrain::height`.
- **Time**: an integer `tick`, with `t = tick·dt`. Sensor, controller and policy rates must divide the physics rate; this is validated when configs load.
- **Determinism**:
  - Fixed iteration order (`Vec` with stable IDs; no `HashMap` iteration). No parallel float reductions; rayon only writes to disjoint slots.
  - **SeedTree**: each seed is `blake3(parent ‖ label)` → `ChaCha8Rng`. Streams: `map/<gen>/<tile>`, `env/<i>/episode/<k>/{spawn,wind,rand}`, `agent/<id>/sensor/<name>`. Uniform and normal sampling are our own code, so a `rand` upgrade cannot change sequences. Normals use a 128-layer ziggurat (Doornik's ZIGNOR, one `u64` per draw, Marsaglia tail; tables built with `libm`), 3× faster than Box–Muller; a test checks the CDF out to ±4σ and the third and fourth moments.
  - `procgen` uses the `libm` crate, so map hashes are bit-exact on both machines.
  - Map content hash = BLAKE3 over the postcard encoding of the map (metadata incl. generator version and seed, grid, heights, materials, water, obstacles in generation order, material table). The cache key is a separate hash of generator, version, config JSON and seed. Golden hashes are committed in `fixtures/golden_hashes.toml`.
- **Serialization**: toml for configs, postcard + zstd for the map cache, JSON for MCAP messages. Avoid bincode: 3.0.0 is a tombstone release.

### Core abstractions
```rust
trait JointModel { nq, nv, transform(q), motion_subspace(q) /*S(q), may depend on q → M2 K&C joints*/, bias(q,v) /*c_J*/, integrate(q,v,dt) }
enum Joint { Free, Revolute{axis}, Prismatic{axis}, Fixed, Spherical, KcTravel(Arc<KcTable>) /*M2*/, Custom(Arc<dyn JointModel>) }
struct MultibodyModel { parent: Vec<Option<u16>> /*parent < child*/, joint, x_tree, inertia, q_off, v_off, prescribed: BitSet }
struct MbState { q, v: SmallVec<[f64;16]>, qdd, kin: KinCache }        // per-agent, small
fn aba(model, state, forces: &ForceAccum, g, ws: &mut AbaWs)           // allocation-free
trait Terrain { extent; height(x,y); height_normal(x,y); material(x,y); water_level(x,y); height_bounds(min,max);
                closest_point(p, max) -> SurfacePoint{point, normal, signed distance, material, kind}; raycast(ray, max, HitMask) }
trait StaticGeometry { raycast(ray, max, HitMask); sphere_contacts(center, r, margin, HitMask, out); nearest_distance(p, max, HitMask);
                       query_candidates(min, max, HitMask, out: Vec<u32>); sphere_contact(id, center, r, margin) }
```
- **Static-world queries**: contacts and sensors take `&dyn Terrain` and `&dyn StaticGeometry` separately (`FlatTerrain`/`PlaneTerrain` exist for analytic tests). `StaticWorld` adds combined `raycast`, `clearance` and `is_free` for sensors and spawn sampling. `HitMask` selects terrain, water, solid, foliage and agents; foliage is hollow for rays (a sensor inside a canopy sees its boundary).
- **HeightGrid**: f32 heights, per-cell materials and water, and a min/max pyramid used for hierarchical ray marching and `height_bounds`. It is a solid block over its extent: rays entering through a side below the surface hit the "skirt" at the entry point.
- **VehicleDef**: immutable, loaded from TOML and shared by `Arc`. It holds a multibody template, **force elements**, actuators, colliders, sensor mounts and a default controller.
- **Force elements** are a closed enum: `Rotor`, `BodyDrag`, `Spring`; later `Tire`, `AeroSurface`, `Driveline`; plus a `Custom` escape hatch. Each declares `n_aux` extra continuous states, e.g. motor ω, or tire relaxation deflection in M2.
- **Integrator**: steps `(q, v, x_aux)` as one state vector (semi-implicit Euler by default, RK4 option). This is the main extension point for M2.
- **VehicleState**: `MbState` + aux states + `ParamScales` (domain randomization). The small per-agent state keeps swarms cheap and makes snapshot/restore trivial.
- **Agents**:
  - `Agent { id, group, vehicle: Multirotor, controller, setpoint, action, sensors: Vec<Sensor>, goals, events, disabled, air, turbulence }`. **Deviation**: M1 agents hold a `Multirotor` directly; the `Vehicle` enum over families (dispatched by enum rather than `Box<dyn>`) arrives with the second family in M2.
  - **Agent groups**: agents with the same vehicle, observation and action spec. Python gets arrays `[num_envs, group_size, dim]`, which map directly onto PettingZoo and parameter sharing.
  - A `WorldInstance` holds the `Arc<CompiledScenario>`, the episode's `Arc<StaticWorld>`, the environment state, agents, agent shapes and the clock. It is `Clone`, and a clone is the snapshot. Recorders are attached from outside (`step_with` calls back after every tick).
- **Batching**: `BatchSim` steps N worlds with `par_iter_mut` on its own rayon pool. Inside a world with at least 32 agents, the per-agent phases run in parallel. Agent–agent contacts use sweep-and-prune, with pairs resolved in a fixed order.

### Physics tick (per world, as implemented in `sim::world`)
1. Agent–agent contact forces from the agent shapes at the start of the tick (serial sweep, fixed order) into per-agent buffers.
2. Per agent (parallel from 32 agents): ground plane below; wind, turbulence and density on environment ticks (100 Hz); controller at the held setpoint (outer loops at 100 Hz inside); rotor forces with ground effect; terrain and obstacle penalty contacts; the buffered agent contacts; integration (ABA, motor lag, quaternion update); event bits; new shape.
3. The clock advances; every agent's sensors sample the new state at their dividers into latency lines (the phase is skipped when no agent has sensors).
4. Event bits are OR-ed per agent over the policy step: `CRASH_TERRAIN | CRASH_OBSTACLE | CRASH_AGENT | WATER | OUT_OF_BOUNDS | NAN | FOLIAGE | GROUND_CONTACT | LANDED | DISABLED | GOAL_REACHED | FINISHED`.
5. An attached recorder writes every k-th tick.

### Where task logic lives
| Concern | Location |
|---|---|
| Spawn/goal sampling (needs free-space queries), map choice from the pool, domain randomization | Rust `Scenario` (TOML); deterministic, so the viewer can reproduce episodes |
| Observation composition | Rust observation terms with scale and clip (`goal_rel_world`, `rot6d`, `lin_vel_body`, `ang_vel_body`, `last_action`, `goal_rel_body`, `lidar_log`, `agl`, `imu`, `gps_*`, ...; see the Sim section). The same spec lets the viewer or ROS run a policy without Python |
| Rewards, termination, truncation | Python, vectorized numpy over exported state/goal/event arrays, for fast iteration. Stable terms move into a Rust `RewardTerm` registry later |
| Conditions at physics rate (crashes between policy steps) | Rust event bits accumulated over the sub-steps |

### Python API
```python
import gymnasium as gym, autonomousim   # importing registers the environments
envs = gym.make_vec("autonomousim/QuadHover-v0", num_envs=256, num_threads=9, seed=0,
                    action_mode="ctbr", physics_hz=500, policy_hz=50)   # native VectorEnv
env  = gym.make("autonomousim/QuadHover-v0")                             # one world, used for check_env
```
**As built in step 11:**
- **`_native.BatchSim(scenario_json, num_envs, seed=0, num_threads=0)`** (`crates/py`, PyO3 0.29 + numpy 0.29, abi3-py312):
  - Output arrays are created once and overwritten in place after every `step` and `reset`, one set per group: `obs(g)` `float32 [N, count, obs_dim]`, `state(g)` `float64 [N, count, 20]`, `events(g)` `uint32 [N, count]`. Groups are addressed by index or name.
  - `step(actions)` takes one array per group (float32 or float64, any shape with N rows), or a single array when there is one group. Actions are staged into Rust buffers, and the step runs with the GIL released (`py.detach`). `reset(mask=None, seeds=None)` takes numpy arrays or lists; compiling the scenario also releases the GIL.
  - Metadata: `group_info(g)` (name, count, vehicle, action mode, dimensions, mass, observation layout), `dt`, `policy_dt`, `decimation`, `scenario_json` (with every default), `map_hashes`, `time(i)`, `map_index(i)`, `state_hash(i)`.
  - Recording: `attach_recorder(i, path, state_hz, lidar)`, `detach_recorder(i)`, `close()`. Dropping the object finishes open recordings.
  - Module level: `STATE_FIELDS`, `EVENTS`, `TERMINAL_EVENTS`, `normalize_scenario(text, toml)`, `default_scenario()`, `vehicle_presets()`.
  - Errors: scenario errors raise `ValueError`, I/O errors `OSError`, bad actions `ValueError`/`TypeError`.
  - Concurrency: PyO3 requires `Send + Sync`, so the simulation sits in a `Mutex` that `&mut self` methods reach through `get_mut` without locking. PyO3's borrow flag rejects concurrent calls instead of deadlocking.
  - `_native.pyi` stubs and `py.typed` are included.
- **Package** (`python/autonomousim`):
  - `events`: an `Event` `IntFlag`, `TERMINAL`, `names()` and `is_terminal()`.
  - `scenario`:
    - `STATE` slices;
    - `load_scenario` (TOML or JSON, validated, defaults filled in by Rust) and `normalize`;
    - `deep_merge` (dicts recursively, lists element-wise);
    - `quat_up_z`.
  - `tasks`: `Task`, `QuadHover`, `QuadRecover`, `make_task`.
  - `vector_env`, `env` and `bench`.
- **Tasks** (one agent per world):
  - A task builds the scenario dict. Rewards and end conditions are computed with numpy over the state rows and event bits of all worlds at once.
  - Common options:
    - `vehicle`;
    - `action_mode`, with the layouts documented in `Task`: ctbr = roll, pitch and yaw rate, then thrust;
    - `map`: `flat`, `forest`, `wild` (a pool of `map_count` training maps) or a map-source dict;
    - `physics_hz`, `policy_hz`, `episode_time`, `wind`, `randomize`, `obs`;
    - `overrides`, deep-merged into the scenario.
  - Every episode ends on a terminal event. A `terminal_penalty` (default 5) keeps early endings from paying off while the per-step reward is negative.
  - **QuadHover-v0**:
    - Spawn: cf2x in `ctbr`, 1.5–3.5 m AGL, tilt up to 30°, up to 1 m/s and 1 rad/s.
    - Goal: 0–2 m away horizontally.
    - Reward: `exp(−‖e_p‖) − 0.05‖ω‖ − 0.01‖Δa‖²`.
    - End: leaving a ±5 m box around the goal, or tilt > 90°. Truncated at 10 s.
    - `assets/scenarios/hover.toml` is the same scenario as a file; a test keeps them equal.
  - **QuadRecover-v0**:
    - Spawn: 3–5 m AGL at any tilt (uniform angle up to 180° about a uniform horizontal axis; this oversamples near-upright and inverted attitudes compared with uniform SO(3)), up to 5 rad/s and 3 m/s.
    - Goal: hold the spawn point.
    - Reward: `exp(−‖e_p‖) + 0.5·up_z − 0.05‖ω‖ − 0.01‖Δa‖²`.
    - End: leaving a ±10 m box. Truncated at 5 s.
- **`AutonomousimVectorEnv`** (`vector_entry_point`; also constructible directly with a task instance):
  - Spaces: `Box(−∞, ∞, obs_dim)` and `Box(−1, 1, act_dim)`, both float32. Rewards are float64 and flags bool.
  - `reset(seed=s)` gives world `i` the seed `s + i`, or `seed[i]` for a list; `options={"reset_mask": mask}` resets only those worlds.
  - Autoreset `SAME_STEP` (default): `info["final_obs"]` is a dense `float32 [N, obs_dim]` array (Gymnasium uses an object array), with the `_final_obs` mask.
  - Autoreset `DISABLED`: the user resets.
  - `NEXT_STEP` is rejected: worlds always step together, and masked stepping could be added in Rust if it is ever needed.
  - `info["events"]` on every step. When episodes end, `info["episode"] = {"r", "l"}` with `_episode`, in the layout of `RecordEpisodeStatistics`.
  - Observations are copied by default (`copy=False` returns the in-place view). `state` exposes the state rows.
- **`AutonomousimEnv`** (`entry_point`): N = 1 without autoreset. `reset(seed=s)` reproduces world 0 of the vector environment after `reset(seed=s)`.
- **Tests** (`tests_py`, 25):
  - Native layout and in-place outputs; action validation, clipping and non-finite actions; reset masks and seeds (lists and arrays).
  - World 0 independent of batch size and thread count; scenario errors; scenario helpers; events; the MCAP recording read back with the `mcap` package; every file in `assets/scenarios` builds.
  - `check_env` for both environments; vector spaces and dtypes; SAME_STEP final observations, episode statistics and the next episode; truncation; seeding; world 0 against the single environment; DISABLED with `reset_mask` (frozen agents); Recover spawn attitudes; task options; a wild map pool with its cache.

### Recording
- **Format**: MCAP with zstd chunks and JSON + jsonschema messages, so files open in Foxglove. Writing goes through a `TelemetrySink` trait, so a live viewer can later attach over zenoh or TCP.
- **Channels**:
  - `/meta`: scenario (including the map source and generator config), the map pool (generator, version, seed, content hash per map), vehicle defs, rates. `/episode` records which map of the pool an episode used. Maps are rebuilt and their hashes checked, never stored.
  - `/agent/{id}/state` at 50–100 Hz.
  - `/agent/{id}/action`, optional `/agent/{id}/lidar`, `/task/{reward,markers}`, `/events`.
- **Size**: measured about 18 kB/s per agent with JSON and zstd (5 agents, 3 of them with a 64-beam LiDAR at 10 Hz). CDR or protobuf encodings would shrink this if it matters.
- **ROS 2 path**: CDR encodings added later allow conversion to rosbag2.

## Milestone 1: Quadrotor in a wild map

### Defaults
| Parameter | Value |
|---|---|
| Physics | **500 Hz** for aerial-only worlds; 1 kHz preset for ground or mixed worlds (M2). One dt per world |
| Controller loops | Rate and attitude at physics rate; velocity and position at 100 Hz |
| Policy rate | 50 Hz (100 Hz for `motors` mode) |
| Sensor rates | IMU 500, baro 50, mag 50, GPS 10 (100 ms latency), rangefinder 50, LiDAR 10 Hz |
| Maps | Showcase: 2048×2048 m at 1 m, relief 0–300 m, ~60k trees. Training: 512×512 m, pool of 16 (each < 1 s to generate, < 5 MB) |
| Contact | ω_c = 0.2/dt (100 rad/s), ζ = 0.8; foliage ω_c = 20, ζ = 2 |
| Presets | `cf2x.toml` (27 g Crazyflie) and `iris_like.toml` (1.5 kg); values below |

**Preset values** (checked against the sources on 2026-09-23; `assets/vehicles/*.toml`, embedded by `vehicles::presets`):
- `cf2x`: m = 0.027 kg, arm = 0.0397 m, I = diag(1.4e-5, 1.4e-5, 2.17e-5), k_T = 2.88e-8 N/(rad/s)², k_Q = 7.24e-10 (converted from kf/km in N/RPM²), prop radius 23.1 mm, T/W = 2.25 (ω_max = 2273.5), rotor drag 9.18e-7 (`drag_coeff_xy`); hover ω ≈ 1516 rad/s. Props 0/2 CCW, 1/3 CW (pybullet yaw signs). Source: gym-pybullet-drones `cf2x.urdf`. Motor τ = 0.07 s from Crazyflie identification (arXiv:2404.07837). Rotor inertia (4e-8), body drag and colliders are estimates.
- `iris_like`: m = 1.5 kg, I = diag(0.029125, 0.029125, 0.055225), asymmetric arms (±0.13, ∓0.22 / ±0.20), k_T = 5.84e-6, k_Q = 0.06·k_T, ω_max = 1100 (hover ω ≈ 794, T/W ≈ 1.92), motor τ = 12.5 ms up / 25 ms down, rotor drag **1.75e-4** (current upstream; 8.06e-5 was the older RotorS value), rolling moment 1e-6, rotor inertia 2.74e-5 (sdf izz divided by `rotorVelocitySlowdownSim` = 10). Source: PX4-SITL_gazebo-classic `models/iris`. Body drag and colliders are estimates.

### Models and algorithms
- **Dynamics**:
  - ABA (Featherstone, *RBDA*, Table 7.1). Gravity enters as a fictitious base acceleration a₀ = −g.
  - The free base uses S = I₆ in body coordinates, with the 6×6 D solved by LDLᵀ.
  - Prescribed joints (hybrid dynamics, RBDA §9.2) are needed for M2 steering.
  - Semi-implicit Euler with the quaternion exponential map; RK4 for validation. For single free bodies (all aerial vehicles) the velocity-product term uses the implicit midpoint rule (simplified Newton, one 6×6 LU per step), which conserves kinetic energy exactly for torque-free motion. The explicit version gained 13 % energy in 10 s of tumbling. Articulated models still use the plain scheme; revisit wheel gyroscopics in M2/M5.
  - An unprescribed free root is solved in the ABA with a single Cholesky solve of `I^A` (3× faster for a single body).
  - **IMU pitfall**: under the gravity trick, specific force = `a_lin + ω×v_lin` (+ lever-arm terms). Spatial and classical acceleration differ.
- **Penalty contact** (`core::contact`, done):
  - Sphere colliders on links. Terrain points come from `Terrain::closest_point`, obstacle points from parry `project_point`. Normal force `F_n = max(0, kδ − c·v_n)` with `k = m_eff·ω_c²` and `c = 2ζ·m_eff·ω_c`. Stiffness set from ω_c·dt keeps it stable at any mass. The material's `stiffness_scale` s scales ω_c (k by s², c by s), for mud and snow.
  - Because of the force clamp (no pulling), ζ = 0.8 gives restitution e ≈ 0.16–0.18, not the linear-model 0.015. The test compares against a 1-D RK4 reference of the clamped model.
  - Bristle friction on persistent contacts, keyed by (collider, `HitKind` = terrain or obstacle index). Per step: `s ← proj_⊥n(s) + v_t·dt`, `F_t = −k_t·s − c_t·v_t`, capped at `μ·F_n`, with s reset to the cap while sliding. Measured on an incline: no creep in 9 s (drift 0); Coulomb slide acceleration exact; a rolling sphere reaches 5/7·g·sinθ within 0.04 %. The cache is per instance, so contacts must be evaluated exactly once per step.
  - Foliage: ω_c = 20 rad/s, ζ = 2 (stops a 2 m/s impact within about 2 cm, then pushes the body out).
  - Broadphase per agent: terrain `height_bounds` over the collider AABB, and one BVH query collecting the obstacle candidates. Each sphere then tests only the candidates, first with the AABB, then with an analytic shape distance lower bound (half-space or slab), and only then calls parry.
  - M1 drone colliders are spheres only (feet, prop guards, body corners).
- **Multirotor** (`vehicles::multirotor`, done):
  - `MultirotorDef` (TOML, `deny_unknown_fields`, validated): body, shared rotor parameters, rotor mounts (position, axis, spin), sphere colliders tagged `gear`/`frame`/`rotor`, contact constants, optional battery. `Multirotor` is the instance: runtime parameters (rebuilt by `set_scales` for domain randomization), state, buffers. Step phases are `begin_step → apply_rotors / apply_contacts / apply_force → finish_step`, so the sim can add agent–agent forces.
  - First-order motor lag with separate spin-up and spin-down τ, stepped exactly (`ω ← ω_cmd + (ω − ω_cmd)·e^{−dt/τ}`); commands are clamped to [ω_min, ω_max·V/V_full].
  - `T = k_T·(ρ/ρ_ref)·ω²·GE`, yaw torque `−s_i·(k_Q·ω² + J_r·ω̇)`, rotor drag `−ω_i·K_d·Π_⊥(v_hub − w)`, RotorS rolling moment, body quadratic drag `−½ρ·C_dA⊙|v|⊙v` and optional linear angular damping.
  - Rotor gyroscopics `−ω_b × Σ s_i·J_r·ω_i·a_i` go into the implicit-midpoint velocity update as internal angular momentum (`semi_implicit_euler_with_momentum`). Treated explicitly, this skew coupling grew the rates by up to ~18 %/s for cf2x.
  - Cheeseman–Bennett ground effect `1/(1 − (R/4z)²)`, clamped at z ≥ 0.5R. `z` is measured along the rotor axis to the local tangent plane of the terrain or water surface (`GroundPlane::below`, one `height_normal` query per vehicle).
  - Optional battery: linear open-circuit voltage curve, internal resistance and efficiency. The loaded voltage solves `V² − V_oc·V + P·R = 0` and scales ω_max.
  - The body frame is at the centre of mass, so the midpoint update takes a block-diagonal fast path: 3×3 Newton on ω, closed-form Cayley step for v.
  - Tests: hover equilibrium (error < 1e-9 after 2 s at ρ = 1.1), 63 % motor response at τ (spin-up and spin-down), per-motor roll/pitch/yaw signs for both presets, pure yaw without roll or pitch, GE(z = R) = 16/15, drag terminal velocity, wind ≡ airspeed, spin-up reaction conserving angular momentum exactly, rotor gyroscopics (bounded 1.8e-3, no drift over 100 s; a sign flip gives an O(1) error), battery sag, landing at rest on the gear with penetration g/ω_c².
  - Wind: per-episode mean (log profile with roughness length) + Dryden turbulence (MIL-F-8785C, with V floored at 2 m/s) + 1−cos gusts.
    - Each agent carries its own normalized Dryden filter state; agents don't share a turbulence field. The discretization is exact (matrix exponential and integrated noise covariance), so the statistics don't depend on dt, and the scaling keeps altitude or airspeed changes free of transients.
    - A step costs ~0.14 µs, so it can run at the 100 Hz controller rate.
- **Control** (`control::multirotor`, done; PX4-style cascade in physical units):
  - `position ─P→ velocity ─P+DOB→ thrust vector → attitude ─P→ body rates ─PID→ torque ─B⁺→ rotor speeds`. Rate and attitude loops run every physics step, velocity and position at 100 Hz. A `Setpoint` enters at any level; changing mode resets the outer loops (the rate integrator is kept unless the rate loop was bypassed). The controller only knows the nominal `MultirotorDef`, so randomized parameters are disturbances it must reject.
  - **Allocation**: B (4×n: roll, pitch, yaw torque and body-z thrust per newton of rotor thrust; columns `r_i × a_i − s_i·(k_Q/k_T)·a_i`), `B⁺ = Bᵀ(BBᵀ)⁻¹`; layouts whose unit-row-scaled Gram determinant is ≤ 1e-6 are rejected. PX4 sequential desaturation: collective first (reduce-only unless airmode), then roll, then pitch, then yaw with a 15 % margin, then a final reduce-only collective pass and a clamp → `ω_cmd = √(f/k_T)` at the design density.
  - **Rate loop**: PID in angular-acceleration units, derivative on the measurement, `τ = J·α + ω×Jω`. Gains by pole placement on `1/(s(τ_m·s + 1))`: `τ_m·s³ + (1+K_d)s² + K_p·s + K_i = τ_m(s² + 2ζω_n·s + ω_n²)(s + p)` with `ω_n = min(60, 3.5/τ_m, 0.15/dt)`, ζ = 0.7, p = 0.05·ω_n; yaw at 0.5·ω_n. PX4 i-factor and saturation-flag anti-windup (flags from the torque the allocation could not produce), integrator limited to 30 % of each axis' torque authority.
  - **Attitude loop**: tilt-prioritized quaternion control (Brescianini & D'Andrea 2018, PX4 `AttitudeControl`, yaw weight 0.4), gain ω_n/3.5, yaw-rate feedforward, rate setpoints clamped. A commanded yaw rate is integrated into a heading setpoint that may lead the actual heading by at most 0.5 rad.
  - **Velocity/position loops** (PX4 `PositionControl`): tilt limit, collective `m(g + a_z)/cos θ` floored at `thrust_min`, vertical priority with a horizontal margin, `bodyz → attitude` at the setpoint heading. Velocity gain = attitude gain/3, position gain = velocity gain/3.
  - **Deviation from PX4, velocity loop**: PX4 uses a velocity PID; this loop uses P plus a **disturbance observer**. The observer low-pass filters (bandwidth = velocity gain) the difference between the measured acceleration and the one the rotors produced, predicted from the actual attitude and the controller's copy of the motor lag. The loop then subtracts that estimate. It estimates wind, drag and model errors (mass, density, motor constants) and is cleared while the vehicle rests on something solid (`StateEstimate::ground_contact`). Why: a P–PI cascade always leaves an integrator pole–zero pair, so setpoint steps wound up the integrator and left a slow tail (iris 1 m climb: 2 % settle in 2.7 s). The observer does not see setpoint changes, which also suits policies that move the velocity setpoint every step. Trade-off: it differentiates the velocity, so with noisy estimates the bandwidth (`disturbance_ratio`) may need to come down.
  - **Velocity setpoint shaping** (`velocity` mode only): the commanded velocity passes through a critically damped second-order reference, `v̈_r = ω²(v_cmd − v_r) − 2ω·v̇_r`, with ω = `reference_ratio` (0.5) × velocity gain. Its acceleration is clamped to `accel_xy` (5 m/s²), `accel_up` (4) and `accel_down` (3), and while moving toward the command it is limited to ω·|error|, so it brakes without overshoot. The reference acceleration is fed forward. Heading-frame commands are shaped in the heading frame, rotated to the world and given the centripetal feedforward `ẑ × v_r · yaw rate`, so a turn does not lag. The reference restarts from the measured velocity when the mode or frame changes. Position mode stays unshaped (the 1 m step would take 7.7 s). Why: exploration noise and bang-bang policies move the command every step. Unshaped, the P loop turned each jump into a full tilt, saturated the rotors and lost lift, so random actions crashed every episode. Shaped, a ±3 m/s square wave switched every 40 ms moves the vehicle by less than 0.3 m/s.
  - **Action modes**, normalized to [−1, 1] (non-finite values read as 0): `motors` (ω_min…ω_max), `ctbr` (collective thrust 0…max + body rates ±(2π, 2π, π); the default), `attitude` (tilt vector in the heading frame ≤ 35°, yaw rate, thrust), `velocity` (heading frame, ±5 m/s horizontal on a disc, ±2 m/s vertical, yaw rate), `position` (offset ±(5, 5, 2) m in the heading frame from where the action was taken, heading ±π).
  - All gains come from `Tuning` ratios, so one configuration flies both presets. Measured on the full vehicle model over flat ground (cf2x / iris_like):

    | Check (requirement) | cf2x | iris_like |
    |---|---|---|
    | Roll/pitch rate step, 10–90 % rise (< 50 ms) / overshoot (< 20 %) | 34 ms / 10.4 % | 28 ms / 9.9–11.9 % |
    | Yaw rate step rise (< 150 ms) | 92 ms | 116 ms |
    | 10° tilt step, 5 % settle (< 0.3 s) | 0.116 s | 0.110–0.126 s |
    | 1 m step under 3.6 m/s wind, 2 % settle x / z (< 2 s), final error | 1.55 / 1.53 s, ≤ 1.4e-8 m | 1.18 / 1.34 s, ≤ 1.6e-8 m |
    | Wind switched on while holding: peak error, time to < 1 cm | 4.6 cm, 1.5 s | 3.9 cm, 1.2 s |
    | Recovery from 8 random attitudes incl. inverted: tracking the thrust direction | ≤ 0.72 s | ≤ 0.64 s |
    | Take-off to 2 m, 2 % settle; again after pushing into the ground for 5 s | 1.78 / 1.73 s | 1.51 / 1.50 s |

    Also tested: B·B⁺ = I (1e-12, including a hexarotor), saturated 30 rad/s roll command recovers without windup, full throttle keeps roll authority, a 12 m/s velocity command respects the 45° tilt limit, heading-frame velocity with a yaw rate, and a randomized vehicle (mass, inertia, motor, density scales) holding position.
- **Sensors** (`sensors`, done):
  - **Timing**: every sensor is updated once per physics tick with `BodyKinematics` (pose, world velocity, body rates, specific force at the CoM, angular acceleration) and a `SensorEnv` (world, ray scene, geodetic origin, atmosphere, magnetic field, g). It measures on the ticks its rate divides; a reading becomes visible after its latency through a `DelayLine`. Latencies must be whole ticks (checked to 1e-6 when the config loads), so timing is exact. `update` returns true when a new reading became visible.
  - **Randomness**: each sensor owns one seed stream (`agent/<id>/sensor/<name>`). Per-episode errors (turn-on biases, scale factors, hard-iron offsets, initial drift drawn from the stationary distribution) are redrawn on `reset(seed)`. Readings depend only on the seed and the kinematics, not on update order across sensors.
  - **Noise primitives** (`noise`): white noise from a spectral density (σ = N/√dt), exactly discretized first-order Gauss–Markov (no τ: random walk), quantization, overlapping Allan deviation.
  - **IMU**: `y = (1+s)·x + b₀ + b(t) + n`, clipped to the range and quantized, per axis. `x` is the true value averaged over the sample interval (like an anti-aliasing filter or delta-angle integration). The accelerometer sees `f + α×r + ω×(ω×r)` at its mount point. Default preset: ADIS16448 with the RotorS/PX4 Gazebo values (accel N = 0.004 m/s²/√Hz, K = 0.006, τ = 300 s, turn-on 0.196 m/s², ±18 g; gyro N = 2·35/3600 °/√s, K = 2·4/3600 °/s/√s, τ = 1000 s, turn-on 0.5 °/s, ±1000 °/s), 500 Hz; `ideal()` for noise-free tests.
  - **GPS**: 10 Hz, 100 ms latency. Gauss–Markov position error (1.5 m horizontal / 3 m vertical, τ = 60 s) + white noise (0.3 / 0.6 m, 0.05 m/s). Antenna lever arm applied. Reports ENU position, geodetic position through the map origin, velocity, EPH/EPV.
  - **Barometer**: ISA pressure at the sensor's altitude + turn-on offset (20 Pa) + drift (3 Pa, τ = 300 s) + white noise (1.5 Pa), and the pressure altitude derived from it. **Magnetometer**: world field in the sensor frame, 2 % scale error, 1 µT hard-iron offset, 0.3 µT noise.
  - **Rangefinder**: single beam (default straight down), 0.05–40 m, noise 2 cm + 0.5 % of range, optional dropout.
  - **LiDAR**: rings (elevations × azimuths over a field of view) or explicit beams; `rl64` (±8°, ±24° × 16 azimuths, 40 m) for RL and `vlp16_like` (16 × 1800, 100 m) for the viewer, both 10 Hz. A scan is taken at one instant (no motion distortion), with no latency. Ranges are `f32` (+∞ = no return), plus a `ReturnKind` per beam.
  - **Targets** for ray sensors: terrain, solid obstacles, foliage and agents each return the beam or let it pass. Water always stops the beam and returns it only if `water = true` (default false: near-infrared light is absorbed). Foliage is hollow, so a sensor inside a canopy sees its boundary.
  - **Ground truth**: pose, world and body velocity, rates, angular and specific acceleration, AGL (terrain or water below) and clearance to the nearest obstacle (up to 20 m), at 50 Hz with no noise.
  - **Config**: `SensorSpec { name, type, ... }` in TOML (`[[sensors]]`, `deny_unknown_fields`), dispatched through a `Sensor` enum. Not boxed: vehicles keep sensors in a `Vec` and update them in place every tick.
  - **Closed-form obstacle ray casts**: sphere, capsule, cylinder, cone and cuboid are intersected analytically; parry's GJK cast is kept only for convex hulls. A test compares 200k rays against parry (toi within 1e-4, the closed-form hit point on the surface within 1e-9 m, normals within 1e-3). This made LiDAR 3.3× faster; cone casts through GJK cost about 2 µs.
  - Tests: static IMU on a 10° gravel slope reads `R⁻¹(0, 0, g)` within 1e-6·g with zero rates; lever arm and mount rotation; Allan deviation recovers N (0.01 → 0.01000) and K (0.0316 → 0.0318); turn-on biases reproducible per seed; GPS latency exact in ticks and error statistics within 15 %; baro and mag against the models; rangefinder over ground and water; LiDAR in a walled arena exact against analytic ranges (walls, ground and open sky); foliage targets, noise and dropout rate; order-independent readings.
- **Sim** (`sim`, done):
  - **Scenario** (TOML or JSON, serde defaults, `deny_unknown_fields`): rates (physics 500, policy 50, environment 100 Hz; each must divide the physics rate), map source (a test world, or a pool of generated wild maps with one drawn per episode), nominal environment plus per-episode randomization (mean wind speed with uniform direction, turbulence W20, Poisson 1−cos gusts, temperature offset, sea-level pressure), event thresholds, agent groups. `compile()` validates everything, builds maps and resolves vehicles (preset name, TOML path or inline) into a `CompiledScenario` shared by `Arc`.
  - **Groups**: `count`, vehicle, controller config, action mode and limits, named sensors, observation terms (default: the 19-value hover observation), spawn (random in free space with clearance, separation and water avoidance, or a grid; AGL range or resting on the ground; initial speed, tilt, yaw, rates; motors at hover or idle), goals (hold the spawn, or chained random waypoints in free space; a draw that has to be clamped into the map is used only when no draw lands inside; `radius` > 0 advances to the next goal automatically once the vehicle is within it), parameter randomization (mass, inertia, k_T per rotor, k_Q, motor τ, drags; uniform 1 ± s; the controller keeps nominal values) and `disable_on_terminal`.
  - **Seeding**: world base seed → episode `k` → streams `map`, `environment`, `spawn`, `goals`, `agent/<id>/{vehicle, sensor/<name>, turbulence}`. Every value is drawn whether its option is set or not, so enabling one randomization does not shift the others. `reset(Some(s))` restarts the episode count of seed `s` and reproduces exactly. In a `BatchSim`, world `i` has base seed `seed/env/i`, independent of the batch size.
  - **Reset**: the environment and map are drawn, then per agent spawn, hover rotor speed at the local air density, goals, parameter scales, sensor and turbulence seeds (Dryden drawn from its stationary distribution). The setpoint holds the spawn pose until the first action.
  - **Events**: a contact is a crash when it touches the airframe or a rotor, or when the gear hits terrain or a solid faster than `crash_speed` (2 m/s). Gear contacts are otherwise `GROUND_CONTACT`, plus `LANDED` below 0.1 m/s and 0.3 rad/s. `WATER` is raised by contact with the water surface or by a collider below it. `OUT_OF_BOUNDS` covers the map edge minus a margin, `max_agl` and `ceiling`. `NAN` is raised by a non-finite state or integrator failure. Terminal events (crashes, water, out of bounds, NaN) freeze the agent until the next reset: it no longer moves, collides or appears to sensors, and reports `DISABLED`. With a goal `radius`, reaching the current goal raises `GOAL_REACHED`, and reaching the last one also `FINISHED` (not terminal: the task decides).
  - **Agent–agent contacts**: frictionless penalty spring–dampers between collider spheres. Stiffness and damping of the two agents act in series; each agent's values come from its contact model at its own mass. Both agents get `CRASH_AGENT`. Full-shape contacts with friction come in M3.
  - **Observation terms** (each with scale and clip; non-finite values written as 0; sensor terms read 0 before the first reading): goal relative in the world, body or heading frame; goal heading (sin, cos); position, height, AGL; rot6d, quaternion (w ≥ 0), gravity in the body frame, heading; linear velocity in the world, body or heading frame; body rates; rotor speeds mapped to [−1, 1]; last action; clearance (≤ 20 m); IMU (6 values, or accel or gyro alone); GPS position, velocity, goal − GPS position; baro altitude; magnetic field (µT); range; LiDAR ranges linear or `ln(1 + r)/ln(1 + max)`. `layout()` gives name, offset and length per term.
  - **Outputs**: observations `f32 [count, obs_dim]`; state rows `f64 [count, 20]` (`STATE_FIELDS`: position, orientation xyzw, world velocity, body rates, goal, goal yaw, AGL, goal index (= goal count once finished), clearance to the nearest terrain or solid obstacle up to 20 m) for rewards in Python; event bits `u32 [count]`.
  - **BatchSim**: N worlds on a dedicated rayon pool. Each world writes its outputs into its own buffers in parallel; these are then copied into the batch arrays `[N, count, dim]` per group in world order. `step(actions per group)`, `reset(mask, seeds)`, `world(i)`/`world_mut(i)` + `refresh(i)`, and an optional recorder attached to one world.
  - **Recorder** (`TelemetrySink` trait; `McapSink` with zstd chunks and JSON messages with jsonschema; `MemorySink` for tests):
    - `/meta` once: scenario, map pool (metadata and content hash per map), vehicle definitions, rates, state fields, agent list.
    - `/episode` on every reset: episode number, seed, map index, environment, spawn poses and goals.
    - `/agent/<id>/state` and `/agent/<id>/pose` (`foxglove.PoseInFrame`) at 50 Hz; `/agent/<id>/action` every policy step; optional `/agent/<id>/lidar`; `/events` on new event bits.
    - Log time is simulated time since the recording started, continuing across episodes. Errors never interrupt the simulation; `finish()` returns the first one.
  - **Tests**:
    - Same seed and actions give bit-identical state hashes (rigid-body state, rotor speeds, events, goal index, all sensor readings) over 150 policy steps. The mixed scenario has cf2x in `ctbr` with IMU, GPS and LiDAR, iris in `velocity` with baro and mag, a forest, wind, turbulence, gusts and randomized parameters.
    - World 0 is identical in a batch of 1 and of 12, and with 1 or 4 threads. A 40-agent world (parallel phases) matches with 1 and 6 threads.
    - Masked and seeded resets; `reset(Some(s))` reproduces; snapshot/restore round-trips.
    - 128 cf2x on a 1.5 m grid hold position in a 2.2 m/s wind for 10 s with no events. Worst drift is < 5 cm, final speed < 2 cm/s.
    - Crash, landed, water, out of bounds (above `max_agl` and past the edge), agent crash and NaN events.
    - A LiDAR sees another agent at the expected range.
    - Observation and state layout; action clipping and NaN handling.
    - MCAP round trip read back with `mcap::MessageStream`: message counts, timestamps, scenario equal after parsing. Recording does not change the simulation.
    - Stepping and observing allocate nothing: a counting global allocator sees zero allocations over 50 steps with all sensor types.
- **Wild procgen** (`procgen`, done; about 1.9 s for the 2 km showcase map, 0.1 s for a 512 m training map):
  1. **Noise** (`noise.rs`): OpenSimplex2-style 2-D gradient noise (quartic kernel with r² = 0.5, 24 gradients, integer hash). It uses only `+ − × floor`, so it is bit-identical everywhere. fBm and Musgrave ridged multifractal use per-octave seeds and a rotation between octaves. Golden values are pinned in a test.
  2. **Terrain** at twice the cell size (2 m):
     - A mask (fBm, 1.6 km period) blends fBm hills (800 m period, 0.25 × relief, gain 0.4) with ridged mountains (1.6 km, 0.9 × relief, gain 0.4).
     - A two-level domain warp (2 octaves, 80 m) bends both.
     - Tuning lesson: gain 0.5 adds equal slope per octave, and a 4-octave warp folds the coordinates. Both made cliffs; fixing them took the median slope from 39° to 18°.
  3. **Erosion**:
     - Particle hydraulic erosion (Beyer/Lague): 0.6 droplets per 2 m cell, brush radius 3, parameters in relief units, sequential with one seeded stream. The reference implementation's speed-update sign is fixed.
     - Then thermal erosion: a Jacobi gather form with symmetric pairwise exchange (mass-conserving, parallel), 60 iterations, talus 38°.
     - Finally a Catmull–Rom 2× upsample plus detail noise (0.25 m, 6 m period).
  4. **Hydrology** at 1 m:
     - Priority-Flood with a pit queue (Barnes 2014), keyed by (height in total order, index). The discovery tree gives flow accumulation.
     - Lakes are 8-connected groups of raised vertices. The level is the spill level unless the lake would cover more than 6 % of its catchment; then it drops (no outflow).
     - Lakes need at least 0.5 m depth and 200 m². A cell is wet when one of its corners is submerged.
  5. **Materials** per cell, with slope measured over a 3-cell baseline:
     - Lakebed: sand below 0.5 m depth, mud deeper. Rock above 42°.
     - Snow above the snowline (0.72 × relief ± noise) where the slope is below 38°.
     - Above the treeline (0.55 × relief ± noise): scree above 30°, otherwise alpine grass.
     - Sand on shores (within 3 m horizontally and 0.6 m above the lake level). Mud on wet flats.
     - Otherwise forest floor or grass, chosen by a forest noise mask plus moisture (log upstream area, blurred over 8 m).
  6. **Scatter**:
     - Trees:
       - Candidates are generated per 64 m tile in parallel (tile seeds `trees/<i>`). A candidate is accepted by material density (forest 350/ha, meadow 6/ha) if the slope is ≤ 35°, the ground is dry and it is below the treeline.
       - A sequential pass then enforces 3 m spacing through a spatial hash.
       - Species: conifers (cone crown; more of them higher up) or broadleaf (sphere crown, tag 6), 8–26 m tall and shorter at altitude. Trunks are sunk according to the slope.
     - Rocks:
       - Shape: Pareto sizes (0.4–4 m, exponent 2.2) as 14-point perturbed-ellipsoid hulls, 30 % sunk.
       - Density: 120/ha on rock and scree, 2/ha elsewhere, kept off trunks and water.
  7. **Presets**:
     - `training`: 512 m, 100 m relief, noise periods halved; about 3–4k trees and 1 MB on disk.
     - `showcase`: 2048 m, 300 m relief; about 43–49k trees, 7–10k rocks and 8–12 % water; 20.5 MB on disk.
     - Configs are TOML or JSON with defaults and `deny_unknown_fields`. `WildConfig::from_preset(preset, overrides)` deep-merges partial overrides.
  8. **Map files** (`world::mapfile`):
     - Layout: `AUTOSIMM` ‖ format version ‖ content hash ‖ zstd(postcard(map)). The hash is checked on load, and writes are atomic.
     - Obstacles are stored as externally tagged records, because postcard cannot read the internally tagged `ObstacleShape`.
  9. **Cache** (`procgen::cache`):
     - Key: BLAKE3 of generator, version, config JSON and seed.
     - Directory: `$AUTONOMOUSIM_MAP_CACHE`, else `$XDG_CACHE_HOME` or `~/.cache`, under `autonomousim/maps`.
     - Damaged or stale entries are regenerated.
  10. **CLI**:
      - `autonomousim mapgen` with `--preset --config --size --seed --threads --out --no-cache --preview (PPM) --print-config --json`.
      - `map-hash FILES…` and `map-info FILE`.
  11. **Sim**: `map = { type = "wild", seed, count, preset, config, cache }` builds a pool (maps are generated in parallel). Each episode draws one map.
  12. **Tests**:
      - Golden hashes of 3 maps, identical with 1 and 12 threads and in debug and release builds (`AUTONOMOUSIM_BLESS=1` rewrites them).
      - Invariants: lakes are flat, wet cells have a submerged corner, lakebeds are sand or mud. Trees are dry, grounded and not on cliffs; rocks are dry.
      - Config round trip and validation.
      - Cache: hit, miss, one entry per seed, and regeneration of a damaged entry.
      - A sim map pool with hashes in `/meta` and map indices in `/episode`.

### File layout (representative)
```
Cargo.toml  rust-toolchain.toml (1.98)  pyproject.toml  Makefile  .gitignore
crates/core/src/{math/, dynamics/{model,joint,kinematics,aba,rnea,crba,integrator}.rs, contact/, terrain.rs, rng.rs}  tests/  benches/
crates/world/src/{heightgrid,static_world,obstacles,materials,mapfile,testworlds}.rs, environment/{atmosphere,wind,magnetic}.rs
crates/procgen/src/{noise,terrain (base terrain, erosion, upsampling),hydrology,scatter,wild (config, pipeline, materials),cache}.rs  tests/wild.rs
crates/vehicles/src/multirotor/{def,rotor,motor,aero,ground_effect,battery,model,presets}.rs
crates/sensors/src/{lib (kinematics, env, targets, mount, timing),imu,gps,baro,mag,rangefinder,lidar,ground_truth,noise,latency,suite}.rs  tests/sensors.rs  benches/
crates/control/src/multirotor/{mod (cascade),action,allocation,rate,attitude,position,tuning}.rs  tests/closed_loop.rs  benches/
crates/sim/src/{agent,world (instance, tick schedule, outputs, hash),events,scenario (spec, compile, spawn/goal sampling, randomization),obs,interaction (agent contacts, agent ray scene),batch,record}.rs  tests/{sim,alloc}.rs  benches/sim.rs
crates/cli/src/{main (mapgen, map-hash, map-info),preview}.rs
crates/scene/src/{mesh (MeshData, primitives),terrain (chunks, terrain and water meshes),props (merged obstacles, multirotor visual)}.rs
crates/viewer/src/{main (live/replay commands, app, map seed regeneration, demo, screenshot),convert,sim (fixed-step live sim or replay, pilot modes, keyboard and gamepad),replay,history (plot data),overlay (goals, trails, LiDAR),world_view (map, lights, LOD, map switching),vehicle_view,camera,hud}.rs
crates/py/src/lib.rs  (BatchSim, module constants)
python/autonomousim/{__init__ (registration),_native.pyi,events,scenario,vector_env,env,bench,rl}.py, tasks/{base,hover,recover}.py; later recording.py, tasks/{waypoint_forest,landing}.py
assets/scenarios/{hover,forest}.toml
examples/{ppo_continuous,sac_continuous,eval_record,export_policy}.py
tests_py/  assets/{vehicles,maps,scenarios}/  fixtures/{pinocchio/,golden_hashes.toml}  tools/gen_pinocchio_fixtures.py
```
**pyproject**:
- `build-backend = "maturin"`; `manifest-path = "crates/py/Cargo.toml"`; `python-source = "python"`; `module-name = "autonomousim._native"`; feature `pyo3/abi3-py312`.
- `[tool.uv] cache-keys` over `crates/**/*.rs` and the `Cargo.toml` files, so uv rebuilds the extension when Rust changes.
- Dependency groups:
  - `dev`: pytest, gymnasium 1.3, numpy, mcap.
  - `train`: torch 2.14 CPU index; a ROCm variant for the desktop.
  - `oracle`: Pinocchio `pin` (4.1), for test fixtures. Pinocchio's `crba` assumes depth-first joint order, so the fixtures build M from RNEA columns.
- Use a Python 3.12 venv.
- Note: `~/.cargo/bin` is not on PATH in non-interactive shells, so scripts must `source ~/.cargo/env`.

### Implementation order (each step testable on its own)
| # | Step | Done when |
|---|---|---|
| 0 | `git init`, workspace scaffold, toolchain pin, lints, Makefile, pyproject + uv venv, maturin hello-module | `cargo test` passes and `uv run python -c "import autonomousim"` works |
| 1 | `core::math` | Property tests pass (X·X⁻¹ = I, ×* duality, inertia round-trip, quaternion exp/log, frame conversions) |
| 2 | `core::dynamics` (Free, Revolute, Prismatic, Fixed, Spherical; FK, RNEA, CRBA, ABA, integrators) | Analytic, conservation and Pinocchio tests pass; criterion baseline recorded |
| 3 | `world` runtime (HeightGrid, BVH, water, test worlds, atmosphere/wind/mag) | Height/normal/ray agree with parry; ISA values correct; Dryden PSD within tolerance |
| 4 | `core::contact` | Drop, incline stick/slip and restitution tests pass |
| 5 | `vehicles::multirotor` + presets | Hover equilibrium, sign and motor-response tests pass |
| 6 | `control`: allocation, cascade, action modes | Closed-loop step and saturation tests pass on a flat test world |
| 7 | `sensors` | Static IMU, Allan variance, analytic LiDAR and GPS latency tests pass |
| 8 | `sim`: WorldInstance, schedule, events, scenario, ObsSpec, BatchSim, MCAP recorder | Determinism suite passes; a swarm of 128 drones hovers; benchmarks recorded |
| 9 | `procgen::wild` + cache + `mapgen`/`map-hash` CLI | Golden hash identical with 1 and 12 threads; 2 km map ≤ 15 s cold, ≤ 1 s cached |
| 10a | Minimal viewer: terrain chunks, drone, chase camera, keyboard flight in `velocity` mode | Can fly over a generated map (built early to help debugging) |
| 11 | `py` bindings + package, `QuadHover`/`QuadRecover` tasks, `bench.py` | `check_env` and vector-semantics tests pass; throughput targets met |
| 12 | `ppo_continuous.py`, `sac_continuous.py`, `eval_record.py` (writes MCAP) | Hover policy converges; recording plays back |
| 10b | Full viewer: LOD, merged per-chunk vegetation, water, fog, cameras, egui HUD with plots, gamepad, MCAP replay with scrubbing, quality presets | ≥ 60 fps at 1080p "medium" on the Iris Xe |
| 13 | `QuadWaypointForest` (LiDAR). Stretch: `QuadLanding`, Rust MLP policy playback in the viewer (done, see the viewer notes) | > 80 % success on unseen maps; end-to-end demo |

### Tasks and training
| Task | Obs | Action | Reward / end conditions | Budget (laptop) |
|---|---|---|---|---|
| QuadHover-v0 | Position error, rot6d, v, ω, last action (19) | `ctbr` | `exp(−‖e_p‖) − 0.05‖ω‖ − 0.01‖Δa‖²`. Ends on crash, leaving a 10 m box, or tilt > 90°; truncated at 10 s | PPO 10M steps in ~5 min; SAC 200k steps in ~3 min (measured, see below) |
| QuadRecover-v0 | Same | `ctbr` / `motors` | Upright + position terms, starting from uniform SO(3) attitude, \|ω\| ≤ 5 rad/s, \|v\| ≤ 3 m/s; 5 s | 5–10M steps |
| QuadWaypointForest-v0 | Body-frame goal, v, ω, rot6d, AGL, 4 × 32 LiDAR (148) | `velocity` | Progress + waypoint bonus − proximity − canopy − smoothness − failure. Ends on crash, water, leaving the map or 10 m AGL, success (3 waypoints within 2 m), or 60 s | 30M steps in 45 min with `--bound-coef 0.01`: 88 % success on unseen maps (see step 13) |
| QuadLanding-v0 (stretch) | Noisy GPS/baro/rangefinder, pad position | `velocity` | Shaped approach + soft-touchdown success | — |

**As built in step 12** (`examples/`, `python/autonomousim/rl.py`; `make train-deps` installs torch 2.14 CPU and tensorboard):
- **`autonomousim.rl`** (numpy only, shared by the scripts): `RunningMeanStd`, `ObsNormalizer` (clip ±10, saved with the policy), `RewardScaler` (divides by the running std of the discounted return, as Gymnasium's `NormalizeReward`), and `evaluate(policy, env_id, episodes, seed)`. `evaluate` runs one deterministic episode in each of N worlds with autoreset DISABLED and reports the mean return and length, the fraction that reached truncation, and their median final distance to the goal. For tasks with a success condition (`Task.has_success`), it also reports the fraction that succeeded, the fraction that failed and the mean number of goals reached.
- **`ppo_continuous.py`** (CleanRL `ppo_continuous_action` with these changes):
  - The native vector env (`gym.make_vec`, SAME_STEP) steps all worlds in one call.
  - Truncated episodes bootstrap: `r += γ·V(final_obs)`.
  - Observation and reward normalization run in the loop rather than in wrappers.
  - Separate thread budgets: `--sim-threads 9`, `--torch-threads 3`.
  - Defaults: N = 256 × 64 steps, γ = 0.99, λ = 0.95, learning rate 3e-4 with linear decay, 4 minibatches × 5 epochs, clip 0.2 (policy and value), 2×128 tanh MLPs with orthogonal init, state-independent log-std starting at −0.5, 10M steps.
  - Task options: `--env-kwargs '{"action_mode": "motors", "map": "wild"}'`.
  - Output: `runs/<run>/` with `policy.pt` (network, observation statistics, arguments, `algo`), `args.json`, `eval.json` and TensorBoard scalars.
- **`sac_continuous.py`** (CleanRL `sac_continuous_action`):
  - 16 worlds with 4 gradient updates per vector step (0.25 per transition), 2×128 ReLU MLPs, tanh-squashed Gaussian actor, twin Q networks with Polyak targets (τ = 0.005).
  - Delayed actor updates (every 2); automatic entropy tuning reusing the actor's log-probabilities.
  - numpy ring buffer of raw observations, normalized with the current statistics when sampled. Episode-ending transitions store `final_obs` as the next observation and mask only terminations.
  - 10k random-action steps before learning; 200k steps by default.
  - Two torch threads. An update takes 2.4 ms at 128 wide and 4.9 ms at 256 wide (CleanRL's default). Four threads are slower (7.0 ms at 256 wide), because of E-cores and synchronization. On hover, 256 wide learns no better per sample.
- **`eval_record.py`**:
  - Loads any checkpoint through the script that wrote it (`algo`), flies `--episodes` deterministic (or `--stochastic`) episodes in one world with `attach_recorder`, and prints return, length, outcome, final error and events per episode.
  - It then checks the file in two ways:
    1. **Read-back**: every recorded state (position, orientation, velocity, rates) equals the live state rows bit for bit, with one action per step.
    2. **Replay**: a fresh `BatchSim` built only from `/meta` rebuilds the maps, checks their hashes, resets with the episode seeds and feeds the recorded actions. It reproduces every recorded state bit for bit. The viewer replay (10b) relies on the same property.
  - Recordings go to `recordings/<run>.mcap`: about 16 KiB/s for one cf2x at 50 Hz.
- **Results on QuadHover-v0** (laptop, powersave; evaluation over 512 unseen episodes, deterministic policy):

  | Run | Wall time | Survival | Final error | Notes |
  |---|---|---|---|---|
  | PPO, 10M steps (default) | 4.5 min alone (6.2 min alongside a second run) | 100 % | 3.1 cm | Return ≈ 435 of 500. Episodes last the full 10 s from about 2.5M steps |
  | PPO, 5M steps | 2.2 min | 96–100 % (by seed) | 0.28–0.29 m | The mean action hovers about 0.2 m high; the sampled policy holds 0.1 m. Exploration noise on the body rates tilts the drone and costs lift, and the mean thrust learns to compensate. More steps shrink the std (0.32 → 0.19) and the bias |
  | PPO, 5M steps, log-std −1 | 2.5 min | 88–91 % | 9 cm | Learns faster at first but visits fewer states. The mean policy sometimes keeps pitching until it flips |
  | SAC, 2×128, 200k steps (default) | 3 min (1,100 SPS) | 100 % | 0.36 m | Full-length episodes from 120k steps. The deterministic and sampled policies are equally imprecise: Q (≈ 55 against ≈ 95 at a perfect hover) is still converging |
  | SAC, 2×256, 200k steps | 5.8 min (570 SPS) | 100 % | 0.29 m | Same learning curve per sample |
  | SAC, 2×128, 1M steps | 20.6 min (1,000 SPS) | 100 % | 0.14 m | Return 393 (training 377); Q ≈ 79 and still rising, α ≈ 0.005. Precision keeps improving slowly; PPO with 10M steps gets further in a quarter of the time |

  The full PPO loop runs at 37–43k SPS: the simulation takes 12–14 % of the time and torch the rest.

**As built in step 13** (`python/autonomousim/tasks/waypoint_forest.py`, plus what the task needed in `sim`, `control` and the Python package):
- **Simulation support**:
  - Goals with a reach `radius` (`GoalSpec.radius`, default 0 = advanced only explicitly). Within the radius of the current goal, the agent advances to the next one and raises `GOAL_REACHED`. After the last one it also raises `FINISHED`, and `goal_index` equals the goal count. Neither event is terminal: the task decides.
  - Random goal chains prefer draws inside the map region. A draw that had to be clamped to the edge is used only when no draw lands inside; otherwise goals piled up in map corners.
  - A `clearance` column in the state rows: the distance to the nearest terrain or solid obstacle, up to 20 m (`STATE_DIM` = 20). It is used for rewards, not for observations.
  - Velocity setpoint shaping in the controller (see Control): with raw velocity commands, random or exploring actions crashed every episode.
- **Python API**:
  - `Task.has_success` and `succeeded()`: success ends the episode as terminated without the terminal penalty.
  - `settings()` for scenario-level entries (e.g. event thresholds).
  - `reset(mask, state)` so tasks can keep per-episode state.
  - `reward(state, action, prev_action, events)`.
  - Results: the vector env reports `info["episode"]["success"]`, the single env `info["success"]`, and `evaluate` reports success, failure and goals reached.
  - Training scripts:
    - `ppo_continuous.py` `--save-every` writes checkpoints during training;
    - `--eval-env-kwargs` (both scripts) sets the final evaluation's task options, e.g. unseen maps with `{"map_seed": 1000}`;
    - `ppo_continuous.py` `--bound-coef` adds rl_games' bounds loss `Σ max(0, |mean| − 1.1)²` on the action mean (default 0, so the hover results above are unchanged);
    - both scripts log the success rate.
  - `eval_record.py` records at the policy rate and reports success as an outcome.
- **QuadWaypointForest-v0**:
  - **Setup**:
    - vehicle: `iris_like` in `velocity` mode, policy at 25 Hz;
    - wind: 0–3 m/s mean per episode;
    - maps: a pool of 16 generated 512 m wild maps (`map_seed` picks the pool);
    - spawn: 2–4 m above the ground, with 3 m clearance to solids, at least 40 m from the edge;
    - goals: 3 waypoints, each 20–50 m from the previous one, 2–5 m above the ground, 3 m clearance, reach radius 2 m.
  - **Ends**:
    - success when the last waypoint is reached;
    - failure on a crash (gear impacts count above 2 m/s), water, the map edge minus 5 m, or more than 10 m above the ground. The height limit keeps the vehicle below the treetops, so it has to fly through the forest rather than over it;
    - truncation after 60 s.
  - **Observation** (148 values): goal in the body frame (1/20, clipped to ±3), body velocity and rates, rot6d, AGL, last action, and log ranges of a 4 × 32 LiDAR (±8°, ±24°, 40 m, 10 Hz).
  - **Reward per step**:
    - progress toward the current goal, measured from the positions before and after the step, so a goal switch does not jump;
    - +10 per waypoint;
    - −0.2·max(0, 1 − clearance/1.5 m)²;
    - optional (`closing_weight`, default 0): −w·(the rate at which the clearance shrinks)·max(0, 1 − clearance/3 m). This penalizes approaching the nearest obstacle but not flying past it;
    - −0.1 inside canopies;
    - −0.02·‖Δa‖²;
    - −50 on failure.
- **Tests**:
  - Rust `goals_advance_within_the_radius`: events, `goal_index`, nothing after finishing, radius 0 never advances.
  - Python, using a scripted goal-seeking pilot:
    - it finishes all three waypoints on open ground;
    - it trips the height limit when told to climb;
    - `evaluate` reports success 1.0 for it;
    - hover tasks report no success key.
  - The clearance column is checked against the map query.
- **Training** (PPO defaults except 2×256 networks and 6 torch threads; laptop). Success is measured on the unseen map pool (`map_seed` 1000) with the deterministic policy, on two sets of episodes: the run's own evaluation (seeds 1000 + i) and a second set of 256 (seeds 5000 + i). Failures are from the second set.

  | Run | Changes | Wall time | Success (set 1 / set 2) | Failures |
  |---|---|---|---|---|
  | try1, 10M | `rl64` LiDAR (4 × 16), failure penalty 25, 3 torch threads | 17 min (9.7k SPS) | 69 % (64 episodes) / 64 % | obstacle 23 %, terrain 5 %, height/edge 5 %, timeout 4 % |
  | try2, 10M | 4 × 32 LiDAR, failure penalty 50 (both now defaults) | 16 min (10.7k SPS) | 77 % (128) / 81 % | obstacle 12.5 %, height/edge 3 %, timeout 2 %, terrain 1 % |
  | long, 30M | as try2 | 46 min (10.8k SPS) | 78.5 % (256) / 83 % | obstacle 11 %, terrain 2 %, height/edge 2 %, timeout 2 % |
  | γ = 0.995, 10M | an 8 s horizon at 25 Hz instead of 4 s | 15 min | 73 % (256) | 23 % failed, 5 % timeouts |
  | closing, 10M | `closing_weight` 1 | 15 min | 74 % (256) | 16 % failed, 9 % timeouts |
  | no foliage, 10M | the LiDAR passes through canopies | 16 min | 71 % (256) | 25 % failed |
  | meadow 50, 10M | trained on maps with 50 instead of 6 trees/ha on grass | 15 min | 77 % (256) | 19 % failed, 5 % timeouts |
  | bounds, 10M | `--bound-coef 0.01` | 15 min | 81 % (256) / 78.5 % | obstacle 16 %, height/edge 3.5 %, water 2 % |
  | **bounds, 30M** | `--bound-coef 0.01` | 45 min (11.2k SPS) | **88.1 % (512) / 88.3 %** | obstacle 8.6 %, terrain 2 %, height/edge 1 %, timeout 0.4 % |

  Runs with the same settings and budget differ by about ±4 % in success, so the 10M rows between 71 % and 77 % are not clearly different from each other.

- **Findings**:
  - **Open ground versus forest.** Episodes can be grouped by the median clearance along the path they flew:
    - above 3.5 m (open ground at about 4 m AGL; 80 % of episodes): 88–96 % success;
    - 2.5–3.5 m (through forest): 33–50 %;
    - below 2.5 m: almost never.

    So the overall rate mostly measures open terrain. Flying through dense forest (350 trees/ha, about 5 m between trunks) is the weak skill. The final policy reaches 98 % on open paths and 63 % on forest paths.
  - **Failure mode.** The policies fly close to full speed everywhere: 4.6 m/s commanded with an obstacle within 1.5 m, against 4.8 m/s in the open. Before an obstacle crash the vehicle heads at the goal (cos ≈ 0.85) at full command with 2–3 m clearance. The LiDAR sees the trunk: its minimum range matches the ground-truth clearance. About 0.5 s before impact, at ≈ 2 m, the vehicle can no longer brake: stopping from 4.7 m/s takes about 2 m at the 5 m/s² limit, plus the reference lag.
  - **What helped:**
    - The denser scan (at 5 m, 16 beams per ring leave 2 m gaps that trunks of 0.4–1.3 m fit into) and a failure penalty large enough to make slowing down pay together halved obstacle crashes (try1 → try2).
    - More steps (10M → 30M).
    - The bounds loss. For the 30M try2-style policy, the mean horizontal action lies outside the action box in 92 % of steps (p90 |mean| 2.6), with an exploration std of 0.25. Nearly every sample is clipped to the same full-speed command, so PPO cannot see what slowing down would earn. With `--bound-coef 0.01` the means stay closer (p90 1.55), and 10M steps match the 30M baseline. With 30M steps it reaches 88 % (training success 84–86 % from 13M steps on, against 72–78 % without the loss), and forest success roughly doubles.
  - **What did not help at 10M:**
    - γ = 0.995: slower flight and more timeouts.
    - The closing-speed penalty: fewer crashes, but more timeouts.
    - A LiDAR that ignores foliage: canopy returns are not what hides the trunks.
    - Training on maps with more scattered trees: no gain in forest.
  - **A faster velocity reference** (`reference_ratio` 1.0) would react sooner, but random actions then tilt the vehicle to p99 55° (max 78°), against 25° at 0.5. It was not used.
  - **Torch threads:** 6 is best on this laptop with 9 simulation threads: 19k SPS at the start of training, against 15k with 3 or 10. The simulation and torch alternate, so torch can use cores that would otherwise sit idle. With 10 threads, OpenMP spin-waiting slows the simulation (41 % of the time). The 128-beam LiDAR raises the simulation's share from 15 % to 25 %.
  - **Ideas for forest flight** (not done):
    - a squashed (tanh) Gaussian policy, which keeps exploration inside the action box;
    - stacked scans or a recurrent policy, so trunks between beams are remembered;
    - a curriculum on forest density, or goals placed inside forests;
    - a speed limit that depends on the clearance.
- **End-to-end demo** (the result above; `runs/QuadWaypointForest-v0__ppo__1__30M` holds the policy, and `runs/` and `recordings/` stay out of version control):
  ```
  uv run python examples/ppo_continuous.py --env-id autonomousim/QuadWaypointForest-v0 --total-timesteps 30000000 \
      --hidden 256 --torch-threads 6 --bound-coef 0.01 --eval-env-kwargs '{"map_seed": 1000}' --eval-episodes 512
  uv run python examples/eval_record.py runs/<run>/policy.pt --episodes 6 --lidar --env-kwargs '{"map_seed": 1000}'
  cargo run -p autonomousim-viewer --release -- replay recordings/<run>.mcap
  ```
  - All 6 recorded episodes on unseen maps succeed.
  - The file is 2.3 MiB for 147 s (15.8 KiB/s with the LiDAR scans).
  - Read-back and re-simulation from `/meta` match bit for bit.
  - The viewer rebuilds the unseen map and plays the flights with trails and LiDAR hits.

### Viewer (Bevy 0.19.1 + bevy_egui 0.42)
**As built in step 10a** (`autonomousim-viewer`, live mode only):
- **Build**: Bevy with `default-features = false` and only the features in use: X11 through winit (no Wayland or audio backends, and the gamepad backend is opt-in, so no system libraries or pkg-config are needed; it runs under XWayland), PBR, gizmos, `tonemapping_luts`, `png` for screenshots. bevy_egui with `render` + `default_fonts`. The viewer is excluded from `default-members`; `make test-viewer` runs its headless tests.
- **Simulation**: a `Sim` resource owns a `WorldInstance` and steps it at the physics rate from a fixed-step accumulator. At most 0.1 s of simulated time is added per frame, so slow frames slow the simulation instead of freezing it. The time scale runs from 0.125× to 4×; pause is available. Vehicle poses are interpolated between the last two ticks. Events are latched per episode for the HUD. A non-finite state resets the episode.
- **Pilot**: the first agent flies in `velocity` mode with a heading-frame velocity setpoint and a yaw-rate command. Keys: W/S forward/back, A/D left/right, Space/Shift up/down, Q/E yaw, `-`/`=` maximum speed (1–40 m/s, default 8), R reset episode, P pause, `[`/`]` time scale, C camera, H HUD, F1 help, Esc quit.
- **Scene** (`scene` crate, renderer-independent, f32, ENU/FLU coordinates, linear vertex colours):
  - `MeshData` with primitives: cone, cylinder, open tube, cuboid, icosphere and convex hull.
  - Terrain chunks of 64 × 64 cells at a chosen stride. They follow the grid's triangulation exactly, take their colours from the averaged material colours and have outward-facing skirts. There is one water quad per run of equally high wet cells.
  - Obstacles are merged per terrain chunk, with seeded brightness variation and tag colours for trunks, needle crowns and broadleaf crowns.
  - The procedural multirotor has a hub, a front marker, arms (red in front) and motor cans, plus rotor discs sized from the definition.
  - The viewer converts `MeshData` to Bevy meshes with the ENU→Bevy basis change. That change is a proper rotation, so triangle winding is kept. Meshes are uploaded as render-world-only assets, with explicit AABBs.
- **Level of detail**: the 64 m chunks use strides 1/2/4/8 by distance to the chunk box: below 160 m, below 380 m, below 800 m, and beyond. There is 15 % hysteresis. At most 24 chunks are rebuilt per frame, nearest first, in parallel with rayon. Chunks and their props beyond the view distance are hidden.
- **Lighting**: one sun at 40° elevation with 2 shadow cascades out to 150 m, plus ambient light. Linear distance fog runs from 30 % of the view distance to the full view distance, in the sky colour. Water is alpha-blended and casts no shadows. Rotor discs are unlit and translucent, blue for CCW and orange for CW, and grow more opaque with rotor speed.
- **Quality presets** (as changed in 10b):
  | Preset | Shadows | MSAA | View distance | Coarse obstacles beyond |
  |---|---|---|---|---|
  | `low` | no | off | 500 m | 250 m |
  | `medium` | 1 cascade to 100 m | 2× | 900 m | 400 m |
  | `high` | 2 cascades to 150 m | 4× | 1600 m | 700 m |
- **Cameras**:
  - **Chase** (default): swings behind the heading with τ = 0.5 s.
  - **Orbit**: controlled by the mouse.
  - **First person**: fixed to the airframe, tilted up 10°.
  - Mouse drag turns the camera and the wheel zooms. A ray cast against terrain, trunks and crowns pulls the camera in front of anything between it and the vehicle. The camera also stays above the ground and water.
- **HUD** (egui): shows the map, seed and pool size; vehicle and episode; time, position, AGL, speed, climb, heading, command and rotor speeds; latched events (red when terminal); run state, time scale, real-time factor, fps and camera; and the key help.
- **Options**:
  - `--preset/--seed/--size` or `--scenario` choose the map; `--vehicle`, `--wind`, `--episode-seed`, `--no-cache` and `--quality` are also available.
  - `--window WxH` sets the size in physical pixels; the default is 1600×900 logical.
  - `--demo` flies forward on its own at 8 m/s and 40 m AGL, turning slowly.
  - `--screenshot PATH --frames N` saves a PNG and logs the fps before exiting: since 10b, the mean after a 120-frame warm-up and the average of the last frames.
  - `--no-vsync`. GNOME throttles windows it considers hidden to about 1 fps under vsync, so screenshot runs always render without vsync.

**As built in step 10b** (`live` and `replay` modes):
- **Command line**: `autonomousim-viewer [live] [map options]` (live is the default, so the step-10a options still work) or `autonomousim-viewer replay FILE [--episode N] [--once]`. Both take the display options `--quality`, `--window`, `--no-vsync`, `--plots`, `--lidar-view`, `--screenshot` and `--frames`; live options are rejected after `replay`.
- **Replay** (`replay.rs`, with the typed reader `sim::record::Recording`):
  - The file is parsed into episodes with per-agent states, actions, LiDAR scans and events. The recorded scenario is compiled again, and the map hashes must match the recorded ones.
  - The agents of that world act as puppets. Each frame, the state at the playback time (position lerp, orientation slerp, rotor speeds) is written into the vehicles with `reset` and `set_motor_speeds`, together with events, the disabled flag, the goal list and the current goal. Cameras, vehicle visuals, the HUD and the overlays therefore work unchanged.
  - A different map index switches the world's map (`WorldInstance::set_map`), and `sync_map` rebuilds the scene.
  - Controls: P play/pause, `[`/`]` speed (0.125–8×), ←/→ ±1 s, Shift+←/→ one state sample, N/B (PageDown/PageUp) next/previous episode, Home or R start of the episode. The playback loops over all episodes unless `--once` is given.
  - A timeline window has previous/play/next buttons, an episode list with durations, the speed, a loop switch and a scrubbing slider.
  - Checked on `recordings/QuadHover-v0__ppo__1__10M.mcap` (5 episodes).
- **Pilot modes** (M cycles; also shown in the HUD):
  - **Velocity**: the step-10a mode.
  - **Attitude**: the sticks set a tilt of up to 25° as a rotation vector in the heading frame. Q/E sets the yaw rate. Collective thrust is hover × (1 + 0.5·up), divided by the cosine of the tilt (up to 2×).
  - **Rates** (acro): up to 3 rad/s roll and pitch, with the same thrust.
  - Tests fly each mode forward without a crash.
- **Gamepad**: mode-2 layout. Left stick: climb and yaw. Right stick: forward and sideways. Start: pause. Select: pilot mode. East (B): reset. Bevy's `gamepad` input feature is always on, so this code is compiled and linted. The gilrs backend sits behind the viewer feature `gamepad` (`cargo run -p autonomousim-viewer --release --features gamepad`) because it needs `libudev-dev` and `pkg-config`, which this laptop does not have installed (installing them needs sudo).
- **Free camera** (fourth mode on C): starts where the previous camera looked. W/A/S/D move it, Space/Shift raise and lower it, the mouse drag looks around, the wheel sets the speed (1–200 m/s) and Ctrl moves 4× faster. The camera stays above the ground. The flight keys control the camera while it is active, so the pilot's sticks are zeroed.
- **Overlays** (gizmos; O toggles goals and trails, L toggles LiDAR):
  - The current goal is a sphere sized to the vehicle, and the later goals are fainter and joined in order. A line runs from each vehicle to its goal.
  - The trail covers the last 10 s: in live mode from the drawn poses, in replay from the recorded states. The followed agent's trail is pink.
  - The followed agent's latest LiDAR hits are drawn as crosses coloured by range: live from its sensor, and in replay from the recorded scans with the beam directions of the rebuilt sensor.
- **LiDAR view** (V or `--lidar-view`; `lidar_view.rs`): a 2D window with the followed agent's latest scan, the same one as the hits drawn in the scene (`overlay::latest_scan`, live or replayed).
  - Header: layout, range and rate; returns out of beams and the nearest return.
  - **Top view** around the sensor, heading up (rotated by yaw only, so it does not tilt with the vehicle), with range rings and the current goal (on the edge when out of range).
  - **Range image** for ring patterns: one row per ring (highest on top), one column per azimuth, ahead in the middle and left on the left, marked behind / left / ahead / right. Colours as in the scene (red near, through yellow, to green far); grey means no return. Dense patterns share one cell per point across, which shows the nearest return (VLP-16-like: 1,800 azimuths in 260 cells).
  - The image is a mesh of coloured cells. A texture updated with each scan flickered, because bevy_egui makes every full texture update a new image asset, which is not yet on the GPU in the frame it is created.
  - Tests: range-image layout checked against the beam directions (full turn and partial field of view), and a live scan over flat ground (every return lies on the ground; rings above the horizon see nothing).
- **Plots** (G or `--plots`; egui_plot 0.37):
  - Height AGL and distance to the goal over time.
  - The setpoint (dashed) against the measured value (solid) at the level where the setpoint enters the cascade: body rates, tilt, velocity in the setpoint's frame, position, or rotor speeds.
  - Live, the followed agent is sampled every frame over the last 30 s. A new episode, map, agent or kind of setpoint starts the plots over.
  - In replay, the curves cover the whole episode with a cursor at the playback time. Setpoints are rebuilt from the recorded actions through the group's action map.
- **Map seed control**: in live mode with a generated map, the HUD has a seed field and a Generate button. The new maps are compiled on a background thread and swapped in when ready, starting a new episode. A failure is shown in the HUD.
- **Map switching**: `spawn_lights` runs once. Every map entity carries `MapEntity`, and `sync_map` rebuilds all of them when the simulation's map is no longer the one shown. That happens on a regenerated map, on a replayed episode from another map of the pool, or when a live reset draws another pool map.
- **Far obstacle LOD**: each chunk has a second obstacle mesh with `PropDetail::far()`. It uses 4 segments for cones and cylinders, 3 for trunks and a bare icosahedron for crowns, and leaves out obstacles smaller than 1.5 m. It replaces the detailed mesh beyond 400 m (250 m on low, 700 m on high). On the showcase map this is 0.87M triangles against 2.7M. Billboards were not needed.
- **HUD**: time and AGL are correct in both modes (`agl_now`), plus the goal distance and index, and the pilot mode with its stick values. The status line reads playing, running or paused. The key help changes with the mode.
- **Tests** (`make test-viewer`, headless): playback interpolation, episodes and puppet state; the pilot modes; the tracking quantities; command-line parsing; and a regenerated map that is swapped in and rebuilt as entities (the scene rebuild is run as a system in a bare ECS world).

**As built after M1: policy playback** (M1 stretch item; `sim::policy`, `examples/export_policy.py`, the viewer's `policy` command):
- **Export**: `uv run python examples/export_policy.py runs/<run>/policy.pt` writes `policy.json` next to the checkpoint (2.3 MB for the 2×256 forest policy). Since M4c the training scripts (`ppo_continuous.py`, `sac_continuous.py`, `ppo_multiagent.py`) export automatically after their final save (`--no-export` skips it; the multi-agent script writes `policy_<group>.json` per group, and `policy.json` with one group). It holds:
  - the task's scenario (`task.scenario()`), the agent group and the episode time;
  - the observation normalisation (mean, variance, clip, eps);
  - the actor as dense layers (row-major f32 weights, bias, activation) and the output: PPO's action mean clipped to [−1, 1], or SAC's tanh of the mean;
  - 40 observations from the first second of 8 episodes, with the deterministic actions PyTorch computed for them.
- **`sim::policy`**: `PolicyFile` reads and checks the format. `PolicyFile::policy()` builds a `Policy` and runs it on the stored observations; a difference above 1e-4 from PyTorch rejects the file, so a misread layout fails at load time instead of flying badly. `Policy::act` normalises in f64 like `ObsNormalizer`, then runs the f32 layers without allocating.
- **Viewer**: `autonomousim-viewer policy FILE [--map-seed N (1000)] [--agents N] [--episode-seed N] [--no-cache]`, plus the display options. The command runs the policy's scenario on one map of the given seed; the HUD seed control generates others.
  - An `Autopilot` in `Sim` observes the group and sets every agent's action at each policy-step boundary (`tick % decimation == 0`), so the timing matches `WorldInstance::step`.
  - T takes over the followed agent, which the keys then fly in the usual pilot modes; the policy keeps flying the others.
  - While the policy flies every agent, the next episode starts 1.5 s after all of them ended theirs (terminal, disabled or finished) or the task's time ran out. The HUD counts the episodes that reached the last goal and shows the followed agent's action.
- **Check against Python**: the ignored test `exported_policy_success_rate` flies 64 episodes on each of 4 unseen maps (1000–1003) with the policy in Rust. The forest policy reached 220 of 256 (86 %) with one drone per world, against 88 % in the Python evaluation on other unseen maps. With 8 drones per world it reached 209 of 256 (82 %), because they now meet each other (they trained alone).
- **Performance**: 4 drones with their LiDAR and the LiDAR view run at 123–149 fps at 1440×810 (medium).
- **Tests**: the numpy forward pass of an exported PPO and SAC network matches the stored actions (Python). In Rust: a hand-computed forward pass, rejected files and shapes, the take-over, and the automatic restart.

**Still to do**: `attach` (M3).

## Verification
| Area | Checks |
|---|---|
| Rigid body | Torque-free motion: energy and angular momentum drift < 1e-9 (RK4) and < 1e-10 (semi-implicit at 2 ms, thanks to the midpoint gyroscopic term). Dzhanibekov flip period within 1 % of the analytic value. Symmetric-top precession. Heavy-top gyroscopic precession on a Spherical joint |
| Multibody | Double pendulum vs closed-form equations (1e-12). proptest: ABA ≡ CRBA⁻¹(τ − RNEA) on random trees. Pinocchio golden fixtures (1e-10): ABA with and without f_ext, RNEA, M, KE and COM on 7 models, including a car-like floating base and non-depth-first trees. Momentum conserved with internal torques only |
| Contact | Rest penetration ≈ mg/k; restitution matches ζ; stick on incline when tanθ < μ (< 1 mm in 10 s); slide acceleration g(sinθ − μcosθ) within 2 % |
| Rotor and controller | Hover gives \|a_z\| < 1e-9. Per-motor signs correct; pure yaw produces no roll/pitch; motor 63 % at τ; GE(z = R) = 1.0667. B·B⁺ = I. Rate step rise < 50 ms, overshoot < 20 %. 10° attitude step settles < 0.3 s. 1 m position step settles < 2 s with zero error under wind. Same gains work for cf2x and iris |
| Sensors | Static IMU reads +g on z; Allan variance recovers noise parameters within 10 %; GPS latency exact in ticks; LiDAR exact on analytic plane and box scenes |
| Procgen | Same seed gives an identical hash (1 vs 12 threads, debug vs release, repeated runs; the desktop still to check); golden hashes committed; invariants hold (no trees in water or on cliffs, lakes flat); all done in step 9 |
| Sim determinism | Bit-identical trajectory hash for the same seed and actions. Env 0 identical at N = 1 and N = 12 and at 1 vs 4 threads; parallel in-world phases identical at 1 vs 6 threads. `reset(seed)` reproduces; snapshot/restore round-trips (all done in step 8) |
| Python API | `gymnasium.utils.env_checker.check_env`; vector tests (shapes, dtypes, spaces, SAME_STEP `final_obs`, seeding, reset masks); wheel smoke-tested on 3.12 and 3.14 |
| End to end | `tests_py/test_rl.py`: the `autonomousim.rl` helpers; short PPO and SAC runs (loop, checkpoint, evaluation), each followed by `eval_record.py` with read-back and bit-exact replay. A learning test would need a minute or more, so convergence is checked manually by full hover training (results in *Tasks and training*). A trained PPO hover policy was recorded with `eval_record.py` and played back in the viewer (10b), and so was the forest-navigation policy, with its LiDAR scans (step 13) |

Commands: `make check` (fmt + clippy), `make test` (`cargo test` + `uv run pytest tests_py`), `make bench` (criterion + `python -m autonomousim.bench`), `make test-viewer`, `cargo run -p autonomousim-viewer --release -- [live] --preset showcase --seed 0`, and `cargo run -p autonomousim-viewer --release -- replay recordings/<run>.mcap`.

### Performance targets (i7-1365U, release build; recalibrate after step 8)
| Metric | Target | Measured 2026-09-23 (powersave) |
|---|---|---|
| One quad physics tick (ABA, rotors, contacts, IMU, controller) | ≤ 1 µs | free-body step 0.23 µs; full multirotor step without controller and IMU 0.34 µs (cf2x and iris_like, airborne over the forest); controller update 0.14 µs (`ctbr`) to 0.26 µs (`velocity`/`position`); controller + multirotor step in the open 0.45 µs |
| ABA, 20-DoF chain | ≤ 3 µs | 2.95 µs |
| Quad contacts (4 feet + 4 guards) | part of the tick | 56 ns in open air; 0.19 µs between trees (canopy candidates, no contact); 0.83 µs landed (4 contacts); 0.93 µs inside a canopy (8 contacts) |
| Terrain height / surface query | ≤ 50 / 100 ns | 9 / 107 ns (exact closest point 0.5 m above ground; far reject 29 ns) |
| `rl64` LiDAR scan | ≤ 70 µs | 28 µs in a 150 trees/ha forest (40 m beams; 0.44 µs per ray); `vlp16_like` 15.6 ms (28 800 beams, 100 m) |
| Other sensors, per measurement | part of the tick (IMU) | IMU 70 ns per tick (ADIS16448; 25 ns ideal); GPS 0.33 µs; baro 0.33 µs; mag 22 ns; rangefinder 0.20 µs; ground truth 0.62 µs (clearance query) |
| One world, one quad, per policy step (10 ticks) | — | 5.6 µs in the open, holding velocity (0.56 µs per tick: controller, air, rotors, contacts, integration, events, shape, observation); 9.5 µs with IMU, GPS, baro, mag, wind and turbulence; 15.2 µs in a forest with IMU and `rl64` LiDAR |
| Swarm of 128 in one world | — | 0.61 ms per policy step (33× real time; in-world parallel phases with 12 threads). Scaling inside a world is weak on this laptop: 1.0 ms in a 1-thread pool, 0.69 ms with 2 threads, 0.58 ms with 4. The tick was restructured from three fork–joins per tick to two (one without sensors); 0.85 ms serial |
| `BatchSim`, N = 256, 10 threads, raw | ≥ 250k env-steps/s (≥ 50k with LiDAR) | 418k env-steps/s raw; 161k in a forest with IMU + `rl64` LiDAR. Threads 1 / 2 / 4 / 8 / 12: 162k / 286k / 269k / 379k / 417k raw (2 P-cores + 8 E-cores at 15 W; stepping allocates nothing, so the limit is the hardware) |
| Through Python `VectorEnv` including task | ≥ 150k env-steps/s | 360–450k env-steps/s with QuadHover, N = 256 and 10 threads, random actions (episodes last about 47 steps, so about 5 worlds reset per step). Per step: native step 380–450 µs, resets 70–100 µs, Python 120–150 µs (task, bookkeeping, copies). N = 1024: about 500k. Measured with `python -m autonomousim.bench`; run-to-run noise is about ±15 % (powersave governor) |
| Full PPO loop | 20–50k SPS | 37–43k SPS on QuadHover (N = 256, 9 sim and 3 torch threads); the simulation takes 12–14 % of it. SAC: 1,100 SPS (16 worlds, 4 updates of 2×128 networks per vector step; 570 at 2×256) |
| Map generation (2 km) | ≤ 15 s cold, ≤ 1 s cached | 1.9 s cold with 12 threads (3.9 s with 1; hydraulic erosion 0.9 s, hydrology 0.3 s), 2.2 s including the cache write; 0.37 s from the cache (20.5 MB file). 512 m training map: 0.1 s |
| Viewer | ≥ 60 fps at 1080p "medium" | Showcase map (2 km, 2.7M obstacle triangles, 0.87M in the far LOD), `--demo` flight at 1920×1080 on the Iris Xe, release, mean over 1,080 frames after a 120-frame warm-up (the climb out of the forest to 40 m AGL), with 30 s cool-down between runs: **80 fps medium** (one shadow cascade to 100 m, 2× MSAA; 80.6 and 79.9 in two runs of the final build), 145 fps low, 59 fps high; the last frames at 40 m AGL run 93 / 154 / 77 fps. Medium at the step-10a settings (two cascades to 150 m, 4× MSAA) measured 60–62 fps: 97 without shadows, 79 without MSAA, 72 with one cascade, 68 with 2× MSAA. The plots cost nothing measurable. Back-to-back runs without cool-down are 10–20 % slower (thermal throttling). Forest scenario (512 m pool maps, two agents, LiDAR) near the ground: 88 fps medium. Rechecked after step 13: 78.5 and 81.7 fps medium. The same `--demo` flight in the forest scenario runs at 95.9 and 96.5 fps, and at 95.4 and 95.3 with the LiDAR view open, so the view costs nothing measurable over the flight. Over the lighter last frames at 40 m AGL it costs about 1 ms per frame (100–102 against 112–119 fps) |

Benchmarking: take the median of 5 runs with the performance governor (the laptop's P/E cores and thermal throttling add noise). Results go to `benchmarks/results/*.json`.

## Milestone 2: Ground vehicles I

Planned 2026-09-24; **done 2026-09-25** (steps 0–9, as built below). Scope from the roadmap: suspension kinematics (`KcTravel` joint), Magic Formula tires with transient slip, steering, powertrain, brakes; Ackermann cars, diff-drive and skid-steer robots; ground action modes; a 1 kHz preset. Two things were decided with the user at the start:
- **Validation oracle**: Project Chrono (PyChrono, conda) in its own environment outside the repo. It is used only to generate committed fixtures, as Pinocchio is for the multibody tests. Analytic checks come on top.
- **Closing demo**: `CarWaypointOffroad-v0`. A 4×4 drives through waypoints over wild-map terrain between the trees, using LiDAR and `(v, κ)` actions.

### Design
- **Agents become vehicle-generic**. The vehicles crate gets `enum Vehicle { Multirotor(Multirotor), Wheeled(Wheeled) }` (a `Tracked` variant follows in M4, so nothing ground-generic may assume wheels) with the shared queries: pose, twist, mass, definition, the step phases, contact colliders, visual state. Controllers become `enum Controller`, and each action mode belongs to a family (`ActionMode::{Motors, Ctbr, …}` for multirotors, `GroundActionMode` for wheeled vehicles). The state row (`STATE_FIELDS`) stays common. Family-specific data goes to recordings and observation terms, never into the row.
  - **Bit-exactness**: multirotor trajectories, the golden hashes and the M1 recordings must not change. The determinism suite runs before and after the refactor.
- **Multibody additions** (`core`):
  - Joints whose motion subspace depends on `q`. The bias `c_J = Ṡ(q)·q̇` enters ABA, RNEA and CRBA.
  - `JointType::KcTravel(Arc<KcTable>)`: one DoF (wheel travel `s`). Tables over `s` give the wheel-carrier pose relative to the design position: vertical, lateral and longitudinal offsets, toe, camber and caster. Cubic Hermite interpolation gives a smooth `S(s)` and `Ṡ`. A straight vertical table reduces it to a prismatic joint; a test checks this.
  - **Auxiliary states** in the integrator (motor, engine and tire relaxation states) are stepped together with `(q, v)`, as the M1 plan intended. Stiff aux states use exact exponential or implicit updates.
- **Wheeled vehicle** (`vehicles::ground`):
  - **Tree**: chassis (Free) → one `KcTravel` per corner → steering `Revolute` about the kingpin axis (prescribed on steered wheels) → wheel spin `Revolute`. A 4-wheel car has 6 + 4 + 2 + 4 = 16 DoF, which fits `MbState`'s inline storage.
  - **Force elements**: coil spring (preload, progressive rate) with bump and rebound stops, a damper with bilinear or tabulated force-velocity, an anti-roll bar per axle (torsion over the travel difference), and chassis aerodynamic drag.
  - **Steering**: the rack position is limited in rate and travel. Each wheel's angle comes from a table over the rack position (Ackermann geometry, a percentage-Ackermann parameter). The angles are prescribed joint trajectories (hybrid dynamics), and the rack force is reported.
  - **Brakes**: maximum torque per wheel with a front/rear bias. Friction is regularised (`T = −T_max·clamp(ω/ω_ε, ±1)`, with `ω_ε` chosen from `I/dt`), so a braked car holds on slopes without creep. The parking brake uses the same model.
  - **Powertrain**:
    - `Combustion`: engine torque over rpm at full and zero throttle, interpolated in throttle; engine inertia and friction; a clutch for launch, or a torque converter table (capacity factor and torque ratio over speed ratio); a gearbox with ratios, efficiency, an automatic shift schedule with hysteresis and a shift time.
    - `Electric`: torque and power limits and a motor time constant, per axle or per wheel (robots).
  - **Differentials**: open, limited-slip (preload, torque bias ratio, viscous), or locked. A locked or LSD coupling is a stiff torsional spring-damper between the output shafts, updated implicitly. Its time constant is well under the tick and it adds no stiffness problem for the tree.
  - **Robots**: diff-drive (two driven wheels plus frictionless caster spheres through the existing penalty contacts) and skid-steer (four wheels, left and right driven sides). Each driven wheel has an electric drive.
  - **Chassis collisions**: sphere colliders on the body through the existing contact code. A chassis hit on terrain or an obstacle raises the crash events. Full shapes come in M3.
- **Tires** (`vehicles::ground::tire`):
  - `enum TireModel { MagicFormula(MfTire), Fiala(FialaTire) }`, read from `.tir` files or TOML.
  - **MF 6.1/6.2** following Pacejka (2012), ch. 4: pure and combined slip Fx, Fy, Mz, plus Mx and My. Dependence on load, inflation pressure and camber. User scaling factors λ.
  - **Friction per surface**: the material table scales λ_μ as the material's μ over the reference μ (asphalt). The rolling-resistance column scales My.
  - **Transient slip**: first-order relaxation lengths σ_κ and σ_α (MF 6.x), with the MF 6.1 low-speed damping below `v_low`. A parked car on a slope therefore holds without drift and without oscillation at 1 kHz.
  - **Contact**: a single contact point on the local road plane, fitted from 4 terrain samples around the patch (the envelope approach of Chrono and MF). Vertical force comes from the tire's stiffness and damping over the loaded radius (MF vertical stiffness with `Fz ≥ 0`). The effective rolling radius follows MF. In water, the tire gets drag and loses friction.
  - **Fiala** (brush model) for small robot tires that have no MF data.
- **Parameter data**: Chrono's data directory (BSD-3) has `.tir` files and complete reference vehicles (Sedan, HMMWV). The M2 presets reuse their published numbers with attribution:
  - `sedan_like`: the on-road reference for the ISO tests (Chrono's Sedan turned out to be rear-wheel drive; see step 3).
  - `offroad_4x4`: HMMWV-like, the demo vehicle.
  - `rover_diff`: a small diff-drive robot, about 17 kg, Clearpath-Jackal-like.
  - `rover_skid`: a four-wheel skid-steer robot, about 50 kg, Husky-like.
- **Ground control and action modes** (`control::ground`), all normalised to [−1, 1]:
  - `raw`: throttle/brake on one axis, and steering (diff-drive: left/right wheel torque).
  - `vk` (cars): speed and path curvature. A speed PI drives throttle and brake. Curvature becomes steering through the kinematic bicycle model plus yaw-rate feedback.
  - `vw` (robots): speed and yaw rate through per-side wheel-speed PI loops.
  - `per_wheel` (added 2026-09-24): every wheel commanded separately, for torque vectoring, brake-based yaw control, crab and independent four-wheel steering, and swerve-like robots.
    - `DriveInput` gets optional per-wheel channels that replace the mixed commands when present:
      - `wheel_drive[w]`: the command of the motor driving wheel `w`, for electric drives. A motor that drives several wheels takes the mean of their commands. Combustion drives keep the single throttle.
      - `wheel_brake[w]`: pedal fraction per wheel.
      - `wheel_steer[w]`: angle as a fraction of the axle's lock.
    - **Independent steering**: a new axle option `steer_mode = "independent"` (default `"ackermann"`) with its own `max_angle`. It gives the axle's wheels knuckles that follow their own commands (rate-limited, prescribed as now) instead of the Ackermann blend.
    - **Action vector**: only the channels the vehicle has (motors, brakes, independently steered wheels), in wheel order, normalised to [−1, 1]. `action_space` names each channel.
  - Gains come from the definition, as for multirotors.
- **Simulation**:
  - The 1 kHz physics preset for any world with ground vehicles. The same policy rate options apply.
  - **Ground spawning and goals**: on the surface, with a maximum slope, clear of obstacles by the vehicle's radius, and on dry land. Goals are checked for reachability: a coarse grid A* over drivable cells (slope, water, trunks) confirms a path exists.
  - **Events**: `ROLLOVER` (tilt past a limit); crashes from chassis contact; `WATER` when water reaches the chassis; `STUCK` when the vehicle has barely moved for N seconds (not terminal by itself). New observation terms: speed and slip, wheel speeds, steering angle, pitch and roll, gear and rpm.
  - A `wild` preset `offroad`: gentler relief and sparser forest with clearings. It gets new golden hashes.
- **Viewer**:
  - Car and robot visuals: chassis, wheels that spin, steer and travel, and suspension links.
  - Driving keys: W/S throttle and brake, A/D steering, Space handbrake. A gamepad works as well.
  - A chase camera suited to cars.
  - HUD: speed, rpm, gear, steering, per-wheel load and slip. Plots: slip against force, and yaw rate against its reference.
  - Replay of ground recordings.

### Implementation order
| # | Step | Done when |
|---|---|---|
| 0 | Chrono oracle: micromamba env with PyChrono (outside the repo), `tools/gen_chrono_fixtures.py`, `make fixtures-chrono`; pick the `.tir` files and reference vehicle data | A Chrono MF tire test-rig sweep and a Sedan step-steer run write JSON fixtures |
| 1 | `core`: q-dependent joints, `KcTravel`, aux states in the integrator | ABA ≡ CRBA⁻¹(τ − RNEA) with KcTravel (proptest); finite-difference `S`/`Ṡ`; energy conserved; prismatic equivalence |
| 2 | Tires: `.tir` parser, MF 6.1/6.2 steady state, transient slip and low-speed damping, road-plane contact, Fiala | Fx/Fy/Mz sweeps (pure and combined, several loads and cambers) match the Chrono fixtures within 1 % of peak; a parked tire on a 20° slope holds; free rolling decays to rolling resistance |
| 3 | `vehicles::ground`: definitions (TOML), suspension, steering, brakes, powertrain, differentials, the four presets | Static ride heights and wheel loads match the definitions; a braked car holds on a 30 % slope; a straight 0–100 km/h run and a coast-down are plausible against Chrono |
| 4 | `control::ground` + action modes, including `per_wheel` | Speed steps settle without overshoot beyond 10 %; curvature tracking on a circle; diff-drive `vw` tracking. `per_wheel`: on a four-motor car, a left/right torque difference gives the yaw moment and sign that statics predict; braking one wheel yaws toward it; independent four-wheel steering at 30° crabs with yaw rate < 0.01 rad/s, and counter-phase steering beats the front-steer turn radius as the bicycle model predicts; unused channels are absent from the action space |
| 5 | Generalise agents (`Vehicle`/`Controller` enums, family-scoped action modes, recording and observation plumbing), done once the wheeled vehicle exists so that both variants are exercised | All M1 tests pass; golden hashes, trajectory hashes and old recordings unchanged; no throughput loss beyond 3 % |
| 6 | `sim` integration: ground groups, 1 kHz, spawning and reachable goals, events, observation terms, `offroad` preset, BatchSim, recording | Determinism suite with cars; a batch of 64 cars drives on a wild map; benchmarks recorded |
| 7 | Validation suite: ISO 4138 constant radius, ISO 7401 step steer, straight braking, ISO 3888-1 double lane change (path-following driver) | Understeer gradient and yaw-rate response within 10 % of Chrono (Sedan) and of the linear bicycle model at low lateral acceleration; braking distance within 5 % of Chrono |
| 8 | Viewer: ground visuals, driving, HUD, plots, replay | Drive the 4×4 through a wild map at ≥ 60 fps medium; replay a recorded drive |
| 9 | Python: ground tasks and throughput; `CarWaypointOffroad-v0`; training | > 80 % success on unseen maps; exported policy drives in the viewer |

**As built in step 0 (Chrono environment)**:
- micromamba 2.9 in `~/.local/bin`, environment `chrono` (`MAMBA_ROOT_PREFIX=~/.local/share/micromamba`) with Python 3.12 and PyChrono **10.0.0** from the `projectchrono` channel. It takes 5.7 GB after `micromamba clean -a`.
- Chrono 10 has no MF 6.x tire. Its Magic Formula tire is `ChPac02Tire` (MF 5.2 equations, `.tir` input), next to TMeasy, Fiala and Pac89. All 15 `.tir` files in its data directory are MF 5.x. Among them: `hmmwv/tire/HMMWV_Pac02Tire.tir` (the demo 4×4), `sedan/tire/Sedan_Pac02Tire.tir`, and a Goodyear 335/65R22.5 fitted at four pressures.
- Consequence for step 2: the tire code implements MF 5.2 and MF 6.1 behind the `.tir` version, as MFeval does. MF 5.2 is checked against `ChPac02Tire`. The 6.1-only terms (pressure, some camber and turn-slip terms) are checked analytically and by reduction to 5.2.
- The fixture generator `tools/gen_chrono_fixtures.py` (`make fixtures-chrono`) came with step 2 (tyre sweeps); the Sedan step-steer run follows in step 7.

**As built in step 1** (`core`):
- `math::spline::CubicSpline`: a natural cubic spline, continued linearly past the end knots, returning the value and both derivatives.
- `dynamics::KcTable` + `JointType::KcTravel(Arc<KcTable>)`:
  - The carrier sits at `p(s)` with `R(s) = R_z(toe) R_x(camber)`. Curves are given on travel knots; an empty `z` means `z = s`.
  - `S(s)` and `dS/ds` are analytic (formulas in `kc.rs`) and checked against central differences of the transform.
  - Forward kinematics stores the column in `KinCache::s` and adds `c_J = dS/ds·ṡ²` to `c`. ABA, RNEA and CRBA take subspace columns through `KinCache::joint_col` and `joint_motion`. Joints with a constant subspace use exactly the code paths they used before, so the M1 hashes are unchanged.
- **Tests**:
  - The random-tree ABA ≡ CRBA⁻¹(τ − RNEA) and RNEA∘ABA = id checks now include `KcTravel`.
  - A vertical table equals a prismatic joint (to 1e-12).
  - A chain of two `KcTravel` joints and a revolute conserves energy under gravity to 3.5e-9 relative with RK4 at 25 µs. Without `c_J` the same run diverges.
- **Auxiliary states** (motor, engine, rack, tire relaxation) stay in the vehicle models, which step them after the multibody update, as the multirotor does with its motors. A generic aux vector in the core integrator was not needed.


**As built in step 2** (`vehicles::ground::tire`):
- **`.tir` reader** (`tir.rs`): sections, `$`/`!` comments, quoted strings, Fortran exponents; SI units only (anything else is rejected, not converted).
- **Magic Formula** (`mf.rs`): MF 5.2/PAC2002 (FITTYP 5, 6, 21) and MF 6.1/6.2 (61, 62) behind `MfVersion`, following MFeval's equations and corrections: pure and combined Fx, Fy, Mz, plus Mx, My, relaxation lengths, effective rolling radius, vertical force and contact length. Turn slip and MFeval's input limits are left out. The MF 5.2 camber scalings LGAX/LGAY/LGAZ/LKG must be 1.
- **Oracles** (outside the repo, fixtures committed):
  - **MFeval.jl** (MIT; `~/.local/share/autonomousim-oracles/MFeval_julia` with the official Julia 1.13 tarball, about 1.1 GB plus 130 MB in `~/.julia`; `make fixtures-mfeval`). useMode 221, 682 points per tyre (random load, slip, angle, camber, pressure, plus sweeps), on the MF 5.2 and 6.1 sample tyres and the two presets. The oracle reads canonical `.tir` files written by our parser (`examples/tir_canonical.rs`), so its defaults for missing keys play no part. All outputs agree to about 1e-15 of peak (trail and Mz to 1e-7, from MFeval's ε in cos α′). Re and contact length are skipped for files with LCZ ≠ 1, which MFeval ignores.
  - **Chrono `ChPac02Tire`** (`make fixtures-chrono`): pure κ, pure α and combined sweeps at 0.5, 1 and 1.5 × F_z0 on the HMMWV and Sedan files, evaluated at Chrono's own slip quantities. Agreement is 1e-6 to 1e-5 of peak (the test bounds it at 1e-4; the plan asked for 1 %) wherever Chrono's pure-slip curves are not clamped. Chrono's deviations from MF 5.2, documented in the generator: γ is never passed on; B·x is clamped to ±(π/2 − 0.01); the trail lacks the LFZO factor; +0.1 in the stiffness denominators; its combined mode takes the equivalent trail angle's sign from κ. The sweeps therefore use USE_MODE 3 and USE_MODE 4 with FE_METHOD 'NO'.
- **Fiala** (`fiala.rs`, TOML): an isotropic brush with a parabolic pressure distribution under combined slip, `F = μF_z(3z − 3z² + z³)` along the slip demand, with trail `(a/3)(1 − z)³/(1 − z + z²/3)`. Unit tests check the slip and cornering stiffness, saturation, the friction circle and the brush aligning moment.
- **Road contact** (`road.rs`): terrain samples ahead, behind and to both sides (±0.3 R, ± half width) define the road plane; the contact point is where the wheel plane meets it; the loaded radius is measured in the wheel plane.
- **Force element** (`model.rs`, `Tire::step`):
  - F_z comes from MF vertical stiffness (plus MF 6.x bottoming) or linear stiffness, with damping, and F_z ≥ 0.
  - Transient slip uses carcass deflections `u` and `v` (Pacejka §7.2). Their decay term is implicit, and the relaxation lengths lag by one tick.
  - The MF 6.1 low-speed damping `k_Vlow` acts below VXLOW. By default `k_Vlow0` gives damping ratio 0.25 for the nominal corner mass on the standstill carcass spring, which keeps the wheel-spin mode stable at 1 ms explicit steps.
  - Surface friction scales λ_μ as material μ / 0.8 (asphalt). Rolling resistance scales My by the material coefficient / 0.013. Files without QSY coefficients (the Sedan) use 0.013 · F_z · R0.
  - My changes sign with the wheel's rolling direction, smoothed over ±0.05 m/s.
- **Dynamic tests** (`tests/tire_dynamics.rs`, one corner mass with a wheel; Sedan, HMMWV and a robot Fiala tyre):
  - Transient slip settles on the steady state (1e-9).
  - Side-slip relaxation is first order in distance, independent of speed (3 and 25 m/s).
  - A braked wheel on a 20° slope, facing uphill or across it, creeps 0.3–1.2 µm over 8 s at speeds below 1.1e-4 m/s.
  - Free rolling decelerates exactly at the rolling-resistance rate.
- **Performance** (`benches/tire.rs`):
  - MF eval: 148 ns for the Sedan file, 232 ns for the HMMWV file. The HMMWV file uses every term group (curvature E, RVY, REX/REY), which costs about 31 libm calls. Exact shortcuts for zero coefficients (E = 0, SV_yκ = 0, Mx groups) and `cos∘atan`, `sin(2 atan)` identities brought these down from 245 and 300 ns. Cheaper approximations would give up the exact oracle match, so they were not used.
  - Full tyre step on a height grid (road-plane fit, load, transient slip, MF, wrench): 336 and 425 ns.

**As built in step 3** (`vehicles::ground::{def, powertrain, wheeled}`, `assets/vehicles/*.toml`):
- **Definition** (`type = "wheeled"`, `WheeledDef`):
  - The sprung chassis (mass, COM, inertia, drag area per axis).
  - 1–4 axles, each listing its left wheel; the right wheel mirrors y, toe and camber. Wheels are numbered `2·axle + side`.
  - Optional steering (bicycle angle at full lock, rate limit, Ackermann fraction); a powertrain; sphere colliders (`body` or frictionless `skid` casters).
  - Suspension per axle, or none (rigid robots):
    - a `KcTravel` table (spindle x/y offsets, toe, camber over travel) and a carrier mass;
    - a spring as a rate plus preload or as a force table;
    - bilinear damper, linear bump and rebound stops, an anti-roll rate.
  - Tyres are a `.tir` file (built-in names or a path) or Fiala parameters.
- **Automatic preload**: when a rate spring has no preload, `finish()` solves the two-axle statics and preloads each spring so it carries its static load at zero travel. The design positions are then the static ride height.
- **Static solver**: `static_state(g)` solves the statics; it iterates the attitude against the loads by levers, the tyre deflections, the travels and a Gauss–Newton fit of height, pitch and roll.
- **Tree**: chassis (Free) → carrier (`KcTravel`, if sprung) → massless knuckle (Revolute z, prescribed, if steered) → wheel (Revolute y). The Sedan has 16 DoF: 6 + 4 + 2 + 4, with the 2 knuckle joints prescribed.
- **Steering**: the rate-limited bicycle angle becomes per-wheel angles by Ackermann blending about the mean unsteered axle. The knuckle joints follow as prescribed trajectories with `q̈ = ((target − q)/dt − q̇)/dt`, and their torques are reported.
- **Brakes, locked and limited-slip differentials, chain drives**: all are one torsional "bristle" coupling. It is a spring–damper on the integrated relative rotation (stiffness and damping from the coupled inertia and dt: ω = 0.3/dt, ζ = 0.7), with its torque capped at the capacity. A braked wheel therefore holds without creep and a slipping one feels exactly its capacity. This replaces the planned regularised friction.
- **Combustion powertrain**: Chrono's SimpleMap model.
  - Engine speed follows the driveline algebraically.
  - Torque blends the zero- and full-throttle maps in throttle, with a fuel cut above `max_rpm`.
  - Automatic up- and downshift at fixed engine speeds per gear, with an optional torque gap.
  - Fixed-share open differentials, the axle split as the centre differential, and driveline inertia lumped onto the driven wheels.
  - The clutch, torque converter and engine inertia were deferred.
- **Electric powertrain**: motors per axle, side or wheel, with a torque and power limit and a first-order lag. With `no_load_speed`, a motor is a voltage-commanded DC motor: stall torque falls linearly to zero at the commanded fraction of the no-load speed, and back-EMF brakes it. `DriveInput::yaw` mixes into left and right motors.
- **Presets**:
  - `sedan_like` and `offroad_4x4` come from Chrono's Sedan and HMMWV_Vehicle_4WD (BSD-3, attributed in `source`). `tools/gen_chrono_vehicle_fixtures.py` settles each Chrono model under gravity scaled 0.25–2×. From that sweep we take:
    - the spindle paths, toe and camber over travel;
    - the wheel rates (Sedan) and the spring force tables (HMMWV);
    - the masses, damper rates at the wheel, engine maps, gears and shift points.
  - The wheel positions are Chrono's static spindles in its chassis frame, so the Sedan rests pitched 2.6° nose-down as Chrono's does.
  - Chrono's Sedan is **rear**-wheel drive, not front-wheel drive as planned.
  - `rover_skid` (Husky A200 dimensions) and `rover_diff` (Jackal-sized) use Fiala tyres and DC motors; the diff-drive robot has frictionless casters.
  - Aero drag, steering rate, parking brakes, end stops and colliders are estimates.
- **Tests** (`tests/ground_vehicle.rs`, fixtures `fixtures/chrono/vehicle_{sedan,hmmwv}.json`, regenerated by `make fixtures-chrono`):
  - **Settled state**: the simulation settles on the definition's static state (loads 0.5 %, height 0.5 mm, pitch 2e-4, travel 0.2 mm; zero travel with automatic preload).
  - **Statics vs Chrono**: ride height within 2 mm, pitch within 1e-3, axle loads within 0.5 %. Actual agreement: Sedan 0.2309 m and 0.0456 rad, the same as Chrono.
  - **30 % slope**: braked (the Husky on its parking brake), facing uphill and downhill, the Sedan, 4×4 and Husky creep < 0.1 mm in 5 s. Released, they roll away.
  - **0–100 km/h vs Chrono**: Sedan 6.4 s (Chrono 6.05 s), 4×4 9.4 s (Chrono 8.9 s). Speeds at 5, 10 and 15 s are within 8 %.
    - The deficit comes entirely from the launch. With no clutch in either model, the engine's rising torque curve drives the transient tyre's lightly damped wheel mode, so the rear wheels spin up more than Chrono's quasi-steady tyres do. Afterwards the accelerations agree.
    - Beyond about 15 s Chrono keeps its map torque at the engine speed limit, while we cut fuel.
  - **Coast-down vs Chrono**: 30 s in gear from Chrono's speed and gear. Sedan 16.42 m/s against Chrono's 16.42 m/s; 4×4 12.92 against 13.13 m/s; distances within 1 %. Both runs use zero rolling resistance, because Chrono's Pac02 has none without QSY coefficients.
  - **Braking**: the Sedan's stopping distance is within 5 % of Chrono's. The 4×4 needs 34 m against Chrono's 24 m: its wheels lock, and our Pac02 (exact against MFeval) falls to μ ≈ 0.55 at locked wheels, while Chrono's keeps about 0.9. So the step-7 braking criterion applies to the Sedan only.
  - **Robots**: the robots drive straight at their top speed (diff 1.77 m/s, skid 1.18 m/s), turn on the spot counter-clockwise, and stop on back-EMF.
  - **Steering**: HMMWV full-lock angles 30.6°/24.1° as in Chrono.
- **Fix found by the tests**: `rest()` gave the initial velocity along the pitched chassis x axis, which dropped the Sedan onto its bump stops at speed. It now follows the heading.
- **Cost**: one car tick (4 MF tyres, 16 DoF, powertrain, no controller) takes 3.6–4.0 µs (release); a Husky tick 1.5 µs.
- **Follow-ups**:
  - Clutch and torque-converter launch; this also cures the launch wheel-hop.
  - Engine inertia.
  - Caster and upright pitch in the kinematics tables, for anti-dive and anti-squat.
  - Stiff Chrono Sedan springs (3 Hz ride) lift the rear wheels briefly under full braking.

**As built in step 4** (`control::ground`, `vehicles::ground` per-wheel input):
- **Vehicle side**:
  - `DriveInput::wheels: Option<WheelCommands>` carries per-wheel `drive` / `brake` / `steer` arrays (8 wheels). When present they replace the mixed commands. A motor driving several wheels takes their mean; a combustion drive keeps `throttle`.
  - `SteerMode::Independent { max_angle, rate }` per axle adds a rate-limited knuckle per wheel. Without per-wheel commands it follows the Ackermann angle of its `steer` share, so a four-wheel-steered car still drives with plain steering.
- **Controller** (`GroundController::update(setpoint, estimate) → DriveInput`, `GroundEstimate::of(&Wheeled)`). Setpoints: `Direct`, `Pedal`, `Sides`, `SpeedCurvature`, `SpeedYawRate`.
  - **Speed loop**: PI giving an acceleration (τ = 0.5 s).
    - The demand is clamped to [−6, 3] m/s² and to what the engine in its current gear, the motors (including back-EMF) and the brakes can give right now.
    - The integrator runs only within ±1 m/s of the target (integral separation). It trims resistances and grades and does not wind up during large steps. Plain conditional integration left a 13 % overshoot, the PI zero on an integrator plant.
    - Hold on the brakes below 0.3 m/s when the target is 0.
    - Reverse engages only below 0.5 m/s; otherwise a negative request brakes first.
  - **Inverse powertrain**:
    - Engine: the throttle is interpolated between the zero- and full-throttle maps at the current rpm and gear. Brakes add whatever engine braking cannot provide.
    - Electric: the motor torque, plus the back-EMF term for DC motors.
  - **Traction control**: the drive force fades from 1 to 0 as the driven-wheel slip goes from max(10 %·|v|, 0.3 m/s) to twice that, and the integrator is held meanwhile. It was needed: a reverse launch of the Sedan spun the rear wheels to 2.5× the body speed and rang the step-3 wheel-hop mode for 1.5 s (0.46 m/s overshoot on a 3 m/s step, against 0.10 m/s with it).
  - **Curvature**: bicycle feed-forward `δ = atan(Lκ)/share`, plus an integral on κ − r/v above 2 m/s, limited to ±0.15 rad. It takes out understeer.
  - **Side drives** (skid and diff): wheel-speed targets from (v, ω), a per-motor PI on wheel speed through the inverse motor model, and outer integrals on body speed and yaw rate for skid slip. The outer integrals are held while a motor saturates. `vk` maps to ω = v·κ.
  - Cost per update: 47 ns (Sedan, vk), 26 ns (Husky, vw).
- **Action modes** (`GroundActionMap`, components in [−1, 1], named):
  - `raw`: pedal and steering, or left and right for side drives.
  - `vk`: speed and curvature. Positive speed scales to `speed`, negative to `reverse`.
  - `vw`: speed and yaw rate; side drives only.
  - `per_wheel`: only the channels the vehicle has, in order `throttle` (combustion), `steering` (axles steered through the Ackermann linkage), `drive_<wheels>` (one per electric motor), `brake_<w>`, `steer_<w>` (independent axles). One-sided channels read negative values as 0.
  - Default limits come from the vehicle: speed = top speed capped at 20 m/s (Husky 1.18, Jackal 1.80); reverse = 0.3·speed for engines, speed for electric; curvature = 95 % of full lock by the bicycle model, or 2/track; yaw rate = 0.8·speed/track for side drives.
- **Tests** (`control/tests/ground_loop.rs`, flat asphalt, 1 kHz):
  - **Speed steps** 0 → 10 → 20 → 5 → 0 → −3 → 0 m/s on the Sedan, the 4×4 and an electric four-motor Sedan: overshoot ≤ 0.14 m/s (the limit is 10 % of the step), settled error ≤ 0.015 m/s, then held.
  - **Circle**: 30 m radius at 10 m/s both ways, and 15 m in reverse at 3 m/s; curvature within 0.1 % for all three cars.
  - **Robots** (vw): v and ω within 2 % on the Jackal and Husky, including turning on the spot and reversing; vk curvature within 5 %.
  - **per_wheel**, on the electric Sedan with independent steering on both axles:
    - A ±81 N·m left/right torque split at 10 m/s gives a tyre yaw moment of 819 N·m, against 786 N·m from statics (Σ −y·T/r); the car turns left.
    - Braking one wheel yaws the Sedan toward that side (±0.06 rad in 1.5 s).
    - All wheels at 30° crab along 29.7° with |r| ≤ 0.0013 rad/s.
    - Counter-phase steering at 3 m/s gives κ 0.1072 against 0.1089 predicted; front steer alone gives 0.0538 against 0.0544 (a ratio of 1.99).
    - Channel lists are exact: the Sedan has throttle, steering and 4 brakes; the Jackal has 2 drives and 2 brakes; the Husky's drives are `drive_0_2` and `drive_1_3`.
- **Findings**:
  - Crabbing needs zero toe. With the Sedan's static toe-in of about 1°, the toe forces of each wheel pair, turned 30°, form a yaw couple that curves the crab path (κ ≈ 0.004 1/m). The test car's corner modules therefore have no toe.
  - The skid rover cannot hold κ = 1 at half its top speed: the scrub saturates the outer motor. The controller gives priority to neither speed nor curvature; the RL policy or a planner must stay within the limits.
  - `sim`/scenario support (step 6); scenarios reject wheeled vehicles until then.

**As built in step 5** (agents over vehicle families):
- **Baseline first**: before the refactor, `sim/tests/golden.rs` fixed hashes (`fixtures/golden_trajectories.toml`) of observations, state rows, events, world state hashes and recorded messages. They come from 120 batched policy steps (with partial and seeded resets) of `hover.toml`, all five multirotor action modes with LiDAR/GPS/IMU, agent contacts, wind and randomisation, and a small wild map pool. A reference recording (`fixtures/recordings/hover.mcap`) must still be read and reproduced message for message. MCAP files are compared by message, not by byte: the writer orders its summary section by hash map. All of these, the procgen golden hashes and every M1 test are unchanged after the refactor.
- **vehicles**:
  - `Vehicle { Multirotor, Wheeled }` holds the shared queries: pose, twist, mass, `state()`, colliders, contacts, `is_gear(group)` (landing gear or skids), specific force and angular acceleration. It also holds the shared phases: `begin_step`, `apply_contacts`, `apply_force`, `finish_step`, plus `place(pose, v, ω)` for replays. Family actuation is reached through `as_multirotor()` / `as_wheeled()`.
  - `SharedDef` is the `Arc`'d definition of either family (name, family, mass, colliders, contact model).
  - `Family` names the family.
  - Nothing assumes wheels or rotors, so a `Tracked` variant slots in beside them (M4).
  - `Wheeled` now reports its specific force (classical acceleration of the chassis origin, from the free joint's q̈ plus ω×v, minus gravity) and angular acceleration, so the IMU works on ground vehicles.
- **control**: `Controller { Multirotor, Ground }`, `Command { Multirotor(Setpoint), Ground(GroundSetpoint) }` (with `From` impls, so `set_command(agent, setpoint)` takes either), and `ActionMapping { Multirotor(ActionMap), Ground(GroundActionMap) }`. `AgentActionMode` is serialised by name. The names are disjoint, so `"ctbr"` or `"vk"` alone picks the family.
- **Scenario**:
  - `action_mode` is optional; the family's default (`ctbr`, `vk`) is filled into the compiled spec. Multirotor scenario JSON stays byte-identical.
  - New `ground_controller` and `ground_action_limits` fields are only written when set.
  - Setting the other family's fields is an error that names the field: `controller`, `action_limits` or `randomize` on a wheeled group, or the ground fields on a multirotor group. So is a mode of the wrong family, or `motor_speeds` without rotors.
  - Ground vehicles always spawn on the ground, at the static rest pose from `Wheeled::rest`, with the heading drawn from `yaw_deg`. Goals are sampled from the ground point.
  - Terrain-aligned spawning on slopes and reachable ground goals come in step 6.
- **Agent step**: the controller update and actuation match on (vehicle, controller, command). A multirotor runs the cascade then the rotors. A wheeled vehicle runs `GroundController::update` → `apply_drive` → `apply_tires`. Contacts, agent forces, integration, events, shapes and sensors are shared. Event rules are the same for both families: contact on a non-gear collider, or faster than `crash_speed`, is a crash.
- **State and recording**:
  - `STATE_FIELDS` is unchanged and common to both families.
  - `state_hash` adds a ground vehicle's steering angle, per-wheel steer, drive, brake and tyre force, and gear, engine speed and torque.
  - `/agent/<id>/state` carries `motors` for multirotors. For ground vehicles it carries `steering`, `wheels` (spin, steer, travel, drive and brake torque, load), `gear` and `engine_speed`; `RecordedState` reads both, with defaults.
  - `/meta` vehicle definitions stay untagged for multirotors (the old format); other families carry their `type` tag.
- **Python / viewer**:
  - `group_info` adds `family`; `num_rotors` is 0 for ground vehicles, and `mass` is the total mass.
  - The viewer draws ground vehicles as a box with a red nose and wheels posed from the simulated steering, travel and spin (`scene::props::wheeled`). The HUD shows steering, gear and rpm. The keyboard drives them through `Pedal`. Replays place them from the recorded pose.
- **Tests** (`sim/tests/ground.rs`, 1 kHz):
  - Two Sedans (default `vk`), a skid rover (`vw`) and a drone share a flat world. At rest the vehicles move < 2 cm in 1 s with no events, and the chassis specific force equals gravity in body axes within 0.05 m/s².
  - Driving at half the speed limit while turning, speed is within 10 % after 5 s; the drone holds its position meanwhile.
  - `SpeedCurvature(0, 0)` stops a Sedan to < 5 cm/s.
  - Four 4×4s driven straight through a dense forest patch raise `CRASH_OBSTACLE` and are disabled.
  - State hashes are identical at 1 and 4 envs on 1 and 3 threads, and cover the ground state.
  - Recordings carry and read back the wheel data, and scenarios round-trip.
  - Mismatched fields and modes give errors that name the field.
- **Throughput** (criterion against the pre-refactor baseline, same session):
  - Single-world steps are +2 to +3 % (quad 6.85 → 7.0 µs). Repeated runs scatter by about ±1 %, and one run showed −10 %.
  - Batched throughput is within noise: 256 quads on 10 threads +1.8 % (p = 0.08); with forest LiDAR +0.6 % (p = 0.6); the 128-agent swarm +1.2 % (p = 0.08).
  - `#[inline]` on the `Vehicle` wrappers made no measurable difference; the cost is the extra dispatch.

**As built in step 6** (`sim` integration of ground vehicles):
- **Physics rate**:
  - `physics_hz = 0`, the new default, means auto: 1 kHz when any group is a ground vehicle, else 500 Hz.
  - The resolved rate is filled into the compiled spec, so `/meta` and the multirotor goldens are unchanged.
  - Python tasks pass `physics_hz = None` as 0. `hover.toml` and `forest.toml` no longer pin 500.
- **Drivable ground** (`sim::drive`):
  - A group's `drivable = { cell = 2, max_slope_deg = 25, spawn_slope_deg = 15, margin = 1, obstacle_height = 0.25, max_water_depth = 0 }` builds a `DriveGrid` per map.
  - Grids are built in parallel over maps and shared by groups with the same spec and vehicle half-width.
  - A cell is blocked when:
    - its normal, or its gradient over the cell, is steeper than `max_slope_deg`;
    - water is deeper than `max_water_depth`;
    - a solid obstacle's AABB (from `obstacle_height` above the ground up to 2 m above the terrain) comes within its radius + half-width + `margin`. Foliage does not block.
  - Connected components (4-connected, found in scan order) answer reachability, instead of an A* per goal.
  - `DriveGrid::path` (8-connected A* without corner cutting, deterministic) serves scripted drivers and tests.
  - `resistance_cost` (default 0) weighs soft ground: a metre costs `1 + resistance_cost·f`, where `f` is the group vehicle's motion resistance on the cell's material (`WheeledDef::motion_resistance`; added in M4c step 5).
  - A `drivable` table on a multirotor group is an error.
- **Spawning**:
  - Ground vehicles spawn in drivable cells of the largest component, clear of solids by their radius and at most `spawn_slope_deg` steep. The slope is `drive::slope`: the steeper of the normal and the gradient over ±1.5 m.
  - The spawn slope limit is what stops creep on rough terrain. On planar inclines both cars hold within 2 cm up to 25°. On 19–23° rough wild slopes, uneven wheel loads let them creep 0.2–0.6 m.
  - Spawns are posed by `drive::ground_pose`: a least-squares plane through the wheel contacts and the centre, lifted by the largest residual, with the vehicle's rest pose on top. Grid layouts are placed the same way.
  - Goals sit at chassis height over the terrain and are scored by reachability from the spawn. The RNG draw order is unchanged, so multirotor goals are identical.
- **Events**:
  - `ROLLOVER` (bit 12, terminal): up-axis tilt beyond `events.ground.rollover_deg` (60°).
  - `STUCK` (bit 13, not terminal): the vehicle stays within `stuck_distance` (0.5 m) of an anchor for `stuck_time` (5 s; 0 disables it).
  - Water uses the existing `WATER` event.
  - The `events.ground` table is written only when set.
- **Observation terms**: all need a ground vehicle, else error.

  | Term | Dim | Content |
  |---|---|---|
  | `speed` | 1 | Body forward speed |
  | `sideslip` | 1 | `atan2(v_y, max(\|v_x\|, 1))` |
  | `pitch_roll` | 2 | Pitch and roll angles |
  | `wheel_speeds` | n | Spin × radius per wheel |
  | `wheel_slip` | n | Longitudinal slip κ per wheel |
  | `steering` | 1 | Steering angle |
  | `gear_rpm` | 2 | Gear and engine speed in krpm |
- **Traction control** (`control::ground`): open differentials could not launch with crossed axles. Traction control scaled the drive force on the *mean* driven-wheel slip, so two unloaded, spinning wheels shut the throttle off on a 5° hill.
  - Now any driven wheel slipping past the allowance gets a brake, in proportion to its excess slip, up to its share of the drive torque. This goes through the new `DriveInput::wheel_brake` (per wheel, added to the pedal, and not serialised when zero).
  - The throttle fade uses the *least* slipping wheel.
  - The Chrono validation and controller tests are unchanged.
- **`offroad` wild preset** (512 m, gentle):
  - Relief 40 m with broad hills. Slope p50 ≈ 3°, p99 ≈ 16°.
  - Forest patches with clearings: about 1100 trees per 512 m map, and few rocks.
  - Its golden map hash is committed.
- **Tests** (`sim/tests/offroad.rs`, plus a `cars` golden in `golden.rs`):
  - Spawns and goals:
    - Spawns are in the largest component, below 15°, clear of trunks, and aligned within 8° with the finite-difference terrain normal.
    - Goals are reachable, with a path.
    - Creep is < 10 cm in 1 s.
    - The drivable share is > 50 %.
  - The new observation terms are checked against the vehicle state while 8 trucks drive.
  - Rollover, stuck and water events.
  - Batch independence of the thread count.
  - **Batch of 64 4×4s** (8 worlds × 8) on offroad maps: they follow grid paths with pure pursuit for 20 s.
    - 59/64 move more than 20 m, 51 reach at least one goal, and 71 goals are reached in total.
    - 4 end on another terminal event (trees, terrain or bounds). 6 collide with other cars, because the follower ignores the other agents.
  - The `cars` golden covers all four ground action modes and all four vehicle presets, with LiDAR, goals and events. The other golden hashes are unchanged.
- **Throughput** (criterion, powersave governor):

  | Benchmark | Result |
  |---|---|
  | `world_step/car` (sedan circling, 20 ticks) | 83 µs, ≈ 4 µs per car-tick |
  | `world_step/truck_offroad_lidar` | 99 µs |
  | `swarm/64_cars` | 2.47 ms, ≈ 8× real time on one thread |
  | `batch_10t/car_x256` | ≈ 35k env-steps/s |

  The multirotor benches are +2 % against the step-5 baseline. `tools/collect_bench.py` now also collects grouped benches (`group/name`).

**As built in step 7** (handling validation, `vehicles/tests/handling.rs`):
- **Reference runs**: `tools/gen_chrono_handling_fixtures.py` (in `make fixtures-chrono`, about 20 s) drives the Chrono Sedan on flat rigid ground with μ at the tyres' reference value and writes `fixtures/chrono/handling_sedan.json` (0.57 MB):
  - ISO 4138 constant steering input with speed rising from 5 to 16 m/s (a_y up to 5.9 m/s²). These samples carry per-wheel loads, lateral forces, aligning moments and slip angles.
  - ISO 7401 step steer at 80 km/h, two amplitudes (a_y ≈ 1 and 4 m/s²).
  - ISO 3888-1 double lane change at 80 km/h, driven by a pure-pursuit driver along a cosine-blended centreline.
  - Straight braking from 100 km/h at pedals 0.3, 0.5, 0.7 and 1.0.
  - The speed PI and the driver are simple enough to re-implement exactly in the test, so both cars are driven by the same laws. Chrono runs are rolled out at x = −400 m on a 1 km patch (`Run` gained `patch`/`start`; the older fixtures are unchanged).
- **Findings about the Chrono Sedan** that shape the comparison:
  - Its steering linkage maps input to road-wheel angle nonlinearly (0.573 rad per unit at small inputs, 0.612 at full lock) and lags the input by about 20 ms. Inputs are therefore matched by road-wheel angle: the test fits Chrono's front-wheel angle against roll and takes the zero-roll intercept (both cars have the same roll steer, dδ/dφ ≈ −0.24). Matching a single sample instead left an 8 % yaw-rate deficit.
  - `ChPac02Tire` never passes the inclination to its formulas, so its tyres have no camber thrust. The test zeroes the 22 camber coefficients of the `.tir` for the comparisons with Chrono. With camber our Sedan understeers a little more (K +0.00004 at low a_y).
  - Without ABS, Chrono's rear wheels lock from pedal 0.7 and the car spins (1.4–1.75 rad/s yaw rate); at 0.5 it yaws slightly (0.12 rad/s). Ours stays straight, being exactly symmetric. Only pedals 0.3 and 0.5 are compared.
  - Chrono reports tyre forces at the wheel centres. At a_y 5.3 m/s² its total lateral load transfer is 5 % smaller than ours (5130 vs 5392 N) and its outer-wheel aligning moments are about 35 % larger (Pac02 trail without `LFZO` and with its own offsets). Our front slip angles are about 0.0006 rad smaller at the same a_y. The K offset below was not attributed further, and neither model was changed.
- **Acceptance criteria as implemented**:
  - The understeer gradient is compared absolutely. The Chrono Sedan is close to neutral (K = 0.00053 rad/(m/s²), 0.3°/g, below 3 m/s²), so 10 % of it is 0.03°/g, below what the data resolve. The bound is |K − K_Chrono| < 0.0005 rad/(m/s²), 10 % of a typical passenger car's 3°/g, plus path curvature within 5 % at every sample up to 6 m/s².
  - The linear bicycle model is per wheel: each wheel's lateral force is linear in slip angle and inclination, with the Magic Formula's cornering and camber stiffness at static load, and the road-wheel angle and camber changes as the vehicle model produced them (steer, roll steer, compliance, roll camber). The classical two-stiffness model (K ≈ 0.0001) misses the Sedan's rear roll steer and predicts a 22 % lower yaw rate at 4 m/s².
- **Results** (ours vs Chrono):

  | Test | Ours | Chrono | Criterion |
  |---|---|---|---|
  | K, a_y 0.5–3 m/s² | 0.00015 | 0.00053 | Δ < 0.0005 |
  | K, a_y 3–5.5 m/s² | 0.00154 | 0.00204 | Δ < 0.0005 |
  | Path curvature, worst sample | — | — | 3.2 % (< 5 %) |
  | Step steer, small: steady / peak yaw rate (rad/s), t90 (s) | 0.0450 / 0.0453 / 0.140 | 0.0455 / 0.0462 / 0.140 | 10 %, 10 %, 30 ms |
  | Step steer, standard (a_y ≈ 4) | 0.1997 / 0.2083 / 0.130 | 0.1927 / 0.2046 / 0.120 | same |
  | Step steer, small, vs linear model | 0.0450 (0.0447 with camber) | model 0.0481 (0.0479) | 10 % |
  | Braking, pedal 0.3 / 0.5 | 93.8 / 58.0 m | 95.7 / 60.5 m | 5 % |
  | Lane change: peak yaw rate, peak a_y | 0.219, 4.86 | 0.218, 4.86 | 10 % |
  | Lane change: largest lateral difference | 0.15 m | — | < 0.3 m |

  Both cars keep their centre inside every lane. With a 1.8 m body, both would touch the cones of the offset lane by 2–5 cm; the driver is not tuned for clearance.
- The five tests take 0.3 s.

**As built in step 8** (viewer for ground vehicles):
- **Driving**: `--vehicle offroad_4x4` (or any wheeled preset) gives the first agent a "driver" group in `raw` mode that is not disabled on terminal events.
  - Keys: W/S pedal (S brakes, then reverses), A/D steering, Space handbrake.
  - Gamepad: right trigger minus left trigger is the pedal, the left stick steers, South is the handbrake.
  - Skid-steer and diff-drive robots get `Sides { left: forward − steer, right: forward + steer }`, so they turn on the spot.
  - The steering follows the keys at 2.5 per second (rate-limited, so a tap is not full lock). At speed its range fades as 1/(1 + (v/12 m/s)²).
  - `GroundSetpoint::Pedal` gained `handbrake`, which maps to `DriveInput::parking`; the action map's raw mode never sets it. The parking brake acts on the rear axle only, so at speed it slows the car at about 1.6 m/s² and swings its tail out, as a real handbrake does.
  - `GroundController` keeps the last `DriveInput` it produced (`last_input`), for the HUD.
- **Demo driver** (`--demo`): picks a random reachable point more than 80 m away and follows a `DriveGrid::path` to it by pure pursuit (8 m lookahead). It drives at 8 m/s, slowing to 4 m/s in turns, and picks a new route on terminal or `STUCK` events. It is used for the frame-rate runs and screenshots.
- **Visuals**: the procedural `WheeledVisual` gained a cabin for cars (tyre radius > 0.2 m), the driver's eye point, and suspension links (a strut and an arm per suspended wheel, from their chassis mounts to the wheel centre). Wheels spin, steer and travel from the vehicle's wheel poses.
- **Cameras**: the chase camera for ground vehicles sits lower and closer (pitch 0.2 rad, distance 2.6 spans, zoom limit 1.2 spans) and aims above the chassis. First person sits at the driver's eye, tilted 5° down.
- **HUD and plots**:
  - The HUD shows speed in km/h, the driver's pedal, steering and handbrake, and gear, rpm, steering angle and throttle and brake bars.
  - A per-wheel table shows load, travel, κ, α, Fx, Fy and drive and brake torque.
  - The plots show sideslip in place of AGL, yaw rate and speed against their references, and two tyre scatter plots per wheel (Fy/Fz against α, Fx/Fz against κ). The yaw-rate reference is the commanded one in the speed modes and the kinematic v·tanδ/L otherwise.
  - Wheels carrying less than a fifth of the mean load are left out of the scatter plots: their force ratios exploded on bumps.
- **Recording and replay**:
  - `--record <file>` (live and `policy`) writes MCAP with LiDAR through the sim `Recorder` and closes it on exit.
  - `RecordedWheel` gained `spin_angle`, `kappa`, `tan_alpha`, `fx` and `fy` (serde defaults, so older files still read). This changed only the `cars` golden hash; it was re-blessed after checking that the old recorder still reproduced the old hash.
  - Replay interpolates the wheels and poses the vehicle through the new `Wheeled::show`, which sets the joint coordinates, wheel outputs and powertrain status (`Powertrain::set_status`) and runs forward kinematics.
- **Frame rates** (i7-1365U, Iris Xe, 1920×1080, medium, plots and recording on):
  - Demo drive on the showcase map: 74 fps.
  - Offroad preset: 84 fps (87 over the last frames).
  - Replay of that drive: 98 fps.
- **Finding**: on the showcase map the 4×4 bounces hard. The detail noise (0.25 m at 6 m wavelength) leaves at least one wheel unloaded in 63 % of samples, with peak loads of 23 kN. On the `offroad` preset that falls to 6 % (never three or more wheels), with peaks of 18 kN. The model was not changed; `offroad` is the map for driving.
- **Tests**:
  - Viewer: keys drive, steer, brake and hold a car; a skid-steer turns on the spot; the demo command overrides the keys; a ground replay reproduces spin, steer, travel, tyre forces and wheel poses; tracking references and tyre samples; `--record` writes a drive with wheel data.
  - Vehicles: a shown state matches the simulated one.
  - Scene: links only on suspended wheels, and an eye point inside the body.

**As built in step 9** (`python/autonomousim/tasks/car_waypoint.py`, plus what the task needed in `world`, `sim` and the Python package):
- **Simulation support**:
  - `StaticWorld::obstacle_clearance`: the distance to the nearest solid obstacle only. For ground vehicles, the state row's `clearance` column and the `clearance` observation term use it: the terrain under a car is always about a ride height away, so the old terrain-or-obstacle distance said nothing. Multirotors are unchanged; only the `cars` golden trajectory hash changed (re-blessed).
- **Python API**:
  - `Task.failure_events`: event bits a task adds to the terminal set (`TERMINAL_EVENTS`); the car task adds `STUCK`.
  - `map="offroad"`: a pool of `offroad`-preset wild maps (as `"wild"` is for the `training` preset).
  - Ground vehicles and their action modes (`raw`, `vk`, `vw`, `per_wheel`) are accepted by `Task`; `randomize` stays multirotor-only.
  - `bench.py --task car_waypoint`; `--map`, `--vehicle` and `--action-mode` default to the task's own.
  - `ppo_continuous.py --init <policy.pt>` continues training from a checkpoint (weights and observation statistics; the reward normaliser starts afresh).
- **CarWaypointOffroad-v0**:
  - **Setup**: `offroad_4x4` in `vk` mode (speed up to 8 m/s forward, 2.4 m/s in reverse; path curvature up to the steering lock), policy at 20 Hz, physics at 1 kHz; a pool of 16 `offroad` maps; 3 waypoints, each 25–50 m from the previous one and reachable from it, reached within 3 m; 90 s episodes.
  - **Drivable margin**: goals and spawns lie on the drive grid computed with `drivable_margin` = 3 m of room beside the vehicle, so every route passes gaps about 8 m wide. With the default 1 m, routes squeezed between trunks 4 m apart. The same policy (v2, trained at 1 m) reached 52 % success at 1 m, 60 % at 2 m and 72 % at 3 m; the scripted driver went from 46–51 % to 70 %.
  - **Observation** (231 values): goal in the heading frame (1/20, clipped to ±3), speed, body velocity and rates, pitch and roll, steering angle, last action, and a roof LiDAR (3 rings at −10°, −3° and 3°, 72 azimuths, 30 m, log ranges).
  - **Reward**: horizontal progress to the current goal (measured from the positions before and after the step, so switching goals does not jump), +10 per waypoint, a proximity penalty `0.2·max(0, 1 − c/3 m)²` on the obstacle clearance, smoothness `0.02‖Δa‖²`, −50 on terminal events (crash, rollover, water, out of bounds, stuck: less than 0.5 m in 4 s).
  - **Tests**: on the flat map a scripted driver reaches all three goals with the expected return; a car that stands still for `stuck_time` ends `STUCK` (terminated, not a success) with the terminal penalty; spaces, dtypes and `check_env`.
- **Throughput** (i7-1365U):
  - One `offroad_4x4` tick, single thread: 4.8 µs (FK 0.75, drive 0.15, tyres 1.74, contacts 0.12, ABA 1.53). The full Magic Formula stays: an approximation would give up the validated fidelity for at most a third of the tick.
  - `BatchSim`, 20 Hz policy (50 ticks per step), LiDAR on: 4.1k env-steps/s with 1 thread, 6.5k with 2, 8.5k with 4, about 11k with 10. The scaling is poor, most likely from the laptop's power limit (all-core clocks drop); `perf` is not available here to confirm it. Short benchmarks read about 23k because crashed worlds are disabled and skipped; the numbers above keep every world driving.
  - PPO: 3–5.6k SPS (the simulation takes 57–60 % of the time; slower when the laptop throttles). The ≥ 15k raw target in the table below is missed, and the per-tick target (≤ 4 µs) by 20 %.
- **Training** (`ppo_continuous.py --hidden 256 --torch-threads 6 --bound-coef 0.01`, evaluated deterministically on unseen maps, `map_seed` 1000, over 512 episodes; 256 for v1, v2 and the task variants):
  - v1 (32 azimuths, 1 m margin, 10M steps): 60 % success; 27 % obstacle crashes, 12 % truncated.
  - v2 (72 azimuths, 1 m margin, 11M steps): 52 %; LiDAR resolution was not the limit. The crashes fell in two groups: reversing at full speed into trunks the car had just turned away from, and approaching at 4–8 m/s with the turn started too late.
  - v3 (3 m margin, 30M steps, 2 h 5 min, trained with 60 s episodes): 78.5 % with 60 s episodes (11 % obstacle crashes, 7 % truncated, 3 % stuck). Success was still rising slowly when the learning rate reached zero.
  - Attempts to pass 80 % within 60 s, all worse: v3 continued 15M steps at learning rate 1e-4 (76 %); continued 20M steps with γ = 0.995, a 10 s horizon for manoeuvres (77 %); a penalty per metre reversed (70.5 %: in 8 m gaps a forward U-turn, about 11 m across, rarely fits, so reversing is needed, not a bad habit); a proximity penalty sized to the car (5 m from the centre, weight 0.5; the bumpers reach 2.6 m) from scratch (74 %).
  - Where the rest fails: tracing the truncated episodes showed the car shuffling back and forth near goals 3–5 m to its side, inside its turning circle (about 5.5 m), or reversing long stretches towards goals behind it at 2.4 m/s. The same v3 policy scores 83 % with 90 s episodes and 85 % with a 4 m goal radius.
  - **Decision (with the user)**: the episode time became 90 s, since routes detour round trees and a car needs three-point turns a drone does not. The goal radius stays 3 m. **v3 with 90 s episodes: 82.2 % success over 512 episodes on unseen maps** (13 % failures, 4.7 % truncated, 2.6 of 3 goals on average).
  - Not tried: a recurrent policy, or a planner's route direction (`DriveGrid::path`) as an observation term.
- **Viewer**: `export_policy.py` + `autonomousim-viewer policy <json> --agents 4 --lidar-view` drives the exported v3 network in Rust on an unseen map (seed 1000) at about 140 fps; the export is checked against PyTorch.

### Performance targets (i7-1365U, release)
| Metric | Target |
|---|---|
| One car tick (ABA with 16 DoF, 4 MF tires, powertrain, controller) | ≤ 4 µs |
| `BatchSim`, N = 256 cars, 10 threads, 1 kHz physics, 20 Hz policy | ≥ 15k env-steps/s raw (≥ 300k physics ticks/s) |
| Tire force evaluation (combined MF 6.1) | ≤ 150 ns |


## Milestone 3: Multi-agent

Planned 2026-09-25. Scope from the roadmap: a PettingZoo `ParallelEnv` and a native group-batched API, mixed air/ground teams, full-shape agent contacts, swarm performance, neighbour observations. Two things were decided with the user at the start:
- **Closing demo**: `SwarmWaypointForest-v0`. Eight drones share one policy; each flies its own chain of waypoints through the forest, sees its nearest neighbours and its LiDAR, and must avoid the trees and the other drones.
- **Live viewer attach** moves to M9, where the ROS 2 bridge brings the transport (zenoh or DDS).

**Already in place from M1/M2**: several groups per world (any vehicle family), per-agent spawns (`min_separation`) and goal chains, frictionless penalty contacts between agents' sphere colliders with a deterministic sweep-and-prune (`sim::interaction`), `CRASH_AGENT`, LiDAR that sees other agents, per-agent parallel phases from 32 agents on, per-agent MCAP channels, and viewer replay and policy playback of several agents. A swarm of 128 hovering drones runs at 33× real time in one world.

### Design
- **Episodes with several agents**: all agents of a world share one episode. An agent with a terminal event (crash, water, bounds, a task's failure events) or that finished its task stops: it is disabled (`disable_on_terminal`, already the default), frozen, and drops out of contacts and sensors. Its reward stops, and its later transitions are masked out. The world ends when every agent has stopped, or at the time limit (truncation for all agents still going). Autoreset is per world.
- **Native multi-agent vector env** (`autonomousim.MultiAgentVectorEnv`, `gym.make_vec`-free, like the Rust `BatchSim`): arrays per group, keyed by group name:
  - `obs[g]`: `float32 [num_envs, count_g, obs_dim_g]`; `reward[g]`, `terminated[g]`: `[num_envs, count_g]`; `truncated`: `[num_envs]` (per world); `info["active"][g]`: agents still going before this step.
  - `step(actions)` takes `{group: [num_envs, count_g, act_dim_g]}`, or one array for single-group scenarios.
  - SAME_STEP autoreset with dense `final_obs[g]` and a per-world `_final_obs` mask, as the single-agent env.
  - Groups with different vehicles, observation and action sizes (mixed air/ground teams) work the same way.
- **Tasks for several agents**: `MultiAgentTask` computes rewards and ends on `[num_envs, count]` arrays per group, with world-level truncation and per-agent success. Existing single-agent tasks keep working unchanged (the single-agent `VectorEnv` stays as it is). Agent-to-agent terms (distance to the nearest other agent) come from a new state column rather than numpy pairwise distances, so they cost O(n) and follow the colliders.
- **PettingZoo `ParallelEnv`** (`autonomousim.pettingzoo.parallel_env(task, ...)`): one world. Agents are named `"<group>_<k>"`; stopped agents leave `env.agents` as PettingZoo requires; the episode ends when no agent is left. It passes `pettingzoo.test.parallel_api_test` and the seed test. `pettingzoo` joins the dev dependency group.
- **Neighbour observations** (Rust `ObsSpec`, so the viewer can run multi-agent policies without Python):
  - `neighbors`: the `k` nearest other active agents within `range` (optionally of given groups), sorted by distance with ties broken by agent id. Per neighbour: relative position and relative velocity in the heading frame, and 1 for a present slot. Missing slots are zero. `6k + k` values.
  - `nearest_agent`: distance to the nearest other agent's colliders (surface to surface), up to `range`.
  - A state column `agent_clearance` (STATE_DIM 21), the same distance up to 20 m, for rewards. The golden trajectory hashes include the state rows, so they are re-blessed once, after checking that every world's physics `state_hash` is unchanged.
  - Neighbour search reuses the per-tick sweep order; with ≥ 64 agents a uniform grid on the xy plane replaces the O(n²) scan.
- **Full-shape agent contacts**:
  - Ground vehicles' wheels join their agent shapes (a sphere per wheel at the wheel centre with the tyre's radius), so drones can hit and rest on cars and cars can push each other by the wheels.
  - Regularised Coulomb friction between agents (μ from the two colliders' materials, 0.5 by default), with bristle anchors keyed by the collider pair as for terrain contacts, so a drone can sit on a moving car's roof.
  - Contact normals and forces stay pairwise in the sweep order (deterministic).
- **Swarm performance**: 256 drones in one world at ≥ 20× real time (≤ 1 ms per 20 ms policy step at 500 Hz physics, laptop). Profile first (per-phase timers, since `perf` is blocked); options in order: fewer fork–joins per tick, the xy grid for pairs and neighbours, chunked parallel work, and a structure-of-arrays fast path for multirotors only if the others fall short.
- **Training**: `examples/ppo_multiagent.py`, parameter-shared PPO (IPPO): every (world, agent) slot is a sample, masked while its agent is stopped; per-agent GAE, truncation bootstraps from `final_obs`. Observation normalisation and the network are the single-agent script's, so exported policies run in the viewer's `policy` command unchanged.
- **Viewer**: `policy` runs multi-agent policies (neighbour terms come from the Rust `ObsSpec`); the HUD shows the followed agent's neighbours; replay shows agent contacts.

### Implementation order
| # | Step | Done when |
|---|---|---|
| 1 | Neighbour observation terms, `agent_clearance` state column, xy grid for agent queries (done 2026-09-25; grid moved to step 5) | Terms match a brute-force reference on random swarms; order and ties deterministic; disabled agents are invisible; physics state hashes unchanged (goldens re-blessed for the new column only) |
| 2 | Full-shape agent contacts: wheel spheres, friction with bristle anchors (done 2026-09-25) | A drone lands on a parked car's roof and stays there while the car drives off at 2 m/s; two cars pushing wheel to wheel exchange equal and opposite forces; a drone falling onto a car hits `CRASH_AGENT`; determinism suite passes |
| 3 | Python: `MultiAgentVectorEnv`, `MultiAgentTask`, per-agent stopping and per-world autoreset; a mixed drone + car scenario (done 2026-09-25) | Shape, dtype, masking, autoreset and seeding tests for one group and for a mixed team with different obs/act sizes |
| 4 | PettingZoo `ParallelEnv` (done 2026-09-25) | `parallel_api_test` and the seed test pass for a single-group and a mixed-team task |
| 5 | Swarm performance (done 2026-09-25) | 256 drones hovering in one world ≥ 20× real time; 128 unchanged or faster; benchmarks recorded |
| 6 | `ppo_multiagent.py` and a quick check task (`SwarmHover-v0`: N drones hold assigned slots in a formation without touching) (done 2026-09-25) | Formation error < 0.3 m and no agent contacts in 95 % of episodes after ≤ 15 min of training |
| 7 | `SwarmWaypointForest-v0`, training pipeline, viewer (done 2026-09-25) | The task trains end to end; the exported policy flies the swarm in the viewer; a recorded episode replays. Policy targets (≥ 80 % finish, < 2 % agent collisions) moved to the user's own training (2026-09-25) |

**As built in step 1** (`sim::interaction`, `sim::obs`, `sim::world`):
- **Queries** (`sim::interaction`): `surface_distance` (the smallest gap between two agents' collider spheres, with a bounding-sphere early exit), `agent_clearance` (over the other active agents) and `nearest_agents` (the `k` nearest active agents by centre distance within a range, sorted by distance, ties by agent index). A unit test checks all three against brute force on 60 random shapes, with inactive agents and an exact tie.
- **Observation terms**: `neighbors` (`count` 1–16, default 3; `range`, default 20 m): per slot the relative position and velocity in the heading frame (scaled and clipped like the other terms), then 1 for a present slot; missing slots are zero, so `7·count` values. `nearest_agent` (`range`): the surface distance to the nearest other agent, `range` when there is none. `count` and `range` are new optional `ObsTerm` fields, validated per term. Filtering neighbours by group is not built; no task needs it yet.
- **State column** `agent_clearance` (STATE_DIM 21), up to 20 m. Python's `STATE` slices follow `STATE_FIELDS` on their own.
- **Test** (`neighbour_terms_and_agent_clearance`): five drones flying apart and a car in one world; both terms and the state column match brute-force values, and a disabled drone disappears from the other agents' observations.
- **Goldens**: before re-blessing, the old and new code were compared scenario by scenario: physics state hashes, observations, events and the first 20 state columns are identical in all four scenarios; the only changed recording message is `/meta` (the new state field). The goldens were then re-blessed.
- **Deferred**: the xy grid for agent queries moves to step 5 (swarm performance), where it will be profiled together with the contact sweep. The queries are O(n²) per world for now, which is cheap at the sizes of steps 2–4 (8 agents).

**As built in step 2** (`sim::interaction`, `sim::agent`, `sim::world`):
- **Wheels**: a ground vehicle's shape gains one sphere per wheel at the wheel centre with the unloaded tyre radius (`Wheeled::wheel_radius`). As spheres they reach a tyre radius sideways, wider than the tyre itself. Their contact forces act on the chassis at the contact point, and the wheel's spin is not part of the contact velocity. Other agents' LiDAR now sees the wheels too, which changed the `cars` golden (one car's LiDAR sees the other's wheels). With the wheel spheres switched off, all four golden hashes were unchanged by the friction code; the `cars` golden was re-blessed.
- **Friction**: a bristle spring per touching sphere pair, as for static contacts, keyed by (lower agent index, its sphere, higher agent index, its sphere) and kept in the world (`AgentContactState`, cleared on reset, part of snapshots). `μ` = `AGENT_FRICTION` (0.5) × both spheres' friction factors: colliders carry no material, so the collider's `friction` factor stands in for one; wheels use 1. Normal and bristle springs of the two agents act in series.
- **Events**: a contact is a crash (`CRASH_AGENT`, both agents) unless one of the two spheres is gear (a multirotor's gear, a ground vehicle's skids or wheels) and the approach speed is below the scenario's `crash_speed`. Such a slow contact counts as `GROUND_CONTACT` for both, and `LANDED` follows as on the ground. A drone landing gently on a car, or two cars leaning on each other by the wheels, is therefore not a crash; a body-to-body touch still is.
- **API**: `WorldInstance::place_agent` (put an agent at a pose; its shape follows at once) and `WorldInstance::agent_contacts` (per agent, the last tick's agent contact forces and flags).
- **Tests** (`crates/sim/tests/sim.rs`):
  - An iris-like drone descends onto a parked 4×4's roof without `CRASH_AGENT`, rests there with its rotors off (`LANDED`), and stays within 0.1 mm of its spot while the car drives 10 m at 2 m/s.
  - The same drone dropped 2 m onto the roof crashes both agents.
  - Two 4×4s placed side by side with their wheels overlapping by 2 cm exchange equal and opposite forces and moments on every tick. They are pushed apart without a crash or rollover.

**As built in step 3** (`python/autonomousim/{multiagent.py, tasks/multi.py}`, `BatchSim.disable`):
- **Teams**: a `MultiAgentTask` is made of teams, `{group name: Team(task, count)}`, where each team's task is an ordinary single-agent `Task`. It supplies the vehicle, the action mode, the group entries and the reward, failure and success rules. The team task is bound over `num_envs × count` slots and gets the state rows flattened to `[num_envs·count, STATE_DIM]`, so single-agent reward code (hover, waypoints, the car) runs unchanged. The team tasks' own truncation is ignored. Map, rates (by default the first team's), episode time and wind belong to the `MultiAgentTask`; the teams' `settings()` are deep-merged in team order.
- **`MultiAgentVectorEnv`**: a plain class with the Gymnasium method names rather than a `gymnasium.vector.VectorEnv`, since the spaces are per group (`observation_space(g)`, `action_space(g)` batched; `single_observation_spaces[g]`). Autoreset is `SAME_STEP` or off (`autoreset=False`) rather than an `AutoresetMode`. Per step it returns, per group, rewards and `terminated` (`[num_envs, count]`), plus a per-world `truncated`. `info` has `active` (agents still going before the step), `events`, and on episode ends `episode` (returns and success per agent, length per world) and dense `final_obs`.
- **Stopping**: an agent stops when its team task says so (terminal event, failure events, `failed`, `succeeded`). The environment then calls `BatchSim.disable(group, mask)` (`WorldInstance::disable_agent`: frozen, and out of contacts and sensors at once), which is idempotent for agents Rust already disabled on a terminal event. Rewards of stopped agents are zeroed, and their `terminated` stays false after the step they stop.
- **Tests** (`tests_py/test_multiagent.py`):
  - shapes and dtypes for one group;
  - an agent flying out of its goal box stops, is masked and stays frozen while the others go on;
  - a world ends by termination when all its agents have stopped, another by truncation, each with its own episode length and a fresh spawn;
  - seeding reproduces every group of a mixed team;
  - a mixed team (3 drones with 19 observations and 4 actions, one 4×4 with 231 and 2) runs at 1 kHz and is truncated and reset per world.

**As built in step 4** (`python/autonomousim/pettingzoo.py`):
- `parallel_env(task, seed=..., num_threads=1)` wraps a one-world `MultiAgentVectorEnv` without autoreset. Agents are named `"<group>_<k>"`, and `observation_space(agent)` and `action_space(agent)` return one cached space object per agent. Actions of agents that are missing from the dict (stopped ones) are zero. `infos[agent]["events"]` holds the event bits; `state()` returns all agents' state rows in `possible_agents` order.
- A stopping agent reports `terminations[agent] = True` once and leaves `env.agents`. At the time limit every agent still going is truncated and the list empties.
- `task` is a `MultiAgentTask` or the name of one registered in `MULTI_TASKS` (`make_multi_task`; `MultiAgentVectorEnv` accepts names too).
- `pettingzoo` 1.27 joined the dev group.
- Tests: `parallel_api_test` (1000 cycles) and `parallel_seed_test` pass without warnings, for four hover drones and for a mixed team of two drones and a 4×4. Further tests check that a stopped agent leaves the dict outputs, and that the time limit truncates everyone and ends the episode.

**As built in step 5** (`sim::interaction::AgentGrid`, `sim::world`):
- **Profile first**: with 256 hovering drones, the physics step was not the bottleneck. Writing the state rows after each policy step took 3.5 ms, because `agent_clearance` compared every pair of agents sphere by sphere (all 256 drones were within 20 m of each other).
- **`AgentGrid`**: a uniform grid on the xy plane over the active agents' centres. Cells hold about two agents each, with at most 64 × 64 cells, stored compressed. It is rebuilt after every policy step, reset, `place_agent` and `disable_agent`. `clearance` and `nearest` search ring by ring and stop once no unvisited agent can matter. Their results equal the scans bit for bit (a property test covers dense, sparse, line-shaped and duplicate layouts, plus an inactive agent outside the grid). Below 32 active agents the scans are used. Agent contacts keep the sweep along x, which was already cheap.
- **Parallel outputs**: from 32 agents on, observations and state rows are written in parallel over agents (disjoint rows, so the results don't depend on the thread count). The static `clearance` term (a terrain closest-point search over 20 m, about 2 µs per agent) was the next largest serial cost.
- **Result** (i7-1365U, 10 threads, `swarm` benchmark, one world stepped in a `BatchSim` pool, 20 ms policy steps at 500 Hz), with the laptop's power profile set to *performance*:

  | Drones | Policy step | Real time |
  |---|---|---|
  | 256 | 0.744 ms | 26.9× |
  | 128 | 0.344 ms | 58× (was 0.61 ms, 33×) |

  In the default power profile, the same 256-drone step took 0.6 ms right after idle but 1.3–1.4 ms under sustained all-core load, as the clocks dropped. The earlier poor thread scaling of the car benchmarks (M2 step 9) probably has the same cause.
- The later fallbacks were not needed: fewer fork–joins per tick, and a structure-of-arrays path for multirotors.

**As built in step 6** (`examples/ppo_multiagent.py`, `python/autonomousim/tasks/swarm.py`, formation goals):
- **Scenario**: goal kind `formation` gives agent `k` of a group slot `k` of a `grid` (rows of ⌈√n⌉) or `circle` layout, neighbours `spacing` apart. The layout is centred on the centroid of the group's spawns and turned by a random heading from `yaw_deg`, and each slot sits `agl` above the surface (ground vehicles: on the terrain). Spawn option `cluster` (m) confines a group's random spawns to a square placed at random inside the region each episode. Unit test `cluster_spawns_and_formation_slots`.
- **`SwarmHover` (`swarm_hover`)**: 8 cf2x drones in `velocity` mode spawn in an 8 m cluster, 1.5 m apart, and fly to the slots of a 2 m grid. Slots are assigned by index, so paths cross. Observations are all in the heading frame, the frame the velocity commands use: goal, rot6d, velocity, rates, last action, 3 `neighbors` within 10 m and `nearest_agent` (41 values). Per-agent reward (`FormationHover`): `exp(−e) − 0.05·e − 1.0·max(0, 1 − d/1.5 m)` minus the spin and smoothness terms, where `d` is the `agent_clearance`; −5 when an agent stops early. `formation_error(state)` returns every agent's distance from its slot.
- **IPPO**: one policy per group, shared by its agents. Every (world, agent) slot is a sample stream. A stopped agent's samples are masked out of the loss, the observation statistics and the reward scaler (`RewardScaler` takes a `mask`). A slot's GAE chain ends when its agent stops or its world's episode ends; agents cut off by the time limit bootstrap from `final_obs`. `--total-timesteps` counts agent steps; `--time-limit` stops after that many minutes. The network, normalisation and checkpoint format are those of `ppo_continuous.py` (imported from it), saved as `policy_<group>.pt` and, for one group, `policy.pt`. `evaluate_multi` runs one deterministic episode per world: per-agent return, success and contact rate (`CRASH_AGENT` or zero `agent_clearance`), and for formation tasks the final slot error and `formation_success` (mean error < 0.3 m and no contact).
- **Goldens**: physics, observations and events of all four golden scenarios are unchanged (compared with the recordings left out); `/meta` carries the new scenario fields, so the goldens were re-blessed.
- **Result** (laptop, performance profile, 64 worlds × 8 drones, 9 sim and 3 torch threads, 15 min, 24M agent steps at 27k agent steps/s): on 128 unseen episodes the mean final formation error is 0.038 m, and 96.1 % of the episodes end in formation without any contact (0.98 % of agents touched another). A first run with a weaker proximity term (weight 0.5 within 1 m) reached 94.5 %; all its contacts were crossing collisions at about 2 m/s. The PPO update takes about 90 % of the time; the simulation only about 11 %.
- Tests (`tests_py/test_swarm.py`): formation layout and centring through the environment, the proximity cost, and a short IPPO run whose checkpoint loads with `load_policy`.

**As built in step 7** (`SwarmWaypointForest-v0`, multi-agent export, recording and viewer support):
- **Task** (`swarm_waypoint_forest`, `SwarmWaypointForest` + per-drone `SwarmForestDrone`): 8 iris-like drones in `velocity` mode on the wild 512 m map pool, wind 0–3 m/s, 60 s. The group spawns in a 20 m cluster, 4 m apart, with 2 m clearance; each drone then flies its own chain of `QuadWaypointForest` waypoints. Observations: the forest task's 148 values, then 3 `neighbors` within 20 m and `nearest_agent` (170). Reward: the forest task's, minus `agent_weight·max(0, 1 − d/agent_distance)²` (2.0, 4 m) and `agent_crash_penalty` (50) on `CRASH_AGENT`, on top of the terminal penalty.
- **Spawn fixes** (`SpawnSpec::sample_positions`, found by training):
  - The cluster centre is the most open of up to 32 random centres (obstacle clearance), so the group does not start inside dense forest.
  - Openness is weighted by the dry share of the square (5 × 5 probes). Otherwise lakes, being free of trees, won; the group was then squeezed onto the shore, in 14 of 64 evaluation worlds with drones as close as 0.36 m.
  - When no point satisfies everything, separation now outranks clearance in the fallback. Before, foliage capped every point's score at 0.5 and agents could start in contact.
  - Tests: `cramped_spawns_keep_their_separation` and `clusters_form_on_dry_land`; both fail without their fix. The goldens are unchanged.
- **Training** (`ppo_multiagent.py`):
  - The budget counts agent steps (`while agent_steps < total`); the learning rate anneals by the larger of the step and time fractions.
  - `--init` starts every group from a checkpoint, `ppo_continuous.py`'s included. Observations the checkpoint lacks at the end of the vector get zero input weights; their normalisation statistics are estimated from 50 warm-up steps. Test: `test_warm_start_from_a_single_agent_policy`.
- **Export and recording**:
  - `export_policy.py` exports multi-agent checkpoints: the task and its options come from the checkpoint, and the group name goes into the JSON. The samples are checked against the Rust `ObsSpec`.
  - `eval_record.py` records and evaluates multi-agent tasks. It reads back every agent's states and re-simulates the file bit for bit, including agents stopped by Python: a drone disabled after step s shows `disabled` with only `DISABLED` in state row s + 2.
- **Viewer**:
  - The autopilot stops agents at their last goal, as the environments do (`after_tick`), and counts agents that hit another one; the HUD shows it.
  - Ignored tests fly an exported policy (`exported_policy_success_rate`, all agents by default) and play a recording (`recorded_file_plays`, `AUTONOMOUSIM_RECORDING`).
- **Pipeline check** (warm start from the single-drone forest policy, 60 min, 32M agent steps; not tuned further):

  | Check | Result |
  |---|---|
  | Python evaluation, 64 worlds × 8 drones, map seed 1000 | 88.9 % finish; 3.9 % hit another drone, mostly level crossings at 2–7 m/s |
  | Exported policy flown in Rust, 4 × 64 drones | 202 of 256 finish; 8 hit another drone |
  | Recording | 2 episodes replay bit for bit |

  A from-scratch run (hidden 256, 15M agent steps) reached 67 % with 7 % agent crashes. The policy targets are left to the user's own training.

## Milestone 4: Rural maps, trucks and trailers, tracked vehicles

Planned 2026-09-25. The roadmap's M4 is about three M2-sized parts, so it is split into sub-milestones. Each closes with its own tests, a demo task trained just far enough to exercise the new features, a commit and a push. Decided with the user at the start:
- **Split and order**: M4a rural maps → M4b trucks and trailers (they drive and reverse on the rural roads) → M4c tracked vehicles and soft soil (they cross the rural fields and mud).
- **Demos**, one per part: `RoadFollowRural-v0` (a car follows the road network to a farm), `TrailerReverse-v0` (reverse a truck and trailer into a farmyard bay), `TrackedCrossCountry-v0` (an APC crosses soft fields and ditches to waypoints).
- **Training**: only enough to test what was built (export, viewer, replay); the user trains the agents.

**Already in place**: the `wild` generator (terrain noise, erosion, Priority-Flood lakes, moisture, materials, scatter, cache and hashes), 15 materials including asphalt, gravel, dirt and concrete, cuboid obstacles, `DriveGrid` with A*, wheeled vehicles as Featherstone trees on a Free chassis (`KcTravel` suspension, per-axle steer shares, independent steering), and multi-agent support.

### M4a: Rural maps

#### Design
- **Road network** (`world::roads`, part of `StaticWorld`, stored in map files and hashed):
  - Roads are polylines resampled every 1 m from smoothed splines. Each has a class (`paved` 6 m, `gravel` 4 m, `track` 3 m), a width, and nodes at junctions and ends. Farm yards and field gates are node kinds.
  - Queries use a segment grid: nearest road point (station along the road, signed lateral offset, heading, curvature), whether a point is on a road, and routes over the graph (Dijkstra by length) as polylines.
  - `FORMAT_VERSION` 2; maps without roads read back with an empty network.
- **Rural generator** (`procgen::rural`, `RuralConfig`, presets `training` 512 m and `showcase` 2 km; `RURAL_VERSION` 1):
  1. **Terrain**: the wild pipeline's noise, erosion and hydrology with gentle relief (0–60 m), broad valleys and a few lakes or ponds.
  2. **Farm sites**: Poisson-disc samples on flat, dry, low ground (slope < 5°, not near water), 3–6 per km².
  3. **Road routing**:
     - A main paved road crosses the map edge to edge. Gravel roads join each farm to the nearest existing road. Tracks run from the roads to field gates.
     - A* runs on a 4 m cost grid (length, grade above the class's limit, side slope, water forbidden, turning cost from 16 directions), reusing edges already built so roads merge.
     - The paths are smoothed into centripetal Catmull–Rom splines, limited in curvature per class, and resampled.
  4. **Terrain blending**: along each road the cross-section is flattened to the centreline height with a crown, and shoulders are blended with a smooth falloff over 3 m (cut and fill). The centreline profile is smoothed until the grade stays below the class limit (paved 8 %, gravel 12 %, track 20 %). The blend runs on the final grid and hydrology is not rerun, so roads never cross water; there are no bridges in M4a.
  5. **Materials**: road cells get asphalt, gravel or dirt. Farm yards get concrete. Fields are parcels of 1–6 ha, cut from the land between roads by a jittered Voronoi partition, each with a crop material: meadow, crop, or plowed soil (new; plowed soil is the soft soil of M4c). Grass verges and forest floor in the woods.
  6. **Scatter**, as obstacles with tags:
     - Farm buildings (cuboids: house, barn, shed; cylinders: silos) around a concrete yard facing its road.
     - Hedgerows (soft foliage, like canopies, with a woody core) and wire fences (thin solid cuboids with gaps at the gates) along parcel edges.
     - Tree lines along some roads; small woods from the wild tree scatter on steep or wet parcels; scattered field trees and rocks.
  7. Map hash, cache and golden hashes as for `wild`.
- **Simulation and Python**:
  - `MapSource::Rural(RuralMaps { seed, count, preset, config })`; Python `map="rural"`.
  - `DriveGrid` gains a road preference: an optional per-material cost in A*.
  - **Spawns and goals**: spawn option `on_road` (on a road, heading along it, in the right-hand lane when the road is paved). Goal kind `route`: a destination (a farm yard or a random road point, at a given route length), with goals every `spacing` m along the route polyline.
  - **Observation terms**: `road` (signed lateral offset and heading error to the route's lane, plus curvature samples at 5, 10, 20 and 40 m ahead), `route` (the next route points in the heading frame), and `on_road` (1 when on the road surface).
- **Viewer**:
  - Road ribbons drawn from the network (slightly above the terrain, with crisp edges and a centre line on paved roads) over the per-cell colours.
  - Box and cylinder meshes with roofs for the buildings; hedge and fence meshes; field colours.
  - `--map rural` for live mode, a rural `showcase`, and the route drawn in `policy` mode.
- **Demo**, `RoadFollowRural-v0`: the sedan (`vk` actions) starts on a road and follows its route to a farm yard 150–400 m away. Reward: progress along the route, minus lateral deviation and heading error, minus a penalty off the road surface; success on reaching the yard; terminal on leaving the road by more than 3 m, a crash or a rollover. Observations: `road`, `route`, speed and yaw rate, and `rl64` LiDAR for the obstacles.

#### Implementation order
| # | Step | Done when |
|---|---|---|
| 1 | `world::roads`: network type, segment grid and queries, routes; map file version 2 (done 2026-09-25) | Queries match brute force on random networks; routes are shortest; old map files still load; map files round-trip with roads |
| 2 | `procgen::rural` terrain, farm sites, road routing, terrain blending, road materials; `RuralConfig` and presets (done 2026-09-25) | Every farm is connected; grades and curvatures stay within the class limits; roads stay out of water; the hash is identical with 1 and 12 threads; golden hashes committed; 512 m in < 1.5 s, 2 km in < 20 s |
| 3 | Fields, new materials (meadow, crop, plowed soil), buildings, hedges, fences, tree lines, woods (done 2026-09-25) | Parcels cover the farmland; no obstacle on a road or in a yard; gates connect tracks to fields; invariants tested; goldens re-blessed |
| 4 | Simulation: `MapSource::Rural`, `on_road` spawns, `route` goals, `road`/`route`/`on_road` terms, road cost in `DriveGrid` (deferred); Python `map="rural"` (done 2026-09-25) | A car spawned `on_road` sits in its lane; routes follow the roads; terms match references; the Python env runs on rural maps |
| 5 | Viewer: road ribbons, buildings, hedges and fences, `--map rural`, route display (done 2026-09-25) | A rural showcase renders at ≥ 60 fps at 1080p "medium" on the Iris Xe; the car drives on the roads by keyboard |
| 6 | `RoadFollowRural-v0`, a short training run, export, viewer, replay (done 2026-09-25) | The task trains end to end; the exported policy drives in the viewer; a recorded episode replays |

**As built in step 1** (`world::roads`, `StaticWorld::with_roads`/`roads`, map file format 2):
- **Types**: `RoadNetwork` holds `RoadNode`s (position and kind: junction, end, yard, gate) and `Road`s (class, width, start and end node, and a `Polyline`). `Polyline` stores cumulative horizontal stations and answers `point_at`, `heading_at`, `curvature_at`, `project` (station, closest point, heading, signed lateral offset positive to the left, distance) and `slice` (forward or reversed). Curvature is that of the circle through the points 2 m either side, which is smooth on 1 m polylines; per-segment headings were off by up to 25 %.
- **Queries**: a 16 m uniform grid over the segments (by bounding box) serves `nearest(p, max_dist)`, searching ring by ring and stopping once no ring can hold a closer segment; ties go to the lower road and segment. `on_road(p)` answers whether `p` lies within half a road's width. `route(from, to, max_dist)` projects both points onto the network and runs Dijkstra from both ends of the first road, remembering which end each node was reached from. It compares both ends of the last road and driving along a single road, and returns the polyline and the roads with their directions.
- **Map files**: format 2 appends the roads (postcard, after the map data) only when there are any. The content hash covers them only then, so maps without roads keep their hashes: the wild goldens and existing recordings are unchanged. Format-1 files, such as those already in the map cache, still load.
- **Tests**: `nearest` and `on_road` match brute force on 2000 random points around a lattice with gaps and a curved road; routes between random road points have the Floyd–Warshall length and are continuous (≤ 1 m gaps); disconnected parts have no route; files with roads round-trip, a format-1 file loads, and format 3 is rejected.

**As built in step 2** (`procgen::rural`, `MapCache::rural`, `autonomousim mapgen --generator rural`):
- **Terrain**: `wild::landform` (noise, erosion, upsampling, lakes) is shared by both generators, and the wild goldens are unchanged. The rural presets use relief 100 (training) and 120 (showcase) with few mountains. That gives height ranges of about 25–35 m. A relief of 50 gave only about 15 m and never tested the grade limits.
- **Farm sites**: jittered-grid candidates in a seeded random order are accepted greedily at least `spacing` apart (160 m training, 260 m showcase). A site needs dry ground (30 m from water), a margin of 60 m from the edge, and less than 5 m of relief over the yard. With relief 100 and a 3 m limit, most 512 m maps had one farm or none. The result is 2–4 farms per training map and about 15–20 on the showcase.
- **Routing**: A* over (vertex, direction) states on a 4 m grid, with 16 directions including knight moves. Turns are at most 67.5° per step, and a turn costs its squared angle. A move costs length × (1 + 60·grade² + 400·(grade above the class limit) + 10·side slope²). Vertices within the water clearance are blocked, and so are the intermediate vertices of knight moves. The heap breaks ties by state, so paths are deterministic. The main road links opposite edges (16 attempts). Farms are connected nearest-first by gravel roads to any vertex of the network, with a chamfer-distance heuristic (× 0.92). The contact vertex becomes a junction.
- **Splines**: the paths are split at node vertices. Control points every 12 m, with the ends pinned, feed a centripetal Catmull–Rom resampled every metre. The interior controls are Laplace-smoothed until the curvature more than 10 m from the ends is within 1/min_radius.
- **Profiles**: node heights are the terrain mean over a 5 m ring; yard nodes use the mean over their yard. Each road takes the terrain along it, averages it over 20 m, and eases it onto the node heights over 15 m. It is then clamped into the band reachable from both ends at the class's maximum grade, and clamped forward and backward. Blending a forward and a backward clamp (the first attempt) overshot the limit by up to 20 %. The projection holds it exactly unless two nodes are too steep relative to each other.
- **Blending**: each vertex takes the nearest road within reach. The target is the road height minus the crown. The weight is 1 up to half the width + 0.5 m, then a smoothstep over max(3 m, 1.5 × the cut or fill). Yards (oriented rectangles along their road) are levelled the same way, and the larger weight wins.
- **Materials**: lake beds are sand or mud; yards concrete; road surfaces asphalt or gravel; steep slopes rock; shores sand; wet flats mud; everything else grass (fields come in step 3).
- **Measured** (laptop, release): a 512 m map in 0.11–0.15 s; the 2 km showcase in 2–3.5 s, about 0.8 s of it routing.
- **Tests** (`crates/procgen/tests/rural.rs`, goldens in `fixtures/golden_hashes_rural.toml`):
  - For 8 training seeds: every yard can be routed to both ends of the main road; grades are within the limit; curvature is within 1.15/min_radius more than 12 m from the nodes; no water under the road or its edges; the terrain at the centre line is within 8 cm of the road away from nodes; road cells carry the road's material; yards are flat concrete.
  - Hashes are identical with 1 and 12 threads.
  - Config round-trip and validation.
  - Cache round-trip with roads.
  - An ignored release test times the showcase and checks its invariants.

**As built in step 3** (`procgen::farmland`, `RURAL_VERSION` 2):
- **Materials**: `MaterialId::{MEADOW, CROP, PLOWED}` (15–17) live in `MaterialTable::rural()`, the standard table plus three entries. The table is part of the content hash, so leaving `standard()` alone keeps the wild goldens and existing recordings unchanged.
- **Parcels**: the Voronoi cells of a jittered grid of centres (120 m training, 150 m showcase; jitter 0.8), extending one cell beyond the map. `locate(p)` returns the parcel and the distance to its edge from a 5×5 neighbourhood, per cell and in parallel. Parcel kinds are meadow, crop or plowed by share. A parcel is woods with probability 0.08, or always when its centre is steeper than 15°. A 10° threshold turned 40 % of the map into woods.
- **Materials per cell**, in order of precedence: lake beds; yard concrete; road surfaces; rock on steep ground; shore sand; marsh mud; grass on 2 m verges beyond the road edges, on the farm pads and on 2 m headlands along parcel edges; otherwise the parcel's material. Parcels cover 55–75 % of a map.
- **Tracks**: every non-woods parcel whose centre lies on the map (20 m inside), away from the farms and more than 50 m from any road gets a dirt track. Tracks join the nearest road, nearest first, and end at a `Gate` node at the parcel centre. The field gates are the gaps where tracks cross hedges.
- **Farm pads**: routes avoid a circle around every other farm (cost × 20). The own farm's road may cross its pad, and field tracks avoid all pads. The yard is levelled together with a 14 m pad around it, which holds the buildings.
- **A\* workspace**: it is reused between runs, and only the touched states are reset. Without that, clearing the 16 × 513² state arrays for about 150 tracks would dominate the showcase.
- **Obstacles** (new tags `HEDGE`, `FENCE`, `BUILDING`, `SILO`):
  - Buildings stand outside the yard on its pad, away from the road side: a barn behind, a house on one side, a shed on the other, and sometimes a silo.
  - Parcel edges are found by sampling each neighbour pair's bisector from the midpoint. 40 % carry a hedge (3 m foliage cuboids 1.4 m wide and 1.6–2.8 m tall, each around a solid 0.3 m woody core), 30 % a wire fence (4 m solid cuboids, 0.1 m thick and 1.2 m tall), and the rest stay open. Edges between two woods stay open.
  - 35 % of roads get broadleaf tree lines (12–17 m tall, 3.5–4.5 m beyond the edge, every 12 m).
  - Woods and single trees on grass come from the wild tree scatter.
- **Clearance**: every obstacle group (a tree's trunk and crown, a hedge's foliage and core) is dropped as a whole unless all its members pass. A member passes when:
  - it keeps 1.5 m from road edges, unless its lowest point is 4.5 m above the road (tree-line crowns);
  - it stays out of yards and water and on the map.
  
  Round shapes are checked with their bounding circle. Cuboids use circles about 1 m apart, because a single bounding circle rejected most buildings next to their yards.
- **Measured**: a 512 m map in about 0.2 s (≈ 1,000 hedge and 250 fence pieces, 200–1,400 trees, 8–10 buildings); the 2 km showcase in about 5 s (routing 2.1 s with 74 tracks, materials 1.1 s).
- **Tests**, added to the rural invariants for 8 seeds:
  - field gates are routable to both ends of the main road;
  - parcels cover more than half the map;
  - every farm has buildings; hedges, fences and trees exist;
  - no obstacle footprint point lies on a road below 4 m above it, in a yard or in water;
  - the statistics match the obstacle tags;
  - the scatter configuration is validated.
- `autonomousim mapgen` prints the farmland statistics, and its preview draws hedges, fences, buildings and silos.

**As built in step 4** (`sim::lane`, `MapSource::Rural`, spawn `on_road`, goal kind `route`, terms `road`/`route`/`on_road`):
- **Maps**: `MapSource::Rural(RuralMaps { seed, count, preset, config, cache })` builds the pool like `Wild`, through `MapCache::rural`. `MapSource::{seed, set_seed, set_pool}` replace the viewer's wild-only matches. Python `map="rural"` gives a pool of 512 m training maps.
- **Lanes** (`sim::lane`): traffic keeps right. On paved roads the lane centre lies 0.25 × width right of the centre line; gravel roads and tracks are single-lane. `lane_line` moves a centre-line route onto the lanes of the roads it follows.
- **Spawns**: with `spawn.on_road` (it needs `on_ground`, which ground vehicles always have), each agent draws a road point uniformly by length, 8 m from the road ends. It keeps the best of up to 64 draws for `min_separation`. Without route goals it faces along the road in a random direction, in its lane. `layout`, `region`, `cluster` and `yaw_deg` are ignored.
- **Route goals** (`goals.kind = "route"`, `goals.route = { destination = "yard" | "road", step = 25 }`):
  - Candidate destinations are every farm yard, or 32 random road points, in a random order. The route (`RoadNetwork::route`) whose length is in `distance`, or else closest to it, wins.
  - Goals lie every `step` m along its lane line, the last at the end, at the ride height above the terrain. `count` is ignored.
  - With an `on_road` spawn the route starts at the spawn point and fixes its heading; otherwise it is planned from the spawn position. With no route found, the agent gets the spawn goal.
  - The agent keeps the lane line (`Agent::route`, `Arc<Polyline>`); `set_goals` clears it.
  - Small maps have few yards, so routes may miss the range: 186–445 m for `[150, 400]` in the test.
- **Terms**: `road` (6 values: lateral offset from the lane, positive left; lane heading − heading; lane curvature 5/10/20/40 m ahead), `route` (8 values: lane points 5/10/20/40 m ahead in the heading frame) and `on_road` (1).
  - `road` and `route` follow the route's lane, or else (`Follow`) the lane of the nearest road within 30 m in the travel direction closer to the heading. They read 0 with neither.
  - Projecting onto the whole route can jump where a route passes close to itself; rural routes do not, so this is left alone.
- **Validation**: `on_road` without `on_ground` fails, and so do `on_road` spawns or `route` goals on a pool with a map without roads.
- **Deferred**: the road preference in `DriveGrid` (per-material A* cost). No step-4 goal kind routes over the drive grid: `route` goals follow the road network itself, and `random` goals keep the terrain-only cost. It comes back if a task needs off-road goals that prefer roads.
- **Goldens**: the trajectories are unchanged (checked with the recorder output left out of the hash). The scenario JSON in the recorded `/meta` gained the new fields, so `golden_trajectories.toml` and `recordings/hover.mcap` were re-blessed.
- **Tests** (`crates/sim/tests/roads.rs`, `tests_py/test_envs.py`):
  - Over 4 episodes on 2 maps, sedans spawned `on_road` sit on the road, facing along it, in their lane (within 0.6 m).
  - Routes start at the spawn and every metre of them lies on a road. They end at a yard; the goals lie on them every 20 m at the ride height.
  - The terms match `Follow` and read the lane at the start of a route. Without a route they follow the road in the travel direction. Far from roads there is no reference.
  - The validation errors fire.
  - The Python `BatchSim` builds and caches a rural pool with these terms.

**As built in step 5** (`scene::roads`, `scene::props::obstacle_visual`, viewer `--map rural`):
- **Road ribbons** (`roads_by_chunk`):
  - Strips follow each road's 1 m polyline. Every vertex is a few centimetres above the terrain below it: paved 6 cm, gravel 5 cm, tracks 4 cm, so paved roads cover the roads that join them. Paved and gravel surfaces are sampled across the width (5 and 3 columns) so they follow the crown.
  - Paved roads get dark asphalt, white edge lines and a dashed centre line (3 m dashes every 9 m). Gravel roads take the gravel colour. Tracks get two darkened dirt ruts at 0.55–1.05 m either side of the centre, where the wheels run.
  - Roads are cut into runs per terrain chunk (by segment midpoint) and merged per chunk.
  - The viewer draws them with a `depth_bias` material, without shadows, within the obstacle detail distance (400 m at "medium"). Beyond that the terrain's road materials suffice.
- **Farm obstacles** (`obstacle_visual`, visual only; collision shapes unchanged):
  - Buildings get a gable roof along their long side (rise 0.5 × the short half-width, 0.3 m overhang) above the collision box. House walls are plaster with red tiles, barns wood with dark red, sheds metal with grey.
  - Silos get a conical cap.
  - Fences are drawn as two posts and three wires instead of a solid 0.1 m wall.
  - Hedges are dark green; their woody cores are not drawn.
- **Live mode**: `--map wild|rural` (the default is `wild`); `--preset` is parsed for the chosen generator. On rural maps a ground vehicle spawns `on_road` with `route` goals to a farm yard 150–400 m away. `--demo` then follows the route lane by pure pursuit, 8 m ahead, at up to 12 m/s and slower in bends (2 m/s² lateral), and starts a new episode at the end of the route.
- **Overlay**: an agent's route lane is drawn 0.5 m above the road (orange), in live and policy modes, and in replays since step 6.
- **Measured** (Iris Xe, 1080p, "medium", 2 km showcase, sedan demo): 77–105 fps over five runs, with 109k road triangles and 809k obstacle triangles near (443k far). An iris-like drone demo runs at 87 fps. The map mesh builds in 0.16 s.
- **Tests**: ribbons on a flat test world lie at least 3.5 cm above the terrain, face up, cover the paved road's area (180 m × 6 m) and exist exactly in the chunks the roads cross. Roofs and walls face outwards; the ridge height, overhang, fence extent and silo cap are checked.


**As built in step 6** (`RoadFollowRural-v0`, state column `road`, routes in recordings, two map fixes):
- **State**: rows gain `road` (3 values, `lane::road_state`): the lateral offset from the lane, the heading error, and the distance off the road surface (0 on it, capped at 30 m). They are computed like the `road` term: along the route's lane, or else the nearest road in the travel direction. `STATE_DIM` is 24. Trajectories are unchanged; the goldens were re-blessed for the new `/meta`.
- **RoadFollowRural-v0** (`tasks/road_follow.py`):
  - **Setup**: `sedan_like` in `vk` mode, up to 15 m/s. The car spawns in the lane of a random road on a pool of 16 rural maps, with `route` goals every 20 m (radius 5 m) along a 150–400 m route to a farm yard. The policy runs at 20 Hz and the physics at 1 kHz; episodes last 60 s.
  - **Observation** (97 values): `road`, `route` (×1/20), `on_road`, speed, body velocity and rates, steering, last action, and a roof LiDAR (2 rings at ±3°, 36 azimuths, 40 m, log ranges).
  - **Reward**:
    - progress to the current goal, +1 per goal and +20 at the yard;
    - −0.1·|lane offset| and −0.1·|heading error|;
    - −0.5 per metre off the road;
    - smoothness 0.02‖Δa‖²;
    - −20 on a terminal event, on `STUCK` (5 s), or at more than 3 m off the road.
- **Route fallback**: some maps have roads but no farm (map 8 of the pool with seed 1000). `yard` routes then go to a random road point, the same as `destination = "road"`. If no draw finds a route at all, a road spawn falls back to a lane spawn with the spawn goal; before this it panicked.
- **Recordings**: `/episode` carries each agent's route lane (`route`: points), only when it has one, so older files and the hover fixture are unchanged. `RecordedEpisode::routes` reads it back, and the viewer's replay draws it.
- **Map fixes** (`RURAL_VERSION` 3; golden hashes re-blessed). The first trained policy crashed on two kinds of terrain fault, both invisible to the invariants, which only checked the centre line:
  - **Yard pads beside roads**: blending took the stronger of the road and yard weights. Where a road passed a pad 1.7 m higher, the pad won right at the road's edge and left a step where the wheels run. Now the pads are blended first and the roads on top, so each road keeps its surface out to 0.5 m beyond its edges.
  - **Steps at junctions**: a road whose end nodes differ in height by more than its grade limit allows could not meet both. Its grade-limited profile pulled one end away from its node, leaving a 1.25 m step in the main road that launched the car at 11 m/s. Now `reach_nodes` first moves the free (non-yard) node heights until every piece fits within 90 % of its grade limit. A piece that still cannot fit (between two yards) becomes a straight ramp: continuous, over the grade.
  - **Invariants added**: every road meets its nodes; the terrain matches the crowned surface across the width (8 cm) and 0.5 m beyond each edge (15 cm), away from nodes and other roads. Over both 16-map pools (seeds 0 and 1000), the worst deviation 0.5 m beyond an edge is now 0.19 m away from nodes, down from 1.7 m.
- **Scripted driver** (test): it steers at the lane point 5 m ahead, at up to 9 m/s, slowing to keep lateral acceleration at 2 m/s² for the curvature within 20 m. It reaches every yard in the test with a mean lane offset of 0.21 m; junction corners are cut by up to 3 m. Steering at the 10 m point cut corners by 1.3 m (90th percentile), and without slowing it ran off at junctions.
- **Training** (`ppo_continuous.py --hidden 256 --bound-coef 0.01`, 5M steps, 256 envs, 14 min at 5.9k SPS; basic training only):
  - Success on the training maps was 86 % at the end.
  - Deterministic evaluation over 256 episodes on the fixed maps: **82.8 % success on unseen maps** (`map_seed` 1000) and 87.9 % on the training pool.
  - In 20 recorded unseen episodes, 3 failed: one drove off a paved road into the ditch, one cut onto a track's edge at a field gate, and one got stuck.
- **Export and viewer**: `export_policy.py` wrote `policy.json`. `autonomousim-viewer policy <json> --map-seed 1000` drives it at about 136 fps. `eval_record.py` recordings replay bit for bit, with the route drawn, at about 146 fps.
- **Tests**:
  - Rust: state rows against `road_state`; routes survive a recording; yard-less maps route to road points; the new map invariants.
  - Python: the scripted driver reaches every yard; a full-lock car fails off the road; spaces and `check_env` for the new id.

### M4b: Trucks and trailers

Planned 2026-09-26. Decisions taken at the start (the user asked to go on; open to change):
- **Trailers are their own definitions**, coupled in the scenario (`vehicle = "truck_6x4"`, `trailers = ["semitrailer_3axle"]`), so any towing unit pulls any trailer with a matching coupling.
- **Truck tyres**: the 315/80 R22.5 truck example of Pacejka's book in PAC2002 form (Chrono's `CityBus_Pac02Tire.tir`, BSD-3), as `Truck_Pac02Tire`. With the same tyre in both, high-speed runs compare like for like with Chrono.
- **Demo**, `TrailerReverse-v0`: the 6×4 tractor and 3-axle semitrailer start in a farm yard facing the exit and back the trailer into a bay at the far side of the yard, in front of the buildings.

#### Design
- **Units**: a wheeled vehicle is a chain of units: the towing unit (today's chassis and axles) and up to three trailer units. Each unit is one body of the multibody tree, so articulation needs no loop constraints. Axles, wheels, colliders, drag and contacts attach to their unit's link. Wheels and axles are numbered over all units in order, towing unit first.
- **Couplings** (on the trailer, with the coupling point on the towing unit given by the towing unit's `fifth_wheel` or `hitch`):
  - `fifth_wheel`: a Spherical joint at the kingpin. Roll against a torsional spring and damper (the plate); pitch and yaw free up to end stops (±15° pitch, ±90° yaw), which are stiff torsional springs.
  - `drawbar`: a two-unit trailer. A dolly (drawbar and front axle) hangs on the pintle hitch (Spherical, stops in pitch and yaw, a light roll spring), and the body sits on the dolly's turntable (Revolute about z). A centre-axle trailer is a drawbar trailer without a turntable (the drawbar is part of the body).
  - Articulation angles (yaw of each unit relative to the one ahead) are reported per coupling.
- **Limits lifted**: up to 8 axles over all units (`MAX_WHEELS` 16). Per-wheel arrays stay fixed-size (`Copy`, no allocation per tick); per-wheel masks become `u32`.
- **Static equilibrium for any layout**: for chains or more than two axles, minimise the potential energy (tyre and spring energies, gravity, coupling springs) over the towing unit's height, pitch and roll, each trailer's pitch and roll about its coupling, and the suspension travels (Newton with finite differences; at most about 30 unknowns). Automatic preloads fix the travels at zero and read the preloads from the tyre loads, as today. Single-unit two-axle vehicles keep the current solver (so their goldens stay), and a test checks that both solvers agree.
- **Steering**:
  - Several steered axles: each steered axle turns about the virtual rear axle (the mean of the unsteered axles), `tan δ_i = (x_i − x_r) tan δ / (x_1 − x_r)`, with the Ackermann fraction applied per wheel. `steer` stays the share for a hand-set layout; `steer = "geometric"` computes it.
  - Trailer axles: `steer_mode = { articulation = k }` steers the axle by `k` × the unit's articulation angle (forced steering, as on long semitrailers; with articulation negative in a left turn, the axle steers right, so the trailer's rear swings out onto the tractor's path). Passive self-steering axles are deferred.
- **Brakes**: air brakes with a delay and a first-order lag (`brake = { max_torque, delay, time_constant }`, default instant), trailer axles braked from the same pedal.
- **Presets** (Chrono data where available, extracted by `tools/gen_chrono_truck_fixtures.py` like the sedan):
  - `truck_6x4`: the Kraz 64431 tractor: steered front axle, driven rear tandem, fifth wheel.
  - `semitrailer_3axle`: the Krone semitrailer Chrono pairs with the Kraz.
  - `truck_8x8`: the MAN 10t: two steered front axles, eight driven wheels.
  - `farm_tractor` (rigid rear axle, front axle on a pendulum pivot as a roll-free joint, big Fiala tyres, rear pintle hitch) and `farm_trailer` (a 2-axle drawbar trailer). No Chrono models; the parameters are published figures for a 100 kW tractor and a 10 t trailer.
- **Simulation**:
  - Scenario groups take `trailers = [...]`; the combined definition is stored in recordings.
  - Agent contact shapes get per-sphere velocities and links, so forces reach the unit they hit.
  - Events: `ROLLOVER` from any unit, and `JACKKNIFE` (new) when an articulation angle exceeds `jackknife_deg` (default 70°).
  - State column `articulation` (2: yaw of the first two couplings); obs terms `articulation` (angles and rates) and `trailer_goal` (the goal in the last unit's frame, with its heading error).
  - Recorded states carry the coupling joint angles; replays place every unit.
- **Viewer**: trailer bodies, wheels on their units, articulation in the HUD, a reversing camera (C cycles to it), `--trailer <name>` for live driving, replay.
- **Validation**:
  - Low-speed offtracking: steady circles at 2 m/s; the trailer's path radius against the kinematic steady-state formula `R_t² = R_k² − L_t²` (semitrailer), and the chain of the same for the drawbar trailer; within 2 %.
  - `truck_8x8`: steady turning against Chrono's MAN 10t (same tyres and geometry).
  - Tractor-semitrailer: a step steer and a single lane change at 60–80 km/h against Chrono's Kraz with the truck tyre fitted to both: yaw rates, articulation angle and rearward amplification (trailer over tractor lateral acceleration).
  - Statics: loads sum to the weight, the kingpin load matches the lever rule, a rig rolls straight and brakes straight.
- **Demo**, `TrailerReverse-v0`:
  - Goal kind `bay`: a pose at the far side of a farm yard (from the yard node and its access road), with the rear of the last unit as the reference point; spawns in the yard facing the exit, 15–30 m ahead of the bay, with lateral and heading jitter.
  - Actions `vk` (reverse up to 3 m/s). Observation: `trailer_goal`, `articulation`, speed, steering, last action, and LiDAR on the trailer's rear.
  - Reward: progress of the trailer's rear to the bay and heading alignment; success within 1 m and 5°; failure on a crash, jackknife or leaving the yard.
  - A scripted reversing controller (a feedback law on the articulation angle) proves the task solvable in the tests.

#### Implementation order
| # | Step | Done when |
|---|---|---|
| 1 ✅ | Units and couplings in `vehicles::ground` (fifth wheel, drawbar, turntable), per-unit colliders and drag, `MAX_WHEELS` 16, general static equilibrium, `trailers` composition | A test tractor with a semitrailer and a drawbar trailer settles at the static solution; loads and kingpin load match statics; the two solvers agree on two-axle vehicles; existing goldens unchanged |
| 2 ✅ | Truck tyre, multi-axle and forced steering, air-brake lag, presets `truck_6x4`, `semitrailer_3axle`, `truck_8x8`, `farm_tractor`, `farm_trailer` (Chrono extraction) | Presets load, settle and drive straight; static loads against Chrono's |
| 3 ✅ | Validation: offtracking, 8×8 turning and tractor-semitrailer manoeuvres against Chrono | Tolerances above met or explained |
| 4 ✅ | Simulation: scenario `trailers`, multi-body agent shapes, `JACKKNIFE`, `articulation` state and terms, `trailer_goal`, recordings with articulation | Rigs spawn, drive and record; replays reproduce them; goldens of existing scenarios unchanged |
| 5 ✅ | Viewer: trailers, reversing camera, HUD, `--trailer` | A rig drives by keyboard at ≥ 60 fps on the Iris Xe; recordings replay |
| 6 ✅ | `TrailerReverse-v0`: `bay` goals, scripted reversing controller, short training, export, viewer, replay | The task trains end to end; the exported policy reverses in the viewer; a recorded episode replays |

#### As built
- **Step 1** (units and couplings):
  - **Trailer files and composition**: `TrailerDef` (`type = "trailer"`, loaded by `TrailerDef::from_toml`) gives its `[coupling]` (`kind` = `fifth_wheel` | `drawbar`, the kingpin or eye position, optional overrides), chassis, axles, colliders, an optional rear `[hitch]` and an optional `[dolly]`. `WheeledDef::with_trailers(&[..])` checks each coupling against the hitch ahead (`hitch = { kind, position }` on the towing vehicle) and appends units to `WheeledDef::units`, and their axles to `axles` with `unit` set.
  - **Unit frames**: each unit's frame is moved to its joint centre. A dolly trailer becomes three units: the drawbar (80 kg bar, eye → hinge), the dolly (on a `Hinge` about y at the drawbar's hinge, so the pintle carries almost no vertical load) and the body (on a `Turntable` about z). Couplings are `Spherical` joints with `CouplingJoint::torque`: a roll spring and damper, and pitch and yaw stops, applied as generalised Euler-angle forces mapped through the Euler-rate Jacobian, so they are conservative (unit test).
  - **Defaults**: fifth wheel roll 2·10⁷ N·m/rad, ±15° pitch; drawbar eye roll 10⁴ N·m/rad, ±30° pitch; ±90° yaw; stops 10⁷ N·m/rad with 10⁵ N·m·s/rad damping.
  - **Shared tree**: the multibody tree is built by `ground::tree::build`, used by both `Wheeled` and the static solver. Link order is the towing unit and its wheels (unchanged, so single-unit goldens stay), then each unit and its wheels. `WheeledDef::unit_link(u)` predicts the link index, so `sphere_colliders()` carries per-unit links.
  - **Drag, wheels and API**: drag acts per unit. Trailer wheels start rolling from their centre velocity. New accessors: `Wheeled::{num_units, unit_pose, unit_link, articulation(u)}`. Articulation is the yaw relative to the unit ahead, positive when the unit points left, so negative in a left turn.
  - **Wheel limits**: `MAX_WHEELS` is 16 (8 axles); per-wheel drive masks are `u32`.
  - **Statics**: two-axle single units keep the lever-rule solver. Everything else uses `ground::statics::solve`, which minimises energy with a gradient from the virtual work over finite-difference motions of the tree (forward kinematics) and a Gauss–Newton Hessian of the tyre, spring and coupling stiffnesses. It handles automatic preloads with travels fixed at zero.
  - **Rest state**: computed once in `finish()` (`WheeledDef::rest_state()`); `rest()` and `reset()` use it and place the units at their rest pitch and roll. `StaticState::joints` holds per-unit pitch and roll.
  - **Tests** (`vehicles/tests/trailers.rs`, with a 4×2 test tractor, a tandem semitrailer and a dolly drawbar trailer):
    - the composition builds the right tree, and the JSON round trip is lossless;
    - both solvers agree on the sedan, the 4×4 and the skid rover (height and travel within 1e-6, loads within 1e-5);
    - the loads sum to the weight, and the kingpin load matches the lever rule within 1 %;
    - the rigs settle at the static state (loads within 0.5 %);
    - the rigs roll straight, and turning left gives negative articulation.
  - Existing goldens are unchanged.
  - **Tandem load split**: without load-equalising suspension, a tandem's axles share the load only roughly (the test rig splits 29.2 kN to 25.6 kN), because the trailer's pitch shifts load between them.
- **Step 2** (truck features and presets):
  - **Truck tyre**: `assets/tires/Truck_Pac02Tire.tir` is Chrono's `CityBus_Pac02Tire` (315/80 R22.5).
  - **Tyre mirroring**: Magic Formula tyres now know the side they were measured on (`TYRESIDE`, `MfParams::measured_right`), and a tyre mounted on the other side is mirrored (`MfParams::mirrored`, `Tire::on_side`). The same 18 asymmetric coefficients change sign as in Chrono's PAC2002 (conicity and ply-steer shifts, asymmetric curvatures). `Wheeled` keeps one tyre per wheel. Without mirroring, all tyres pulled the same way, and the Kraz drifted 1° off heading in 100 m.
    - Only the `cars` golden changed.
    - The sedan's high-a_y understeer gradient moved from 4.96e-4 to 5.21e-4 from Chrono's. The test bound is now the 5.3e-4 its doc states (10 % of 3°/g); it was 5e-4.
    - The HMMWV settles 0.3 mm off design travel (the mirrored shifts push the sides apart), so that test's bound went from 0.2 mm to 0.5 mm.
  - **Dual wheels**: `dual = <spacing>` on an axle gives each wheel two identical tyres (`Tire::dual`). Each tyre is evaluated at half the load and the forces and moments are doubled. `width()` is the pair's overall width and `section_width()` one tyre's. The viewer draws two tyres.
  - **Solid axles as equivalent independent corners**:
    - wheel rate = the spring rate (springs at the wheels);
    - the axle's roll stiffness comes from a negative `anti_roll`, `k((s/t)² − 1)/2` for spring track s and wheel track t (allowed with a rate spring while `rate + 2·anti_roll ≥ 0`; a pendulum axle has `anti_roll = −rate/2`);
    - dampers scale by the motion ratio: `c·r²`, `d·r`.
  - **Degressive dampers**: `degressivity_bump`/`_rebound` d gives `F = c·v/(1 + d|v|)` (Chrono's `DegressiveDamperForce`).
  - **Multi-axle steering**:
    - `steer` is a share or `"geometric"`, resolved in `finish()` to `share·(x_i − x_r)/(x_lead − x_r)` about `WheeledDef::steer_reference()` (the mean x of the towing unit's unsteered axles).
    - A geometric axle turns by `atan(ratio·tan(δ_lead))`, then takes the per-wheel Ackermann blend about its own distance from the reference.
    - Control and the bicycle wheelbase use `steer_reference()` and `share()`.
  - **Forced steering**: `steer_mode = { articulation = k }` on a trailer axle of a unit on a coupling or turntable. It steers by `k·articulation`, computed in `finish_step`. Trailer axles may steer only this way.
  - **Air brakes**: `brake = { delay, time_constant }` gives a per-wheel delay line of `round(delay/dt)` ticks, then a first-order lag. The parking brake acts at once.
  - **Automatic preloads per vehicle**:
    - `finish()` solves the preloads vehicle by vehicle: the towing unit alone, then each trailer (a coupling unit and the hinge and turntable units behind it) behind the vehicles ahead with their preloads fixed (`statics::solve(.., auto: Some(first unit))`).
    - So each vehicle's springs carry it at design height, as in Chrono. A tractor squats under the kingpin load rather than being preloaded for it.
    - Before this, the tractor's springs sat at zero travel with the trailer attached, and the Kraz rig's loads were 7 % off Chrono's.
    - `rig_statics` now checks the lever rule about the tandem's centre of load. The test rig's tractor squats 4 cm and the tandem splits 1.7 : 1.
  - **Presets** (`presets::wheeled`; trailers in `presets::trailer`/`trailer_names`, a separate list):
    - `truck_6x4`: Kraz 64431, 13.2 t, two dual driven rear axles, 0.1 s + 0.15 s air brakes, fifth wheel.
    - `semitrailer_3axle`: Krone, 22.2 t laden, 0.25 s + 0.25 s brakes.
    - `truck_8x8`: MAN 10t, 15.6 t, axle 2 geometric, limited-slip axle differentials (Torsen since step 3), 9-speed gearbox.
    - `farm_tractor`: 6 t, pendulum front axle, Fiala tyres, 4WD 40/60, 8 gears, drawbar hitch.
    - `farm_trailer`: 9.9 t, dolly on a turntable.
    - Truck parameters come from Chrono's C++ and JSON sources (`MAN_10t`, `Kraz_tractor*`, `Kraz_trailer*`).
  - **Fixtures**: `tools/gen_chrono_truck_fixtures.py` (`make fixtures-chrono`) writes `fixtures/chrono/truck_{kraz_tractor,kraz_rig,man_10t}.json`: design masses and spindles, static unit poses, loads and spring lengths, and steering locks. It uses the same truck tyre on every wheel.
  - **Tests** (`vehicles/tests/trucks.rs`):
    - The presets load, and the masses match Chrono's within 0.1 %.
    - Static per-wheel loads match Chrono's within 1 % (actual ≤ 0.1 %) for the Kraz alone and with its trailer, and for the MAN; pitch matches within 1 mrad.
    - The Kraz and MAN lead-axle locks match Chrono's within 0.4°.
    - The MAN's geometric second axle follows its law exactly. Chrono's linkage steers that axle by 1.04× the first, not geometrically; step 3 overrides the share for that comparison.
    - All five rigs settle at the static state, drive straight and brake straight. Heading stays within 0.01 rad and lateral drift within 1 % of the distance.
    - Air-brake torque follows delay plus lag within 2 %.
    - Forced steering (k = 0.5 on the rear axle) cuts the Kraz rig's offtracking on a 23 m circle from 1.67 m to 1.11 m.
    - Dual tyres and degressive dampers follow their laws.
- **Step 3** (validation):
  - **Reference runs**: `tools/gen_chrono_truck_handling_fixtures.py` (`make fixtures-chrono`) writes `fixtures/chrono/truck_handling.json`, with the truck tyre on every wheel and μ at the tyre's reference:
    - `man_constant_steer`: the MAN at a fixed steering input while the speed rises from 3 to 9 m/s (ISO 4138 style);
    - `kraz_step_steer`: the Kraz rig at 60 km/h, a steering step in 0.2 s (ISO 7401 style);
    - `kraz_sine_steer`: one period of sine steering at 0.4 Hz (ISO 14791 single sine).
    Chrono's rigs drift on the straight (3–5 m over 200 m), so a pure-pursuit driver on y = 0 and a speed PI (both simple enough to re-implement exactly in the test) hold the approach.
  - **Truck tyre check**: `Truck_Pac02Tire` joined the PAC2002 tyre fixtures (`tire_fixtures.rs`); our forces agree with Chrono's to about 1e-6 of the peak.
  - **Steering replay**: Chrono's steering linkages are compliant. At a fixed input, the MAN's road-wheel angle falls from 0.35 to 0.11 rad between 4 and 9 m/s, and its second axle steers about 0.85× the first (1.04 at lock). The tests therefore replay Chrono's measured lead-axle angle, and the MAN run sets axle 2's share to Chrono's mean ratio.
  - **Torsen differential**: `DifferentialDef::Torsen { bias }`, a coupling whose capacity each step is (B − 1)/(2(B + 1)) times the drive torque entering it, so the slower output takes up to B times the faster one's torque. `truck_8x8` now uses it with B = 2 on its axles, as Chrono's `SimpleDrivelineXWD`. With the old fixed 2000 N·m limited slip, the MAN's path curvature was up to 29 % off Chrono's at low speed (the locked axles push the truck wide); with the Torsen it is within 3.2 %.
  - **Tests** (`vehicles/tests/truck_handling.rs`, no drag, camber-free tyres as Chrono's PAC2002):
    - MAN constant steer: path curvature within 5 % over the speed sweep (actual 3.2 %).
    - Kraz step steer: tractor and trailer yaw rates and articulation, peak and steady values within 5–10 % and peak times within 0.25 s (actual about 1.5 %).
    - Kraz single sine: each lobe's peak within 10 % and 0.25 s. Peak lateral accelerations within 10 %. Rearward amplification is 1.09 against Chrono's 1.01 (absolute tolerance 0.1): our trailer's peak a_y is 7 % higher.
  - **Low-speed offtracking** (`trucks.rs`, steady circles at 2 m/s): chained from the tractor's rear, each unit's axle-line radius follows `R_p² = R_a² + x²` (pivot ahead of the towing unit's axle line) and `R² = R_p² − L²`. The Kraz rig (22.4 m circle) and the farm tractor with its dolly trailer (6.7 m) match within 1.1 % on radii (2 % tolerance) and within 4 % on offtracking (5 %).

- **Step 4** (simulation):
  - **Scenario**: groups take `trailers = ["semitrailer_3axle", …]` (built-in names, listed by `presets::trailer_names` and Python's `trailer_presets()`, or paths to trailer TOML files); `GroupSpec::resolve_vehicle` couples them. The composed definition goes into `/meta` as before. Only ground vehicles tow.
  - **Rig geometry**: `WheeledDef::{unit_origin, wheel_position_in_line, colliders_in_line}` give the design positions with all units in line in the towing unit's frame. Spawn clearance radius, half-width (drive grids) and the terrain-fitted spawn pose use them, so a rig needs a clear circle about the tractor that reaches past its trailer (conservative).
  - **Agent contacts**: `AgentShape::bodies` holds the units behind the towing unit (frame origin, velocity, angular velocity, link); each `Sphere` names its body. Point velocities use the sphere's body, and `AgentContacts::forces` carry the link, so `Wheeled::apply_force_on` pushes the unit that was hit. Wheel spheres belong to their unit. Units behind use the poses of the step's start (as the wheel spheres already did). Single-unit vehicles have no extra bodies, so their contacts are unchanged bit for bit.
  - **Events**: `ROLLOVER` from any unit; `JACKKNIFE` (bit 14, terminal) when a coupling's or turntable's yaw exceeds `events.ground.jackknife_deg` (70°).
  - **State**: columns `articulation` (2: yaw of the first two trailers or dollies relative to the unit ahead, from `Wheeled::articulations`) and `tail` (3: x, y and heading of the last unit's tail, `Wheeled::tail_pose`; position and heading for other vehicles). `STATE_DIM` is 29. The tail (`WheeledDef::tail`: the rearmost extent of the last unit's wheels and colliders, on its centreline) is the reference point for reversing tasks; step 6 needs it in Python for rewards.
  - **Observation terms**: `articulation` (4: the two angles, then their rates) and `trailer_goal` (4: goal − tail in the last unit's heading frame, then sin, cos of the heading error).
  - **Recordings**: ground-vehicle state messages carry `joints` (`Wheeled::joints`: per unit, a coupling's quaternion or a hinge's or turntable's angle) when there are trailers. `Wheeled::show` takes them, so replays place every unit; the viewer interpolates them element-wise.
  - **Tests** (`sim/tests/trailers.rs`, `tests_py/test_native.py::test_trailers`): both rigs spawn and settle without events, the tail sits behind in line; in a left turn the articulation state and terms go negative and agree; a tight turn with `jackknife_deg = 15` raises `JACKKNIFE` and disables the agent; a car touching the trailer's wheels exchanges equal and opposite forces that act on the trailer's link; a recorded rig replays with identical articulation.
  - **Goldens**: re-blessed. The longer state rows and `state_fields` in `/meta` changed the hashes; with the old 24 columns hashed, the trajectories, observations, events and recorded messages match the previous commit's exactly (checked in a worktree of it).
- **Step 5** (viewer):
  - **Visuals** (`scene::props::wheeled`): `WheeledVisual::units` holds a mesh per unit behind the towing unit, in that unit's frame. A unit with colliders gets a box over them (trailer bodies), one with only wheels a low frame between them (dollies), and one with neither a bar to each joint hanging from it (drawbars). The towing unit's body covers its own wheels only. Suspension link mounts are in the frame of the wheel's unit. `span` covers the whole rig in line, so the chase camera stands back far enough.
  - **Cabs**: trucks and tractors (tyres of radius ≥ 0.5 m) with colliders more than 1 m above the frame box get a cab over the frontmost of them, with the driver's eye in it. Before this, the tractors were 1.7 m tall and hidden behind a semitrailer. Cars and rovers are unchanged.
  - **Entities** (`viewer::vehicle_view`): each unit is a child of the agent's root (`UnitVisual`), posed from the towing unit to that unit. Wheels and links are children of their unit. Unit and wheel poses both come from the step's start, so they are consistent with each other.
  - **Reversing camera**: the C key cycles chase → orbit → first person → reversing (ground vehicles only) → free. It looks back from the last unit's tail (`WheeledVisual::rear_eye`, 0.9 × that unit's top), 20° down. Chase starts at a 0.35 rad pitch for rigs to look over the trailer.
  - **HUD**: one line per coupling or turntable: `name angle (rate)`, turning amber past half and red past 80 % of `jackknife_deg`. The wheel table lists every unit's wheels.
  - **CLI**: live mode takes `--trailer <preset or TOML>`, repeatable for a road train; drones with trailers are rejected.
  - **Replay**: needed no change beyond step 4 (the joints are interpolated and shown).
  - **Performance**: on the Iris Xe at 1280×720 "medium", a rural map with the semitrailer rig runs at 140–160 fps and the farm rig at 155–170 fps.
  - **Tests**:
    - `scene`: rig visuals cover every unit, end at the tail, and span the rig.
    - viewer `vehicle_view`: the farm rig's wheels, composed through their unit entities, land where the simulation has them.
    - viewer `replay`: a recorded semitrailer rig replays with the same articulation and trailer pose.
    - viewer `main`: the `--trailer` flag works.
- **Step 6** (`TrailerReverse-v0`, `bay` goals, sensors on trailers, level farm yards):
  - **Bay goals** (`sim::bay`, goal kind `bay`, ground vehicles only; maps without farm yards are rejected when compiled):
    - `bay::yards(world)` finds the yards (the `Yard` nodes) with the generator's heading: toward the eighth point of the road that leaves the yard.
    - The bay lies `depth` (16 m) behind the yard's centre, within ±`offset` (4 m) across it. The goal is the last unit's tail there (`WheeledDef::tail`), heading toward the exit.
    - The spawn puts the rig in line with its tail `distance` m (the task uses 12–24 m) ahead of the bay, within ±`lateral` (2 m) of the bay's line, with the heading within ±`yaw_deg` (10°) of the yard's. Each agent prefers a yard not used yet.
    - `GoalSpec::bay` is written to `/meta` only when changed, so existing goldens stay.
  - **Sensors on trailers**: `SensorSpec::unit` mounts a rangefinder or LiDAR on a unit behind the towing unit, in that unit's frame. It samples with the unit's pose and velocity at the step's start. Inertial and navigation sensors (IMU, GPS, baro, mag) must stay on unit 0. Tests: a LiDAR on the semitrailer's tail scans backward from there; `unit` 2 and an IMU on unit 1 are rejected.
  - **TrailerReverse-v0** (`tasks/trailer_reverse.py`):
    - **Setup**: `truck_6x4` + `semitrailer_3axle` in `vk` mode, up to 3 m/s either way and curvature 0.09 1/m. `bay` goals on a pool of 16 rural maps. 20 Hz policy, 1 kHz physics, 60 s episodes.
    - **Observation** (31 values): `trailer_goal` (×0.1), `articulation`, speed, steering, last action, and a LiDAR on the trailer's tail (one ring 1° down, 19 beams over the 180° behind, 30 m, log ranges).
    - **Reward**: progress of the tail toward the bay; −0.05·|ψ|·min(1, 5/d) for the trailer's heading error; −0.05·φ² for the articulation; smoothness 0.02‖Δa‖²; +20 on success; −20 on a crash, a jackknife (70°), a rollover, or the tail more than 35 m from the bay.
    - **Success**: the tail within 1 m of the bay, the trailer within 5° of the yard's heading, and below 0.5 m/s.
    - **Scripted driver** (`TrailerReverse.scripted`): the tail aims at the point 10 m behind its projection on the bay's line. That gives the trailer a wanted heading, which is turned into an articulation target φ* = asin(L·0.15·e), capped at 0.25 rad. The tractor's curvature holds it (κ = −sin φ/L − (φ − φ*), L = 7.6 m), and the speed tapers to a stop on the bay. It parks 90 % of the rigs (64 episodes on unseen seeds). The rest stop on the bay 5–10° off.
  - **Level yards** (`RURAL_VERSION` 4; rural golden hashes re-blessed). The yards were not level: 0.4–1.4 m of relief across the rectangle.
    - **Cause**: the road into a yard starts at its centre and follows its own graded profile from there. Its surface and shoulders are blended in after the pad, so they cut a ramp through the yard.
    - **Effect**: a rig spawned on the ramp lifted a drive wheel. With the open differentials the engine speed follows the free wheel's spin, so the engine ran at its governor with little torque and the rig stalled at full lock.
    - **Fix**: roads now stay level with the yard while on its pad and climb only beyond it; `reach_nodes` budgets the grade over that length. The yards are now level to 4 cm (the road crown).
    - **Test driver**: the road-follow test driver now also slows while turning onto a lane, as the new terrain made it run wide at one hairpin junction. The off-road test accepts a crash into a cutting's bank as the way off the road.
  - **Map pools**: `CompiledScenario::episode_maps` lists the maps episodes are drawn from. With `bay` goals that is only the maps with farm yards (map 8 of the unseen pool, seed 1000, has none), and a pool with no yards at all is rejected. Otherwise all maps are listed and the draw is unchanged.
  - **Viewer**: a bay goal is drawn as a 1 m sphere (the success radius) with an arrow along the bay's heading. The goal sphere sized to the vehicle was 17 m across for the rig.
  - **Training** (`ppo_continuous.py --hidden 256 --bound-coef 0.01`, 5M steps, 256 envs, 23 min at 3.7k SPS; basic training only, to check the pipeline):
    - Success on the training maps rose to 35 % and was still rising.
    - Deterministic evaluation: 39 % over 64 episodes (training pool). In 10 recorded episodes on unseen maps (`map_seed` 1000), 1 parked; 4 ended `STUCK` or stalled near the bay, the rest ran out of time about 16 m out.
    - A real policy needs far more training; the user trains the agents.
  - **Export and viewer**: `export_policy.py` wrote `policy.json`. `autonomousim-viewer policy <json> --map-seed 0` reverses the rig at 140–170 fps. `eval_record.py` recordings re-simulate bit for bit and replay in the viewer at about 150 fps.
  - **Tests**: Rust `trailers.rs` (bay placement, errors, trailer sensors, bay episodes skipping yard-less maps); Python: the scripted driver parks at least 3 of 4 rigs and the parked ones meet the success conditions; driving away fails; spaces and `check_env`.

### M4c: Tracked vehicles and soft soil
Planned 2026-09-26. The user asked to go on; the decisions below were taken at the start and are open to change. The design notes below ("Tracked vehicles", added 2026-09-24) still hold for the physics. **This section replaces their structure**: tracked vehicles are built into the wheeled ground model, not added as a separate `vehicles::tracked` family.

#### Decisions
- **Tracks as a running gear of `WheeledDef`**, not a new vehicle family. A tracked vehicle is a hull (unit 0) with road wheels on trailing-arm suspension, one "axle" per road-wheel pair. A `[track]` table replaces the tyres: every road wheel carries a **track patch** instead of a tyre, and all road wheels on a side are tied to the band by a locked coupling. Everything built for wheeled vehicles then works unchanged: suspension joints and `KcTravel` arm tables, the powertrain, brakes and couplings, chassis contacts, statics, the controller's side drives, simulation, observations, recordings and the viewer.
- **Band**: the band speed on each side is the common spin of its road wheels (times their radius). The band's, sprocket's and idler's inertia is lumped onto them. The sprocket drives the band through the powertrain's side output. There is no extra DoF.
- **Patch force** (per road wheel, an aux state per patch like the tyre's carcass deflection):
  - normal load from the road wheel's suspension, spread over the patch (the road-wheel pitch × track width);
  - shear displacement `j` integrated from the patch's slip velocity (longitudinal from band speed against ground speed, lateral from side-slip);
  - shear stress by Janosi–Hanamoto, `τ = τ_max·(1 − e^(−|j|/K))` along `j`, with `τ_max = c + p·tan φ` on soil and `μ·p` on rigid ground;
  - skid steering's turning resistance then comes from the patches' lateral shear (no separate moment term).
- **Soft soil** (Bekker–Wong): materials gain optional soil parameters `(k_c, k_φ, n, c, φ, K)`. Plowed soil, mud, crop, meadow, sand and snow get values from Wong's tables; paved and rock surfaces stay rigid.
  - Each patch sinks by `z = (p / (k_c/b + k_φ))^(1/n)`, which lowers its effective surface.
  - Each patch pays the compaction resistance `R_c = b·(k_c/b + k_φ)·z^(n+1)/(n+1)`. Bulldozing resistance applies at the front road wheel above a sinkage threshold.
  - Tyres on soft soil are deferred; they keep their rigid-ground model.
  - The soil parameters are left out of the map hash (materials hash as before), so existing golden hashes stay.
- **Steering and drivelines**:
  - `clutch_brake` (combustion): steering brakes the inner sprocket and cuts its drive.
  - `controlled_differential` (combustion, like Chrono's `SimpleTrackDriveline`): a steering differential biases torque between the sides.
  - The electric side drives of `rover_skid` also drive tracks.
  - Action modes: `raw` (throttle, brake and steering as the side difference), `vw` (speed and yaw rate; the controller's side-drive loops) and `per_side`.
- **Presets**:
  - `tracked_apc`: the M113 from Chrono's `data/vehicle/M113` (BSD-3), extracted by `tools/gen_chrono_tracked_fixtures.py`: hull, 5 road wheels per side on torsion arms, sprocket, idler, track, engine and transmission maps.
  - `rover_tracked`: a rubber-tracked UGV of about 60 kg with electric side motors.
- **Oracle**: Chrono's M113 with single-pin shoes on rigid terrain. The track models differ, so tolerances are loose (static loads 5 %, acceleration and turns within 15 %). Soft soil is checked analytically, because Chrono's SCM would need its own fixtures and has different physics.
- **Demo**, `TrackedCrossCountry-v0`: the APC crosses a rural map's fields (plowed soil, mud, crops) and ditches between waypoints off the roads. The rural generator gains drainage ditches along some parcel edges.

#### Implementation order
| # | Step | Done when |
|---|---|---|
| 1 ✅ | Track running gear: `[track]` table, trailing-arm road wheels, track patches with shear state (rigid ground), side-locked road wheels, sprocket and idler colliders; a test vehicle | It rests at the static solution (road-wheel loads); drives straight with slip ≈ 0 at constant speed; holds on a slope below `atan μ` and slides above it; skid-steers in a circle |
| 2 ✅ | Tracked drivelines (braked differential, electric sides; see as built), controller and action modes for tracks, presets `tracked_apc` (Chrono extraction) and `rover_tracked` | Presets load, settle and drive; `vw` holds speed and yaw rate; static loads against Chrono's |
| 3 ✅ | Validation: Chrono M113 fixtures (static loads, straight acceleration, steady turn at a fixed sprocket speed ratio, braked hold on a 30 % slope) and analytic rigid-ground checks (gradeability, Wong's skid-steer turning) | Tolerances above met or explained |
| 4 ✅ | Soft soil: material soil parameters, sinkage, compaction and bulldozing resistance, soil shear; rural materials get soft values; `sinkage` state and observation term | Drawbar pull against slip matches the Janosi–Hanamoto integral (5 %); sinkage matches Bekker's law; motion resistance matches the compaction integral; existing goldens unchanged |
| 5 ✅ | Simulation and viewer: tracked agents in scenarios, drive grids with soil cost, track visuals (band around sprocket, road wheels and idler), HUD per side (band speed, slip, sinkage), rural ditches | An APC drives by keyboard over a rural map at ≥ 60 fps on the Iris Xe; recordings replay; ditch invariants hold |
| 6 ✅ | `TrackedCrossCountry-v0`: task, scripted driver, short training, export, viewer, replay | The task trains end to end; the exported policy drives in the viewer; a recorded episode replays |

#### As built
- **Step 1 (track running gear)**:
  - **Patches** (`vehicles::ground::tire::track`): a `TireModel::Track(TrackPatch)` next to the tyre models, so a road wheel steps like a tyre.
    - Parameters: radius to the track's ground side, width, length (the road-wheel pitch), pad stiffness and damping, shear modulus `K`, `μ` on the reference surface, internal rolling resistance (default 0.03), optional design load.
    - Each patch has 8 shear cells (`TireState::shear`). The band carries the shoes through them, `dm/dt = −V_s − (|V_b|/h)(m − m_in)`, integrated implicitly in flow order. The inflow comes from the neighbouring patch's end cell (`TireState::inflow`, set by `Wheeled` from `WheeledDef::track_neighbours`); at the track's leading end it is a mirror cell, so fresh shoes arrive unsheared. **Shear therefore accumulates along the whole track** (Wong's theory), not per road wheel as first planned.
    - In steady slip, the cells sit at `j = i·x` and a side sums to Janosi–Hanamoto's integral over the contact length (within 1 % with 8 cells per patch; a test drives a 4-patch chain).
    - The lateral slip per cell includes the yaw rate's share at the cell, so the skid-steer turning moment comes out of the cells.
    - Low-speed damping inside the shear uses a **ratio of 0.7, not the tyres' 0.25**: close to the friction limit the saturating law leaves little damping, and at 0.25 a rover parked at 38° rocked for seconds.
    - Shear is capped at `10 K`. The internal resistance is a moment on the road wheel, independent of the surface.
  - **Definition**:
    - `[track]` has a `patch` table and optional `sprocket` and `idler` (left position, radius). The towing unit's axles without a `tire` (now optional) get the patch, and a patch without a design load shares the vehicle's weight.
    - Sprockets and idlers are `Skid` sphere colliders on both sides.
    - `TireSpec::Track` also allows a patch per axle.
    - Road wheels must be on the towing unit, unsteered and single.
  - **Trailing arms**: `suspension.trailing_arm = { length, angle }` generates the KC table of the arc (`z = s`, `x = L(cos θ₀ − cos θ)`; a negative length is a leading arm).
  - **Test vehicle `rover_tracked`** (56 kg):
    - 4 road wheels per side on trailing arms, 0.2 m pitch, rubber track `μ` 0.9, `K` 1 cm.
    - Electric side motors with locked couplings over the side's road wheels (the band); the road wheels' spin inertia carries the band.
  - **Tests** (`vehicles/tests/tracks.rs`):
    - rests at the static solution (loads 0.5 %);
    - runs straight on the level with slip < 0.1 %;
    - climbs 15° at constant speed with traction equal to the grade pull (1 %), at the slip the patch integral predicts under the actual road-wheel loads (5 %), shear rising front to rear;
    - holds at 38° without creep and slides at 50° at `g(sin θ − μ cos θ)` (5 %);
    - skid-steers a steady circle wider than the kinematic radius (the outer track drives, the inner brakes) and turns on the spot;
    - steady chain shear against the integral.
- **Step 2 (drivelines, controller, `tracked_apc`)**:
  - **Band**: `Wheeled` ties each side's neighbouring road wheels with locked couplings (the band), so any drive or brake on one road wheel reaches the side. `rover_tracked`'s motors lost their own `coupling = "locked"`.
  - **Steering by braking**: for a vehicle with a `[track]`, `DriveInput::steering > 0` adds service brake on the left (inner) track and `< 0` on the right, as Chrono's tracked drivelines do (`CombineDriverInputs`). With the combustion powertrain's per-axle open differentials, which the bands turn into one open differential between the sides, this is **Chrono's braked differential steering (BDS)**. It replaces the planned `clutch_brake` and `controlled_differential`: Chrono's M113 uses BDS. Electric side drives steer through `yaw` as before.
  - **Controller**:
    - Tracked vehicles with an engine get the speed loop plus a PI on the yaw-rate error that sets the steering (`brake_steer_gain` 1 s/rad, integral `yaw_integral` times it), its sign flipped in reverse.
    - While the speed builds up, the yaw-rate target follows the commanded path curvature (at most the pivot's `2/B`), and the steering's authority grows with the speed up to half the target. Without this, a turn asked from standstill braked a track at once, and the turn's resistance stalled the vehicle on its brakes for good.
    - A turn on the spot becomes a pivot turn about the inner track, its centre at `|ω|B/2`.
    - `vw` is allowed for tracked vehicles. `yaw_rate` defaults to 0.8·speed/track as for side drives.
    - Per-wheel traction braking is off for brake-steered tracks.
  - **Statics**: a wheel may hang clear of the ground at rest if its spring has a given preload, as the end road wheels do under the band.
  - **`tracked_apc`**, from Chrono's M113 via `tools/gen_chrono_tracked_fixtures.py` (NSC contact: with SMC the braked vehicle crept at 0.5 m/s on chattering shoes). The script records:
    - design (masses and inertias, the lumped chassis);
    - the settled pose, road-wheel positions and arm angles, averaged over the last 2 s of a 5 s braked settle;
    - the **vertical ground reaction per road wheel**, by binning contact forces on the ground plane (`ReportAllContacts`);
    - the conical gear ratio.
  - **Chrono's M113 findings**:
    - **Suspension forces are not ground loads.** The track's tension loads the springs to about twice the weight in total. The first and last road wheels ride 3–5 cm above the ground run and carry no ground load at rest. The preset keeps this: those two hang (given preload −608 N, their unsprung weight), and the middle three carry the vehicle.
    - **Conical gear ratio**: 0.504 ± 0.04 averaged over 3 s of driving (single samples scatter ±0.1: the iterative solver leaves the shaft constraints loose), so the JSON's 0.5.
    - **Weak launch**: Chrono's full-throttle map falls to 0 N·m at −100 rpm, and the clutchless SimpleMap driveline turns the engine with the tracks, so the M113 needs 2 s to reach 0.3 m/s. The preset holds 610 N·m down to standstill instead (standing in for a launch clutch). Step 3 must give Chrono the same map by JSON.
  - **Preset values**:
    - Chassis: hull + sprockets + idlers + their carriers + shoes + the arms' mass not carried with the wheels (the arm's `I/L²` about the pivot, 26.45 kg, is carrier mass).
    - Road-wheel spin inertia 33.3 = wheel + band (shoes' mass × r²/5) + rollers.
    - Wheel rates and dampers are the torsion rates over `(L cos θ)²` at the settled arm angles.
    - Patch: radius 0.374, `k_z` 2e6, 0.38 × 0.667 m, `K` 1 cm, `μ` 0.8.
    - Brakes per road wheel: 10 kN·m × (0.364/0.245)/5.
    - `final_drive` 0.5 × 0.245/0.364.
    - Chrono's gear ratios and shift points; our own zero-throttle map (Chrono's repeats the full-throttle one).
  - **Turning is weak, and the reference model's own**. Braking the inner track takes drive away rather than passing it to the outer track, and the M113's forward gears (no torque converter) leave about 21 kN (1st) and 12 kN (2nd) of tractive force. The patches' turning resistance (an effective lateral coefficient of about 0.16 at R ≈ 110 m before step 3's Wong–Chiang stress direction, which lowers it for a sliding braked track) then allows only gentle forward turns: 4 m/s at 0.05 rad/s holds, and 6 m/s at 0.1 rad/s does not. The reverse gear is lower, so −2 m/s at −0.1 rad/s holds. Pivot turns are out of reach. A torque converter, or a regenerative steering differential, would lift this; both are left open. (Revised after step 5: `tracked_apc` now has both; see "Torque converter and regenerative steering".)
  - **Tests**:
    - `tracks.rs`: the APC's mass and static height (1 cm), pitch (0.005 rad) and per-road-wheel ground-load shares (3 points of the total) against Chrono; parked, it settles there. Actual: height 0.602 vs 0.604 m, pitch 0.0142 vs 0.0134 rad, shares within 1.2 points.
    - `control/tests/ground_loop.rs`: `rover_tracked` tracks `vw` (straight, turning, on the spot, reverse) like the other robots. The APC tracks (8, 0), (4, 0.05) and (−2, −0.1) to within 5 %, holds on its brakes, and survives an impossible spot-turn command, rolling instead of stalling, then drives off straight.
- **Step 3 (validation)**:
  - **Chrono runs** (`tools/gen_chrono_tracked_fixtures.py`, `fixtures/chrono/tracked_m113.json`): the M113 with **our engine map and gearbox by JSON** (`ReadEngineJSON`/`ReadTransmissionJSON`, `ChPowertrainAssembly`), on a plane with gravity tilted for grades. Runs: accel (4 s full throttle), climb (15 %, 6 s), hold (30 %, 4 s braked), turn and turn_hard (steering 0.3 and 0.6 at 1.2 m/s under a PI throttle, started at 2 m/s by `SetInitFwdVel` to skip the slow launch). The static and driveline records stay from step 2.
  - **The M113 is solver-bound**: at 150 NSC iterations, running resistance rises from 0.045·W at launch to 0.14·W at 2 m/s (0.17·W at 3 m/s), a braked vehicle creeps down 30 % at 5 cm/s although its brakes hold 2.5× the demand, it stalls on 15 %, and its turning response halves from 150 to 600 iterations. The fixtures therefore use 600 iterations (0.115·W at 2 m/s, creep 2.2 cm/s, crawls up 15 %), at about 4 min per simulated second, which is why the runs are short. Chrono is an offline reference only; our APC runs at 150–300× real time.
  - **Model changes**:
    - **Wong–Chiang stress direction** in the patches: a cell's stress acts against its sliding velocity once it slides (shear above 0.5–1.5 `K` and sliding faster than 1–5 cm/s, both blended) and along its shear while it sticks. With the stress along the shear alone, a pivot turn's moment came out at 60 % of Wong–Chiang's; gating on speed alone made the stopped APC rock, gating on saturation alone let slopes creep.
    - `tracked_apc` `rolling_resistance` 0.045, Chrono's M113 at launch.
    - Controller: the yaw integral holds while the steering authority is below 1 (it wound up while the vehicle rolled out of a stalled turn).
  - **Tests** (`vehicles/tests/tracks.rs`), with the results:
    - **Launch**: acceleration over the first 0.5 s within 15 % of Chrono's, and never slower after. After 4 s ours is at 4.5 m/s, Chrono's at 2.8: its speed-growing resistance is the solver's, not modelled (explained, not a tolerance miss).
    - **15 % climb**: both crawl at the traction limit without rolling back, ours at 0.07, Chrono's at 0.15 m/s (mean of 4–6 s).
    - **30 % braked hold**: ours creeps < 5 mm; Chrono's creeps 2 cm/s, its sprockets turning with it (the solver's leak through the brakes).
    - **Brake steering**: at steering 0.3 the yaw rates over 2.5–5 s are within a factor 2 (ours 0.008, Chrono's 0.010 rad/s). At 0.6 both nearly stall (speed < 0.15 m/s, yaw rate < 0.05 rad/s). Chrono's turning moves by a factor 2 with the solver's iterations, so it cannot be a 15 % reference; turning is checked analytically instead.
    - **Gradeability (analytic)**: `sin θ + f cos θ = F/W` with first-gear tractive force `T/(g₁·i_f·r)` and the rotating masses in `m_eff`. The accelerations at 0, 0.5 and 0.9 of the limiting grade are within 5 %; above the limit the APC does not climb.
    - **Pivot turn (analytic, Wong and Chiang)**: `rover_tracked` turning on the spot. Given the measured yaw rate and slips, the outer and inner thrusts match the numerical Wong–Chiang integral over the contact patches within 3 %, and the turning moment within 5 % (3.4 %), below the `μWl/4` of fully sliding tracks.
- **Step 4 (soft soil)**:
  - **Soil parameters** (`core::material::Soil`): `n`, `k_c`, `k_φ`, `c`, `φ`, `K` and a bulk density for bulldozing. They are a built-in property of the material's name (`Soil::of_material`), `#[serde(skip)]` in `Material` and restored by name when a table is read. So map files, their content hashes and all golden map hashes are unchanged. Values from Wong's Table 2.3:
    - sand: dry sand (LLL);
    - mud: clayey soil (Thailand), φ 13°;
    - snow: U.S. snow;
    - meadow: Grenville loam;
    - crop: Rubicon sandy loam;
    - plowed: upland sandy loam.
    `K` (1–4 cm) and the densities are estimates. Grass, forest floor, dirt, gravel and paved surfaces stay rigid. Wild maps' sand, mud and snow are soft too, for tracks. Tyres keep their rigid-ground model.
  - **Patch on soil** (`track.rs`):
    - **Sinkage**: the pads and the soil act in series, `k_v(ρ − z) = b·L·(k_c/b + k_φ)·z^n`, solved by safeguarded Newton. Only on soil; rigid ground takes the old code path bit for bit.
    - **Plastic rut** (`TireState::sinkage`): the soil holds, as rigid, up to the pressure that made the rut, and yields along Bekker's law beyond it. The ground's motion carries the rut away (rate `|v_x|/L`) and replaces it with the rut the patch ahead left (`TireState::rut_in`, set by `Wheeled` like the shear inflow). The rear patches therefore ride in the front ones' rut.
    - **Resistance**: compaction `b(k_c/b + k_φ)(z^(n+1) − z_in^(n+1))/(n+1)` and Rankine's passive bulldozing `b(2cz√K_p + ½ρgz²K_p)`, each counted from the rut depth `z_in` ahead, so a track pays them once, to its deepest rut. There is **no sinkage threshold** for bulldozing (it is small at small sinkage). The resistance acts on the road wheel's centre, not at the contact point: applied at the contact it braked the band, and the patch's own shear cancelled it without any drive torque.
    - **Shear**: Mohr–Coulomb per cell, `c·A_cell + F_z·tan φ`, with the soil's `K`.
  - **Sinkage state and term**: `Wheeled::sinkage()` is the mean over the loaded patches. It feeds state column `sinkage` (`STATE_DIM` 30) and obs term `sinkage`. The goldens were re-blessed after an A/B check against the last commit (states without the new column, obs, events and world hashes identical); `hover.mcap` changed with the column.
  - **Tests** (`vehicles/tests/soil.rs`):
    - **Patch chain** (5 APC patches, plowed soil) at slips from −5 % to 30 %: sinkage is Bekker's to 1e-6, and pull equals Janosi–Hanamoto's integral less compaction and bulldozing to 1 %.
    - **Motion resistance**: the APC at 2 m/s on plowed soil and on sand; drive torque less internal resistance equals the compaction and bulldozing of the deepest Bekker rut within 1 % (1704 vs 1694 N, 1910 vs 1910 N). Each patch sinks by Bekker's law under its load, or rides in a deeper rut ahead (5 %).
    - **Parked**: gravity is ramped over 2 s. The patches sink by Bekker's law up to 8 % deeper (the rut keeps the peak load while the end road wheels take up load), and the hull sits lower by about as much. Dropped from the rigid-ground pose instead, it overshoots by up to 30 %.
    - **Drawbar pull**: the rover on plowed soil at 10, 25 and 40 % of its weight. Pull matches the integral under the actual loads within 5 %, and slip grows with the load.
    - `sim/tests/ground.rs`: the state column and the obs term read the APC's sinkage on sand, and 0 on rigid ground.
  - **Limitation**: sprocket and idler colliders (and the hull's) meet the rigid terrain surface. A vehicle sunk deeper than their clearance rests on them: the rover in snow sinks 5 cm, and its sprocket and idler carry about a quarter of its weight.
- **Step 5 (simulation and viewer)**:
  - **Rural ditches** (`RURAL_VERSION` 5, rural goldens re-blessed): trapezoidal drainage ditches along a share of field edges (`DitchesConfig { share 0.3, depth 0.8, width 6, bottom 1.5, gap 1.5 }`, banks about 20°), mud at the bottom.
    - They ramp out over `DITCH_RAMP` (3 m) near roads, farm yards and water. Distance to water comes from a chamfer transform, so a ditch meets a pond without a wall.
    - `RuralStats` counts them; `mapgen` prints "ditches N (x km)". The ditch lines are kept (not serialized) for tests.
    - **Invariants** (`procgen/tests/rural.rs`), against the same seed without ditches: the centre cut is the depth (0.05 m, bilinear sampling), the cut is 0 at the rim + 1.5 m, the bottom is mud, and no hedge or fence lies in a ditch.
  - **Soil cost in drive grids**: `drivable.resistance_cost` (above). `WheeledDef::motion_resistance(material, g)` is the tyres' rolling resistance, or for tracks the patch's internal resistance plus Bekker compaction and bulldozing under evenly loaded road wheels (APC 0.045–0.052, 0.099 in snow; sedan 0.013 on asphalt to 0.14 on plowed soil). A* edges cost the mean of their cells. Grids are shared only between equal vehicles when the cost is on. A test: a plowed field in meadow, which the car drives around and the APC crosses.
  - **Band-driven rollers** (revises step 4's limitation): sprockets and idlers are rollers (`SphereCollider::spin`, the surface's angular velocity about the link's y axis), turning with the band (mean road-wheel spin × band radius / roller radius). Friction acts on the band's slip against the ground, and its moment about the roller axle moves from the hull to the side's road wheels. Before, a sunk APC's sprocket and idler slid with μ 1 and anchored it in plowed soil. A test lowers the idler 0.25 m onto the ground: it carries > 10 % of the weight, and the speed after 2 s stays within 10 % of the unmodified APC's (the old colliders: 1.70 against 2.27 m/s).
  - **Track visuals** (`scene::props::track_band`): a band around the convex hull of sprocket, road wheels and idler, with grousers at a quarter patch length that move with the band's travel (mean road-wheel spin angle × radius). Road wheels are drawn inside it. Viewer system `sync_tracks`.
  - **HUD**: a table per side for tracked vehicles: band speed, slip (mean κ of the loaded patches), sinkage and load.
  - **Recordings**: `RecordedWheel.sinkage` (omitted when 0, so older recordings and `hover.mcap` are unchanged); replay interpolates it. A test replays an APC on sand, sunk, its bands running.
  - **`assets/scenarios/farm_apc.toml`**: the APC in `vw` mode on 4 rural training maps, random goals, soil-weighted paths, a `sinkage` observation term. `sim/tests/ground.rs` drives it on 12 seeds for 8 s: it moves > 20 m or stops on an obstacle, foliage, terrain or the boundary, is never stuck, and sinks > 5 mm on at least 8 maps.
  - **Viewer**: the APC by keyboard over a rural map at 96–103 fps (1080p, medium, `--demo`), real time ×1.00; across crop fields at 3.9 m/s, 25–35 mm sinkage.
  - **Limitation, launch**: without a torque converter (as Chrono's M113), 1st gear gives about 20.7 kN against 110 kN weight, so from standstill the APC starts uphill only on about 8–10°. Ditch banks are crossed with momentum; `farm_apc` spawns at most 6° steep (at 15°, 3 of 24 seeds stalled at spawn). (Revised: with the torque converter below, `farm_apc` spawns up to 10° steep.)
- **Torque converter and regenerative steering** (after step 5, the open offers of steps 2 and 5):
  - **Torque converter** (`CombustionDef::torque_converter`, Chrono's `ChShaftsTorqueConverter`): the engine speed becomes a state with `engine.inertia`. The APC's is 0.5 kg·m², the transmission's motorshaft: Chrono's powertrain assembly imposes that shaft's speed on the engine's own motorshaft (1.1 in the data), so only the 0.5 turn with the engine. Measured in Chrono, the engine runs up from 0 to 45 rad/s in 0.05 s at full throttle; with 1.6 our launch lagged Chrono's by 0.3 s. The pump loads the engine with `(ω_e/K(R))²` at speed ratio `R = ω_t/ω_e`, the turbine passes `TR(R)` times that to the gearbox. On overrun (`ω_t > ω_e`) the roles swap and `TR = 1`. The gearbox shifts on turbine speed; during a shift the converter carries no torque. Without a converter the powertrain is bit-identical to before (goldens unchanged; the zero engine inertia is not serialized, so recordings' `/meta` is unchanged too).
  - **`tracked_apc` powertrain from Chrono's M113 shafts data** (`M113_EngineShafts.json`, `M113_AutomaticTransmissionShafts.json`): full-throttle table = engine map + losses map, zero-throttle table = losses map (exact at both ends and linear between); capacity factor `[[0,7],[.25,7],[.5,7],[.75,8],[.9,9],[1,18]]`, torque ratio `[[0,2],[.25,1.8],[.5,1.5],[.75,1.15],[.9,1],[1,1]]`. Downshift at 750 rpm instead of Chrono's 1000 (the 1→2 upshift at 1500 rpm leaves 843 rpm in second, which hunted). The Chrono fixture generator gives Chrono the same downshift by JSON.
  - **Regenerative controlled differential** (`[track] steering = { type = "regenerative", ratio, torque }`; the default stays `brake`): two geared couplings (`Coupling::geared`) between the sides' road wheels. Steering `s > 0` engages the left one with capacity `s·torque`, holding `ω_left = ρ·ω_right`: its torque `t` on the inner track comes with `−ρ·t` on the outer, so the inner track's drive passes to the outer one instead of being burnt in its brakes, and the coupling only dissipates. The tightest turn has radius `(B/2)(1 + ρ)/(1 − ρ)`. The APC: `ρ = 0.6`, 15 kN·m (radius 4.3 m). Open loop at full throttle and full steering it holds 4.5 m/s at 0.85 rad/s (R ≈ 5.3 m), where brake steering stalls. Chrono's M113 has BDS only, so the regenerative steering is checked against the geared kinematics, not Chrono.
  - **Controller**: `track_steer` holds the tightest radius (`B/2` for brake steering, the geared radius otherwise). A spot turn drives it at `v = |ω|·R`, the curvature clamp is `±1/R`. With a converter, the throttle inversion divides by `TR(R)` at the current speed ratio and leaves engine braking to the brakes. A launch holds the brakes (`brake = 1` while `|v| <` hold speed, the throttle is on and the engine is below 0.9 of the converter's stall speed for the throttle, found by bisection), so the vehicle does not roll back on a slope while the engine spins up.
  - **Patch stick–slip fix** (`track.rs`): a braked APC walked at 3 cm/s after a reverse turn. A cell's shear flipped sign every step (a Nyquist-rate chatter between sticking and sliding). A cell's sliding weight is now also multiplied by `((loading − (−0.9))/(−0.7 − (−0.9))).clamp(0, 1)`, with `loading` the cosine between its shear and its reversed sliding velocity: a cell whose shear points along its sliding (being unloaded) no longer slides. A gate at 0 (full weight once loaded) cut the rover's steady skid circle from 0.535 to 0.477 rad/s; the narrow gate leaves the tracks, soil and pivot tests unchanged (pivot moment 79.15 vs 81.92).
  - **Results**:
    - Gradeability from standstill: the stalled converter's limit is `sin θ + f cos θ = 0.299` (about 16.6°; before 0.19). `apc_climbs_up_to_its_gradeability` now checks steady crawls at 0.5, 0.8 and 0.95 of the limit: the tractive force at the measured band speed (through the converter's balance) matches the grade within 1 %. At 1.05× the limit it does not climb (< 0.1 m/s).
    - `vw` control (`ground_loop.rs`): (8, 0), (4, 0.05), (5, 0.3), (3, 0.5) and (−2, −0.3) within 0.3 %; it holds after stopping. A spot turn at 0.4 rad/s drives the 4.3 m circle at 1.73 m/s and 0.35 rad/s (asserted: speed within 5 %, yaw rate within 15 %). Before: (5, 0.3) was out of reach, and forward turns were limited to about 0.05 rad/s.
    - **Chrono comparisons**, re-run with the same shafts powertrain in Chrono (`fixtures/chrono/tracked_m113.json`; the static, design and driveline records are unchanged; new run `climb_steep`, 25 %):
      - Launch: 1.52 against Chrono's 1.39 m/s² over the first 0.5 s (15 %), never slower after. After 4 s, 5.2 against 3.4 m/s (Chrono's speed-growing resistance, as before).
      - Climbs from standstill on 15 % and 25 %: both roll back less than 0.1 m while the engine runs up (ours 2.4 cm on 25 %, Chrono's 8.6 cm), track Chrono's speed within 0.15 m/s for the first second and are never slower after. On 25 % Chrono settles at 0.37 m/s, ours reaches 1 m/s after 6 s. Before, both only crawled up 15 %.
      - 30 % braked hold: unchanged (ours < 5 mm; Chrono's creeps about 2 cm/s).
      - Brake steering (the test switches the preset to `TrackSteering::Brake`; Chrono's M113 has BDS only): at steering 0.3, yaw rate over 2.5–5 s 0.014 against 0.013 rad/s (within 30 %); at 0.6 both slow below 0.25 m/s and turn at 0.06 and 0.045 rad/s (within a factor 1.5), then pick up speed through the converter. Before, both stalled into a slow pivot.
    - `farm_apc` spawns on up to 10° (was 6°). At 12°, one of 12 seeds crawled up a steepening bank (12 m in 8 s); the test's distance bound is now 10 m in 8 s.
  - **Chrono is offline only**: the fixtures are generated once by `tools/gen_chrono_tracked_fixtures.py` (micromamba env `chrono`) and committed; tests and training never run Chrono.
  - **Auto-export**: see "As built after M1: policy playback".
- **Step 6** (`TrackedCrossCountry-v0`, `path` goals):
  - **`path` goals** (`GoalSpec::path`, with `random` goals of ground vehicles; validated): each episode plans the cheapest drivable path from the spawn through the goals, one leg per goal (`DriveGrid::legs`: A* on the group's drive grid, then a 7-point moving average that turns 45° steps into curves and cuts a right-angle corner by 1.2 cells). The legs are lifted onto the terrain as polylines (`Agent::legs`). The current goal's leg is the agent's `route` (switched when a goal is reached; the last stays), so the `road` state columns and the `route` term follow it. The vehicle starts facing the path point 8 m ahead, turned by its sampled `spawn.yaw_deg`. `/episode` records the whole path as the agent's route, so the viewer overlay and replays draw it. `set_goals` drops the legs.
  - **`off_road` goals** (`GoalSpec::off_road`): random ground goals prefer points off the roads (the candidate score drops from 1 to 0.5 on a road; unreachable ones stay at −3).
  - Both are serialized only when set, so goldens are unchanged.
  - **TrackedCrossCountry-v0** (`tasks/tracked_cross_country.py`):
    - **Setup**: `tracked_apc` in `vw` mode (6 m/s forward, 3 m/s reverse, 1 rad/s), 20 Hz policy, 1 kHz physics, a pool of rural maps. Spawns 40 m from the edge on ≤ 10° slopes, facing along the path within ±30°. Three `off_road` + `path` goals 40–100 m apart, radius 4 m. Drive grid: slopes ≤ 30°, 2 m margin, `resistance_cost` 30. Episodes last 120 s.
    - **Observation** (95): `route` (1/20, clipped to ±3), `goal_rel_heading` (1/50), speed, body velocity and rates, pitch and roll, `sinkage` (×10), last action, LiDAR (2 rings at −10° and 0°, 36 azimuths, 30 m, log ranges).
    - **Reward**: speed along the path's tangent × Δt, 10 per goal, −0.05·min(1, (offset/5 m)²), −0.02‖Δa‖², −50 on failure. Failures are the terminal events plus `STUCK` (6 s).
    - **Events**: `crash_speed` is 5 m/s. At 2 m/s (the drones' landing threshold), sprockets meeting the far bank of a ditch at 2–4 m/s ended a third of the scripted episodes as terrain crashes. Hull contacts are still always crashes.
    - **Scripted driver** (`scripted(obs)`): pure pursuit on the `route` term, aiming 10 m ahead with a yaw rate of 1.2 × the bearing, at 4.5 m/s less 5 m/s per radian of the larger of the 10 m and 20 m bearings, at least 3 m/s. It finishes about 75 % of the episodes (24–25 of 32). Failures are physically plausible: stalling while climbing out of a ditch with the rear in mud (μ 0.35, pitch 20–23°), brushing a hedge's foliage, or sliding into water. It was not tuned further (at 1 m/s minimum it stalled on banks: 19 of 32).
    - **Tuning history**: the spawn heading and the 2 m margin came from the scripted driver's failures. With random headings, the APC (no pivot turn; 4.3 m tightest radius) turned into banks; with a 1 m margin, it brushed fences and hedges. A lower slope limit (18°) did not help: the paths detoured and ran out of time.
  - **Tests**:
    - `sim/src/drive.rs`: legs through goals (around a tree, unreachable goal, goal in the start cell); smoothing keeps ends and straight lines.
    - `sim/tests/paths.rs`: legs start at the spawn and end at each goal over drivable ground; ≤ 10 % of goals on roads; the route switches per goal; the vehicle faces its path; validation errors; recordings keep the whole path.
    - `tests_py/test_envs.py`: API conformance (`check_env`, spaces); the scripted driver finishes ≥ 4 of 8 episodes with a median offset < 1.5 m; standing still fails as `STUCK` with the penalty.
  - **Training** (`ppo_continuous.py --hidden 256 --bound-coef 0.01`, 3M steps, 256 envs, 14 min at 3.6k SPS; basic training only, to check the pipeline): the training return rose from −52 to +62, with 24 % success. Deterministic evaluation on unseen maps (`map_seed` 1000, 64 episodes): 31 % success, 1.4 of 3 goals.
  - **Export and viewer**: `policy.json` is auto-exported. `autonomousim-viewer policy <json> --map-seed 1000` drives the APC at about 134 fps. `eval_record.py` recorded 4 episodes (3 successes); they re-simulate bit for bit and replay in the viewer at about 120 fps, with the path, the goals and the LiDAR view.

## Milestone 5: Bicycles and motorcycles
Planned 2026-09-27. The user decided the following at the start: both a bicycle and a motorcycle; a rider with a leaning upper body; stabilised actions by default plus a raw mode; and a motorcycle road ride as the closing demo. The rest of this section is proposed and open to change.

### Design
- **Single-track vehicles are built into `WheeledDef`**, not added as a new vehicle family, as tracks were in M4c. Spawns, drive grids, events, observations, recording, statics, the powertrain, brakes, tyres and the viewer then carry over.
  - An axle with `track = 0` carries one wheel on the centre line.
  - About 40 places assume a left/right pair per axle (`w / 2`, `k % 2`, `num_wheels = 2·axles`). They move to a per-axle wheel count.
- **Steering head**: an optional `[steering_head]` on the front axle.
  - It replaces the prescribed knuckle with a **free** revolute joint about the tilted steer axis. Parameters: head angle `λ` from vertical, fork offset (so the trail follows from the wheel radius), and the axis position.
  - A steered body carries the fork, handlebar and front-wheel carrier: mass, COM and inertia.
  - Steer stops, an optional steering damper, and the **steer torque** as the rider's input. The existing steering chain (rate limit, Ackermann blending) is bypassed.
  - The Whipple benchmark's front frame is exactly this body.
- **Suspension**:
  - Front: a telescopic fork, a prismatic joint along the steer axis between the steered body and the wheel carrier, with spring, damper and stops.
  - Rear: a swing arm, reusing M4c's `suspension.trailing_arm` with a spring and damper.
  - Bicycles are rigid: no suspension joints.
- **Rider** (`[rider]`):
  - The lower body is lumped into the chassis. The upper body is a separate link on a revolute **lean joint** about the chassis x axis at the hip, with mass, COM and inertia.
  - A servo holds the lean at a commanded angle relative to the frame: PD with a torque limit, representing the rider's muscles. With the lean locked (or `[rider]` absent) the model reduces to the Whipple benchmark's rigid rider.
  - **Feet**: two sphere colliders beside the footpegs, at ground height when "down". They are down below `feet_speed` (default 1.5 m/s) and while stopped, and up otherwise. They let the vehicle spawn and stand upright and launch from rest. Their contacts are gear contacts, not crashes.
- **Tyres**:
  - **Toroidal contact** (`crown_radius` per tyre): the contact point moves sideways with camber as on a real tyre, `centre − (R − r_c)·ẑ_wheel − r_c·n_ground`. The vertical load uses the loaded radius along that line. It returns a contact up to about 60° of camber (the thin disc stops at "on its side"). A crown radius of 0 keeps today's thin disc, so goldens stay unchanged.
  - **Large camber**: MF 6.1/6.2 already has camber thrust and Mx. It must now hold up to about ±55° (today's MFeval fixtures stop at 5.7°).
  - **Turn slip** (MF 6.2, the `ζ` factors, when the `.tir` has turn-slip parameters): it matters for motorcycles at low speed and in tight turns.
  - Presets need motorcycle tyre data: a published MF parameter set for a 120/70 ZR17 front and a 180/55 ZR17 rear (source to confirm in step 2; as built: Evangelou's MF-MC sets), and a bicycle tyre fitted to measured cornering and camber stiffness (Dressel 2013; an estimate).
  - A `rigid_rolling` tyre option approximates the benchmark's knife-edge, no-slip wheels: very stiff, without relaxation.
- **Linear model** (`single_track::linear`): the Meijaard et al. (2007) matrices `M`, `C1`, `K0`, `K2` computed from a `WheeledDef`'s bodies and geometry. The rear frame is the chassis with the rider locked and the rear wheel; the front frame is the steered body with the front wheel.
  - Used by the validation.
  - Used by the controller: gain-scheduled LQR over speed, with steer torque and rider lean as inputs.
  - Used for eigenvalue plots in tests.
- **Rider controller and action modes** (the default mode is stabilised):
  - `vk` (**default**): speed and curvature. The controller sets a target roll from the curvature, `φ_ref = atan(v²κ/g)` plus a correction for tyre width and rider lean. An LQR on `(φ, δ, φ̇, δ̇)` gives the steer torque. An integral on the curvature error removes steady-state error. Speed is held by throttle and brakes, with front/rear brake balance. The rider's lean stays neutral in `vk`. Below `feet_speed` the feet go down and the vehicle crawls straight.
  - `vw`: speed and yaw rate, through `κ = ω/v`.
  - `raw`: drive (throttle or brake), steer torque and rider lean angle, normalised. The agent must balance the vehicle itself; the feet still deploy at standstill.
- **Events**: the frame, the rider or the handlebar touching the ground is a crash, as for the hulls of other vehicles. `ROLLOVER` (60° by default) stays as a backstop.
- **Observations**:
  - new term `lean` (roll angle and roll rate about the heading);
  - `steering` gains the steer rate for free steering heads;
  - new term `rider_lean`;
  - new term `feet` (down or up).
  - The existing terms (`speed`, `pitch_roll`, `road`, `route`, …) apply unchanged.
- **Presets**:
  - `bicycle_benchmark`: Meijaard 2007's benchmark bicycle, rigid rider and knife-edge wheels (`rigid_rolling`). Validation only.
  - `bicycle_city`: about 18 kg bicycle plus 75 kg rider with a leaning upper body. Tyres 37-622. Pedalling as a power-limited drive (about 250 W sustained, 600 W peak) with a freewheel. Rim brakes.
  - `motorcycle_sport`: about 200 kg plus 75 kg rider. Geometry, masses and inertias after a published sport-bike parameter set: Sharp, Evangelou & Limebeer (2004), Suzuki GSX-R1000; access to confirm. It has an engine map (about 130 kW), a 6-speed gearbox with shift schedule, chain drive, fork, swing arm, a steering damper and disc brakes.
- **Oracles**:
  - The **Whipple benchmark** (Meijaard, Papadopoulos, Ruina & Schwab 2007), analytic.
  - **MFeval** for large camber and turn slip.
  - Analytic steady turning.
  - Published motorcycle modes (Sharp et al. 2004).
  - Chrono::Vehicle has no motorcycle or bicycle model (to confirm), so there is no Chrono oracle this time.
- **Integrator**: PLAN noted "revisit wheel gyroscopics in M2/M5". The spinning wheels' gyroscopic coupling is what stabilises a bicycle. If semi-implicit Euler at 1 kHz moves the benchmark eigenvalues by more than 1 %, articulated models get the implicit-midpoint velocity-product treatment the free bodies already use (or the benchmark runs at a smaller `dt`).
- **Demo**, `MotorcycleRoadRural-v0`:
  - The motorcycle rides a route along the lanes of the rural roads (spawn `on_road`, `route` goals), leaning into the bends, at up to about 25 m/s on paved roads and slower on gravel.
  - Default mode `vk`. `action_mode="raw"` is the harder variant, in which the agent balances.
  - Reward: progress along the route, lane keeping, smoothness, and a penalty for falls and crashes.
  - Scripted driver: pure pursuit on the `route` term, with speed from the curvature ahead (0.5 g lateral, about 27° lean).

### Implementation order
| # | Step | Done when |
|---|---|---|
| 1 ✅ | Single-track layout: centre-line wheels (`track = 0`, per-axle wheel counts), `[steering_head]` with a free steer joint and steer torque, `[rider]` lean joint with servo, feet; statics; `bicycle_benchmark` preset | Builds and settles; static wheel loads match the COM; standing still with feet up it falls over (capsize); freewheeling without dissipation conserves energy within 1e-3 over 10 s; cars, trucks and tracked goldens unchanged |
| 2 ✅ | Tyres for two-wheelers: toroidal contact with `crown_radius`, large camber, MF 6.2 turn slip, `rigid_rolling`; motorcycle and bicycle tyre sets | Contact geometry matches the torus analytically up to 60°; forces and moments match MFeval up to ±55° camber and with turn slip (as in the M2 tyre tests); crown radius 0 leaves goldens unchanged |
| 3 ✅ | Whipple benchmark: `single_track::linear` matrices; numerical linearisation of the full model; integrator fix if needed | The matrices reproduce the published `M`, `C1`, `K0`, `K2` (1e-3); the full model's eigenvalues match the benchmark's over 0–10 m/s within 1 %; weave speed 4.292 m/s and capsize speed 6.024 m/s within 1 % |
| 4 ✅ | Presets and powertrains: `bicycle_city` (pedal drive, freewheel), `motorcycle_sport` (engine, gearbox, chain, fork, swing arm, steering damper, brakes) | Static sag and loads as specified; acceleration, top speed and braking plausible against published figures; steady turning roll angle vs lateral acceleration within 1° of the analytic value with tyre widths; weave and wobble modes present with frequencies and damping trends in Sharp et al.'s ranges |
| 5 ✅ | Rider controller and action modes: gain-scheduled LQR from the linear model, `vk`/`vw`/`raw`, rider lean servo, feet, launch from rest | Straight-line hold under a lateral impulse at 3, 10 and 25 m/s; curvature steps settle without falls; the initial countersteer has the right sign; launch from rest and stop with feet down; the bicycle stays up at walking speed |
| 6 ✅ | Simulation: upright spawns with feet down (also on cross slopes), drive grid width, events, the `lean`/`rider_lean`/`feet` terms, recorded steer, lean and feet; Python `vehicle="motorcycle_sport"` | Scenarios with two-wheelers compile, spawn, ride and record; existing goldens unchanged; terms match references |
| 7 ✅ | Viewer: two-wheeler visuals (frame, tank, fork, handlebar, toroidal tyres, rider with a leaning torso), HUD (roll, steer angle and torque, rider lean, feet, gear), keyboard riding in `vk`, replay | A motorcycle rides by keyboard over a rural map at ≥ 60 fps on the Iris Xe; recordings replay |
| 8 ✅ | `MotorcycleRoadRural-v0`: task, scripted driver, short training, export, viewer, replay | The task trains end to end; the exported policy rides in the viewer; a recorded episode replays |

**To confirm while building**:
- access to the motorcycle parameter set and tyre data;
- whether MFeval evaluates turn slip at large camber (its `useMode`);
- that Chrono::Vehicle has no single-track model;
- whether the bicycle tyre fit from Dressel's measurements is close enough.

#### As built
- **Step 1 (single-track layout)**:
  - **Wheels per axle**: an axle whose (left) wheel sits on the centreline (`position.y = 0`) has one wheel, not a pair; there is no `track` field. `WheeledDef::wheel_axle`, `wheel_side` (0 left or single, 1 right), `axle_wheels` and `axle_ranges` replace the `w / 2`, `w % 2` and `2a, 2a + 1` arithmetic everywhere (vehicles, powertrain, control, sim, scene, viewer). `MAX_WHEELS` (16) now limits wheels rather than axles. A single wheel takes no dual tyres, track or anti-roll bar. The combustion driveline splits an axle's share over its wheels, with no axle differential for a single wheel.
  - **Steering head** (`[axles.steering_head]`, on one single, otherwise unsteered wheel of the towing unit):
    - Given by head angle `λ` and fork offset (the wheel centre's distance ahead of the axis, square to it); the trail follows, `(R·sin λ − offset)/cos λ`.
    - A free revolute about the tilted axis, carrying the steered body (mass, centre and full inertia tensor, chassis frame). The fork (`KcTravel`, if sprung) hangs from it, then the wheel. Without kinematics of its own, the fork slides along the axis (`SuspensionDef::fork_table`).
    - Torques on it: the rider's, `DriveInput::steering × max_torque` (`steering` is the torque command on a head, the angle command otherwise); a damper; lock stops (`lock_stiffness`, default 500 N·m/rad, damped over 10 ms). `WheelState::steer` and `steer_torque` of its wheel report the angle and the applied torque; `Wheeled::steering_angle` returns the head angle.
  - **Chassis inertia products**: `products = [I_xy, I_xz, I_yz]` (the tensor's off-diagonal entries) on chassis, units, steered body and rider, needed for the benchmark's rear frame.
  - **Rider** (`[rider]`): an upper body on a revolute about the chassis x axis through `hip`, after the towing unit's wheels in the tree. Its servo is `stiffness·(lean·max_lean − φ) − damping·φ̇`, clamped to `max_torque`, with the new `DriveInput::lean` (positive to the right, as roll; not serialised when zero). `Wheeled::joints` (and `show`, and the recorded `joints`) append the lean.
  - **Feet** (`[feet]`): two sphere colliders (`Skid`) on the chassis. They move to `down` below `speed` (default 1.5 m/s) and back to `up` above 1.25·`speed`. Down, they hover a few centimetres above flat ground, so the vehicle tips onto one at a few degrees of lean.
  - **Statics**: a single-track towing unit has no roll unknown in the energy solver; it is balanced upright, steering straight, rider upright. Vehicles with a rider or single wheels skip the two-axle lever solver. Masses of the steered body and rider count in `total_mass`, `unit_mass` and `total_com`, summed in the old order so existing vehicles stay bit-identical.
  - **`bicycle_benchmark`**: Meijaard et al. (2007) Table 1, converted to FLU (the x-z inertia entries change sign), origin at the rear contact point. The body colliders are the rider's torso, head and hands. It has stiff Fiala tyres for now (knife-edge rolling comes with `rigid_rolling` in step 2) and a small rear hub motor (40 N·m, 400 W).
  - **Tests** (`vehicles/tests/single_track.rs`):
    - The layout, the centre of mass and the trail (0.08 m, to 1e-9).
    - Static loads by the lever rule (to 1e-3 of the weight).
    - Upright, it stays up on its static loads for 2 s. Leaning 0.02 rad, it falls past 1 rad in 3 s, with the handlebar flopping into the fall, and the rider hits the ground.
    - With feet down, it tips onto a foot and stays there; at 5 m/s the feet go up.
    - Freewheeling without gravity, contacts or stops, with a rider on a conservative servo spring, energy drifts 7e-5 over 10 s.
    - A fork and a swing arm with automatic preloads sit at zero travel.
  - Goldens are unchanged. The first attempt changed the car trajectories: re-summing `total_mass` per unit had altered floating-point rounding.
- **Step 2 (tyres for two-wheelers)**:
  - **Motorcycle tyre data**: no published MF 6.x `.tir` for 120/70 and 180/55 motorcycle tyres was found. Evangelou's PhD thesis (Imperial College 2004, ch. 9 and appendix C) gives complete **MF-MC** sets (Pacejka's motorcycle Magic Formula, with a separate camber sine term `C_γ`) fitted to de Vries & Pacejka's (1997) measurements up to 45° of camber, plus relaxation data. They are a new model, `TireModel::Motorcycle` (`tire/mc.rs`), loaded from TOML (`[axles.tire.mc] file = "Evangelou_120_70_ZR17"`, built in: `Evangelou_120_70_ZR17`, `Evangelou_180_55_ZR17`, `Bicycle_37_622`).
    - The thesis's conventions (side slip `β = −V_y/|V_x|`, SAE forces) map onto ISO-W as `β = tan α`, `F_y`, `M_z` negated; the signs are checked physically (camber thrust and twisting moment towards the lean, aligning trail).
    - `R_o` in the aligning moment is the crown radius (0.06 m front, 0.09 m rear). Only the side slip relaxes, `σ = K_yα0 (c₀ + c₁V + c₂V²)` from Table 9.5. Longitudinal relaxation (0.1 m), radial stiffness (130/140 kN/m) and the unloaded radii (from the size designations) are estimates.
    - The combined-slip loss functions stop at zero (the thesis's fits turn negative beyond about 23° of slip).
    - **Oracle**: the thesis instead of MFeval (which has no MF-MC). Tests (`vehicles/tests/two_wheeler_tyres.rs`): σ/K_yα against Table 9.5 within 7 % (the fit's own error); the cornering and camber stiffnesses against the formulas; F_x peaks of 1.3–1.4 F_z (§9.3.7); leaning 45°, the tyre carries F_y = F_z with under 6° of slip.
  - **Bicycle tyre** (`Bicycle_37_622`): an estimate in the MF-MC form. Cornering stiffness 12 F_z/rad and camber stiffness 0.9 F_z/rad in the range of Dressel & Rahman's measurements, 12 mm trail, peak friction 1.0, the motorcycle curve shapes.
  - **Toroidal contact** (`tire::toroidal_contact`, `Tire::crown_radius`; MF-MC tyres take the file's, `.tir` tyres take `crown_radius`): the contact point is below the crown centre, the deflection is along the road normal, the slip velocities are the crown centre's (Evangelou §9.2), and the rolling radius shrinks by `r_c(1 − cos γ)`. The forces act at the contact point, so a wide tyre's overturning moment follows from the geometry (MF-MC has no `M_x`). Checked analytically up to 60° of lean; crown radius 0 takes the unchanged thin-disc path.
  - **Large camber in MF 6.x**: the MFeval fixtures of the MF 6.1 and new MF 6.2 sample files add 300 random points and a sweep at up to ±55° of camber. They match to 1e-9, as before, without code changes.
  - **Turn slip** (MF 6.2, `turn_slip = true` on a `.tir` tyre; `MfParams::turn_slip`): the ζ factors of Pacejka (2012) §4.3.3 in steady state, fed from the wheel carrier's yaw rate about the road normal and the camber spin (`V_c ≥ VXLOW`).
    - Against MFeval.jl in useMode 222, 1100 points with turn slip: F_x, F_y, M_x, M_y, stiffnesses, trail and relaxation lengths match to 1e-7 or better.
    - MFeval.jl subtracts the camber spin (`φ = −φ_t − (1 − ε)Ω sin γ/V`). With that sign, pure camber through the spin would push away from the lean, so ours adds it; the fixture test feeds MFeval's spin.
    - MFeval.jl leaves ζ₇, ζ₈ (the spin moment) at 1 and uses a garbled `K_zγr0`, so M_z with turn slip is checked for consistency instead: at 0.02 rad of camber, F_y and M_z through the spin agree with the plain camber model to 1e-3. An upright wheel yawing left is pushed right and turned back.
  - **`rigid_rolling`** (a Fiala option): forces linear in the slip without a friction limit, no aligning or rolling moment, as an approximation of the benchmark's knife edges. `bicycle_benchmark` keeps its stiff Fiala tyres until step 3 tunes stiffness and relaxation against the benchmark's eigenvalues.
  - The MF-MC tyres also run the existing transient tests (`tire_dynamics.rs`). Goldens are unchanged.
- **Step 3 (Whipple benchmark)**:
  - **Linear model** (`ground::single_track`): `WhippleParams` (Meijaard et al. 2007, Table 1, in the paper's frame: x forward, y right, z down, origin at the rear contact point) with `benchmark()` and `from_def(def)`; `matrices()` gives `M`, `C₁`, `K₀`, `K₂` by the paper's Appendix A; `WhippleMatrices::state_matrix(v, g)`, `input_matrix()`, `eigenvalues(v, g)`, `stable_speeds(g, v_max)`. `from_def` lumps the chassis, the rider (locked upright) and a rear suspension carrier into the rear frame, the steered body and a fork carrier into the front frame, and takes the wheels' spin inertia with their driveline share; the ground is under the rear wheel at its unloaded radius, the design pose (no sag). `eigenvalues4` (Faddeev–LeVerrier plus Durand–Kerner and Newton polishing) and a small `Complex` type serve the 4×4 matrices without a linear algebra dependency.
  - **Against the paper** (`vehicles/tests/whipple.rs`): the preset's matrices match eq. 5.4 to 1e-9 (the paper's own `benchmark()` values to 1e-12); the eigenvalues at 0 and 5 m/s match Table 2 to 1e-9; the weave speed 4.29238253634 m/s and capsize speed 6.02426201539 m/s to 1e-8.
  - **Full multibody model**: its lean and steer eigenvalues are identified from four slightly perturbed runs per speed (1e-5 rad or rad/s in each coordinate), by dynamic mode decomposition of `(φ, δ, φ̇, δ̇)` averaged over 5 ms intervals for 1 s. The averaging filters the stiff tyres' ringing (undamped at standstill) without moving the slow modes' eigenvalues; without it the fit failed at 0 m/s.
    - Knife-edge limit (tyres stiffened to `C_κ` 1e5, `C_α` 1e6, `σ_y` 2 mm; 0.1 ms steps): within 0.47 % of the benchmark over 0–10 m/s every 0.5 m/s; weave speed 4.292 m/s, capsize speed 6.028 m/s (by bisection of the fitted growth rate).
    - The preset at 1 kHz: within 2.2 % over 1–10 m/s (mostly the castering mode, where the relaxation length is not short against the travel); weave 4.311 m/s, capsize 5.996 m/s.
    - Semi-implicit Euler needed no fix: at 0.1 ms it is exact to the fit's accuracy, and at 1 kHz the tyres' finite stiffness dominates the difference.
  - **`rigid_rolling`** has no low-speed damping. The default damping (ratio 0.25 on the carcass spring with the corner mass) was explicitly unstable against the wheel's spin inertia (`I/r²` ≈ 1.3 kg) at 1 kHz once the tyres were stiff, and pushed the standing bicycle along. At standstill the carcass deflections are therefore undamped springs; `feet_hold_it_up_and_lift_when_riding` runs on ordinary Fiala tyres, as the knife edges bounce on the foot.
  - **`bicycle_benchmark`** now rolls on `rigid_rolling` tyres: `C_κ` 8000 N, `C_α` 1e5 N/rad, `σ_x` 0.03 m, `σ_y` 0.01 m. Stiffer longitudinal springs are unstable at 1 kHz against the wheel's spin inertia, while the lateral ones set the eigenvalues. The explicit step needs `dt` well below `√(m/(C/σ))`: the stiffened knife-edge tyres ran at 0.1 ms but not at 0.2 ms.
- **Step 4 (presets and powertrains)**:
  - **`motorcycle_sport`** (after the Suzuki GSX-R1000 K1–K4, 277 kg with a 72 kg rider): Suzuki's published geometry (wheelbase 1.405 m, rake 23.8°, trail 96 mm), torque curve (118 N·m at 9000 rpm, about 130 kW), gears and final drive; from Evangelou's thesis §9.1 the measured steering damper (6.944 N·m·s/rad), rear spring and damper (divided by an estimated linkage ratio of 2.2² for wheel rates), rider (72 kg, 62 % upper body, lean servo at 11.7 Hz) and the MF-MC tyres. The thesis's front-frame inertias and frame stiffness were never measured, so the inertias, centres of mass (chosen for 51/49 loads and a 0.62 m centre of mass), fork, brakes, drag area, colliders and feet are estimates. The combustion drive has no clutch (the torque map from 0 rpm stands in for it) and a reverse gear that stands for pushing it back.
  - **`bicycle_city`** (93 kg with a 75 kg rider, 44 kg of it leaning): magnitudes after Moore's measured city bicycles and riders; 37-622 tyres; pedalling as an electric motor on the rear wheel limited to 90 N·m and 600 W, with the new `MotorDef::freewheel` (no negative torque). `Bicycle_37_622` now has 250 N·s/m of radial damping (ratio about 0.06; 50 left the frame without suspension hopping under braking).
  - **Tests** (`vehicles/tests/two_wheelers.rs`), with a simple rider in the test (steering torque from lean error and roll rate, plus integral action; throttle holding the front wheel's load; brakes squeezed over 0.5 s and released at the onset of lock):
    - Statics: masses, load shares, and the presets standing still upright on their static loads with zero travel.
    - Motorcycle: 0–100 km/h in 3.3 s (published about 3 s), top speed 283.5 km/h (about 285–295). The drive torque reacts on the swing arm, which gives about 120 % anti-squat (in the range of real sport bikes): the tail rises under drive, and the front lifts at about 0.9 g.
    - Braking from 100 km/h: 45 m with 0.6 of the front brake and the rear skimming the road. The rear load follows the lever rule within 50 N up to 0.96 g. The fork (90 mm of bump travel) runs onto its stop near 1 g; a faster squeeze pitches the bike over, as it would a real one.
    - Bicycle: 7.8 m/s at 250 W, 11.2 m/s at 600 W; 0.48 g of braking from 25 km/h on rim brakes, including the squeeze.
    - Steady turning: the roll angle against the lateral acceleration within 0.44° of the analytic `θ + asin(ρ sin θ/(h − ρ))`, with `tan θ = a/g (1 + Σ I_spin/(r m h))` for the wheels' gyroscopic moment and the load-weighted crown radius ρ: motorcycle at 20 m/s up to 0.8 rad (0.85 g), bicycle at 8 m/s up to 0.3 rad. The bicycle's larger steer angles leave the small-steer formula beyond that (1.1° off at 0.4 rad).
    - Lateral modes of the full models by dynamic mode decomposition of ten states (lean, steer, their rates, yaw rate, lateral velocity, rider lean and rate, both tyres' lateral deflections) from eight perturbed runs, with the speed held by the throttle. This uses a new general eigenvalue solver, `DenseMatrix::eigenvalues` (balancing, Hessenberg reduction and Francis double-shift QR after Numerical Recipes), plus `mul`, `transpose`, `add_scaled` and `inverse`.
    - Motorcycle: weave from 0.7 Hz at 10 m/s to 3.6 Hz at 60 m/s, its damping falling from 20 m/s on (−0.12/s at 60 m/s; 3.3 Hz at 40 m/s against Sharp et al.'s 3.5 Hz); wobble at 10–11 Hz, losing damping with speed, stable with the steering damper and unstable without it (+6.3/s at 50 m/s).
    - Bicycle: self-stable from 4.64 m/s (Whipple: 5.00 m/s). With the tyres' slip and camber forces and the rider's sway, capsize turns slowly unstable at about 5.5 m/s (Whipple: 7.53 m/s), 0.38/s at 8 m/s.
  - **Deviation from Sharp et al.**: their 7.6 Hz wobble, least damped near 13 m/s, comes from a model with torsional frame compliance at the steering head, which lowers the wobble frequency and damps it at speed (Sharp & Alstead 1980; Spierings 1981). The frame here is rigid, and its wobble behaves as in Sharp (1971). A frame-twist DoF would be the addition if it matters.
  - `lean()` in the tests is the roll of the yaw–pitch–roll angles. Roll about the world x axis read low once the bike had turned, and wound the test rider into a spiral.
- **Step 5 (rider controller and action modes)**:
  - **Rider** (`control::ground::rider`, inside `GroundController` for any single-track vehicle with a steering head; `is_single_track()`, `turn_lean(v, κ)`):
    - **Gains**: a discrete LQR at the controller's rate on the Whipple model of the preset (`WhippleParams::from_def`, rider locked) with the steering damper added, state `(φ, δ, φ̇, δ̇)`, input the steering torque; weights 0.05 rad lean and steer, 0.5 and 2 rad/s rates, 0.2 of the largest torque. Zero-order-hold discretisation by Taylor series with scaling and squaring; the Riccati equation by the structure-preserving doubling algorithm (Chu, Fan & Lin 2005), on glam's `DMat4`. Scheduled over 0.5–120 m/s in steps of 8 %, interpolated linearly.
    - **High speed**: with full gains the regulator destabilised the motorcycle's weave from about 45 m/s (at 60 m/s the gains grew a 3 Hz weave the bike alone damps), because the knife-edge model misses the tyres' lag. The steer weights hardly change the gains there (the regulator becomes a steering spring towards `δ ≈ 0.5 φ` whatever they are), so the torque's weight grows as `1 + (v/25 m/s)⁸`: the rider holds the bars ever more loosely and the vehicle's own damping does the rest. Hands off, `motorcycle_sport` itself turns wobble-unstable above about 65 m/s (8 Hz, growing at 70 m/s), near its 78 m/s top speed; no rider tuning fixes that.
    - **Lean reference** from the curvature: `tan θ = v²κ/g (1 + Σ I_spin/(r m h))`, `φ = θ + asin(ρ sin θ/(h − ρ))` (step 4's steady turn), plus an integral on the curvature error (measured as `ω_z/(v cos φ)`, gain `curvature_integral` times `v²/g`, capped at `curvature_correction`), clamped to the new `GroundConfig::max_lean` (0.7 rad). The linear model's steady turn at that lean gives the feedforward steer angle and torque. The rider's own lean stays neutral.
    - **Crawl** below the new `GroundConfig::balance_speed` (1 m/s): a steering-angle servo (5 Hz, critically damped) to the kinematic angle `atan(κ w)/cos λ`; the feet (down below 1.5 m/s) hold the vehicle up. Speed is the existing speed loop; a zero speed request holds the brakes.
  - **Freewheels**: electric drives whose motors all freewheel (`bicycle_city`) decelerate on the brakes (the speed loop's negative demand goes to the pedal; the acceleration bounds take no negative motor torque).
  - **Action modes** (`GroundActionMap`):
    - `vk`: the full-scale curvature defaults to the steering lock's (`tan(0.95 lock cos λ)/w`), and at the commanded speed is capped at `g tan(lean)/v²` (new `GroundActionLimits::lean`, default 0.7 rad), so the whole range stays useful at any speed.
    - `vw` works on single-track vehicles: the yaw rate becomes the curvature at the commanded speed; full scale capped at `g tan(lean)/v`.
    - `raw`: `drive`, `steering`, `lean` (with a rider). `GroundSetpoint::Pedal` gains `lean`; on a single-track vehicle the pedal only brakes below zero (no reverse) and `steering` is the steering torque.
    - `per_wheel` adds `steering` for a steering head and `lean` for a rider: `motorcycle_sport` has `throttle, steering, lean, brake_0, brake_1`.
    - The default full-scale speed of an electric single-track vehicle is also capped at its top speed on the motors' power (drag area and 1 % rolling resistance): `bicycle_city` 11.2 m/s.
  - **`GroundEstimate`** gains `roll` (yaw–pitch–roll), `roll_rate`, `steer_angle` and `steer_rate`.
  - **Tyre fix**: the MF-MC lateral relaxation length `K_yα0 (c₀ + c₁V + c₂V²)` turns negative with the cornering stiffness's sine at extreme loads, which panicked the deflection clamp of a weaving motorcycle; as for MF 6.x, the last positive lengths now hold.
  - **Tests** (`control/tests/rider_loop.rs`; unit tests of the regulator on the benchmark bicycle and of the action modes):
    - Sideways shove at the centre of mass (0.5 m/s worth of the whole mass over 0.1 s): motorcycle at 3, 10, 25 m/s leans at most 0.075 rad, bicycle at 3 and 8 m/s at most 0.079 rad; upright (< 0.01 rad) and straight (< 0.002 1/m) 6 s later, speed held, feet up. At 50 m/s the weave dies out within 15 s.
    - Curvature steps 0 → κ → −κ → 0 (motorcycle 0.02 1/m at 15 m/s, 0.49 rad of lean; bicycle 0.05 1/m at 5 m/s): each within 5 % in 6 s, the lean within 0.008 rad of the steady-turn formula, without falling.
    - Countersteer: turning left, the steering first goes right (4 ms after the step), the lean follows left from 55 ms (bicycle 107 ms), the steering follows left from 240 ms (314 ms).
    - Launch from rest on the feet, feet up at 1.88 m/s, riding upright at 10 m/s (bicycle 5 m/s); stop back onto the feet, standing still.
    - The bicycle balanced at 2 m/s (well below its self-stable range) after a nudge, and crawling straight on its feet at 1.2 m/s.
    - `vw` 0.2 rad/s at 10 m/s gives 0.02 1/m; `raw` brakes on a negative pedal and passes the lean through.
  - The feet are symmetric and flat asphalt is perfect, so launches and stops stay exactly upright; cross slopes and one-footed stances come with step 6's spawns.
- **Step 6 (simulation)**:
  - **Spawns** (`drive::ground_pose`, single-track branch): the wheels' contacts are collinear, so the plane fit would roll the vehicle to the terrain normal. Instead a line is fitted along the heading through the ground under the wheels and the spawn point; the vehicle stands upright, pitched to it, lifted so that no wheel starts below the ground. Ground spawns are at rest, so the feet are down. Where a foot would start inside the ground (a cross slope), the vehicle leans towards the lower side, rolling on its tyres' crowns (about the line through the crown centres), until the lower foot touches (bisection, at most 0.5 rad).
  - **Drive grids**: `half_width` already included the feet (motorcycle 0.51 m, bicycle 0.45 m).
  - **Events**: unchanged. The feet are gear, so standing on them raises `GROUND_CONTACT`/`LANDED` (not terminal); the frame, rider or handlebar touching the ground is `CRASH_TERRAIN`.
  - **Observation terms**:
    - `lean` (2): the yaw–pitch–roll angles' roll and its rate `p + (q sin φ + r cos φ) tan θ`. It needs no wheels, so it works for any vehicle.
    - `rider_lean` (2): the rider's lean relative to the frame and its rate; 0 without a rider.
    - `feet` (1): 1 while the feet are down.
    - `steering` has 2 values (angle and rate) on vehicles with a steering head, 1 otherwise. `CompiledObs::new` takes a new `steering_head` flag.
  - **Recording**: two-wheelers' state messages add `steer_torque` (the rider's plus the damper's, N·m) and `feet`. The rider's lean was already the last of `joints`. `RecordedState` gains both, with serde defaults. Other vehicles' messages are unchanged, so goldens are unchanged.
  - **Python**: `vehicle="motorcycle_sport"` (or `bicycle_city`) works in any ground task; the docstring lists the two-wheelers and their terms.
  - **Stance** (revised after the first commit, when a test on rural maps found a bicycle toppling over its downhill foot on a 12.8° slope):
    - The feet are rigid on the frame and the rider's lower body is lumped into it, so the whole vehicle and rider lean onto one foot. A real rider's weight goes down the leg, which reaches down a slope. The stance is a triangle: the two tyre contacts and the downhill foot.
    - Feet moved beside the centres of mass (along the vehicle), where the triangle is widest: `motorcycle_sport` from (0.45, ±0.42) to (0.62, ±0.45) m (centre of mass about 0.72 m ahead of the rear axle); `bicycle_city` from (0.30, ±0.30) to (0.38, ±0.40) m (centre of mass 0.38 m ahead).
    - A sweep over slopes and headings (across, and diagonally up and down) with the new feet: the motorcycle stands on up to 16° at all headings tried, the bicycle on up to 12°. The bicycle's front wheel carries little on its foot, and heading diagonally uphill it tips back over the line from the rear contact to the foot from 13°.
    - New `FeetDef::max_slope_deg` (default 10°; `motorcycle_sport` 15°, `bicycle_city` 12°): the steepest ground the vehicle stands on. Spawns use the smaller of it and `drivable.spawn_slope_deg`. The limit scores candidates, as the slope limit always did, so a steeper spot is taken only where no other is found.
    - Tried and dropped: the rider straightening the upper body while standing (the lean servo against the frame's roll). Snapping upright threw the bicycle off its wheels. Rate-limited to 0.5 rad/s it helped straight across the slope but not diagonally, and the bicycle rocked between foot and wheels.
    - Stopping on steeper ground than `max_slope_deg` can still tip a two-wheeler over; agents meet that as a crash.
  - **Tests** (`sim/tests/two_wheelers.rs`, `tests_py/test_envs.py::test_motorcycle_rides_off_its_feet`):
    - Both presets standing still for 3 s on flat ground stay exactly upright.
    - On a slope of their `max_slope_deg`, heading across it and diagonally up and down it, they spawn leaning downhill and stay there (lean change < 0.12 rad, < 10 cm) without a terminal event. On 10° the motorcycle settled 0.087 rad beyond its spawn lean: its suspension extends once the foot takes about 40 % of the weight, in about 1 s, briefly unloading the rear wheel. The bicycle rocks gently on its foot for some seconds (its tyres are lightly damped and it has no suspension).
    - On rural maps (2 maps, 8 seeds, spawned on roads and off them) both stand for 3 s without a terminal event.
    - The motorcycle launches off its feet, rides at 10 m/s and turns left, leaning left 0.26 rad.
    - The terms equal the vehicle's state. Over the whole ride the lean rate matches the change of the lean angle (trapezoidal rule over each policy step) within 0.026 rad/s, against rates up to 0.76 rad/s.
    - The recording's last state has the position, steering, steering torque, joints (rider lean last) and feet of the vehicle.
    - A constant steering torque without balance ends in `CRASH_TERRAIN`.
- **Step 7 (viewer)**:
  - **Visuals** (`scene::single_track`, used by `props::wheeled` for any single-track vehicle with a steering head):
    - New mesh primitives `torus` and `ellipsoid`.
    - Motorcycles (combustion powertrain): twin spars, engine, lower fairing, tank, seat, tail, front fairing, screen, headlight, exhaust, swing arm and shock. Bicycles: a tube frame from the bottom bracket, saddle, crank and rack.
    - Wheels: toroidal tyre and rim, hub and spokes (brake discs on the motorcycle).
    - The steered part (fork legs, triple clamp, handlebar, front fairing) turns about the head axis; the fork sliders follow the fork's travel.
    - The rider: pelvis on the frame; torso, head and helmet on the lean joint; arms reach from the leaning shoulders to the steered grips, legs from the hips to the pegs, or to the ground while the feet are down (two-bone IK, `single_track::bend`).
    - `WheeledVisual::single_track` carries the geometry. The viewer's `sync_riders` poses the parts every frame from the state (live and replay).
  - **HUD**: roll (amber past 30°, red past 45°), rider lean, feet, steer angle and rate, a steer-torque bar against the head's `max_torque`, gear and rpm, the `vk` speed and turn setpoints.
  - **Keyboard riding** (single-track vehicles in live mode use `vk`):
    - W/S raise/lower a speed setpoint (0.15/0.4 of full scale per second), Space brings it to zero, A/D turn at 1 of full scale per second.
    - Full stick leans at most 0.35 rad on the motorcycle and 0.2 rad on the bicycle (`sim::RIDE_LEAN`, capping the group's `lean` limit). The turn fades in below 4 m/s.
    - With the `vk` default of 0.7 rad, a full-stick turn crashed the bicycle at every speed and the motorcycle at 4, 12 and 16 m/s. The rider regulator overshoots a sudden lean by about a third to a half. The bicycle rolls over from about 0.35 rad of steady lean: the front tyre's slip angle diverges and the steering runs to its lock, with torque to spare. Treated as a limit of the scripted rider, not of the model; a policy may lean further.
  - **Demo** (`--demo`): riders' curvature is eased (τ 0.5 s) and capped at 1.5× the lateral acceleration allowed at the current speed (2 m/s² motorcycle, 1 m/s² bicycle). The speed setpoint follows `√(a/|κ|)`, with a 3 m/s floor, and is slewed (+1.5/−2.5 m/s²). Lookahead is `max(1.5 v, 8)` m, with the bend sampled to 60 m ahead. Over 10 simulated minutes on the rural showcase map, the motorcycle rode without a terminal event. The bicycle rolled over 4 times, each at crawling speed or on 8–22° slopes.
  - **Replay** restores the steer torque and the feet (`Wheeled::show_feet`) alongside the joints.
  - **Performance**: motorcycle on the rural showcase map at 1920×1080 medium: 114 fps on the Iris Xe (car: 122 fps).
  - **Tests**:
    - Scene: torus volume and normals, ellipsoid volume, two-bone IK keeps segment lengths, both presets assembled around their geometry.
    - Viewer: rider parts follow the state (`rider_parts_follow_the_state`); the keys ride a motorcycle (launch, lean into a turn, stop on the feet); full-stick turns at 20–60 % of full speed stay up on both bikes; a replayed motorcycle leans, steers and stands.
- **Step 8 (`MotorcycleRoadRural-v0`)** → **M5 done**:
  - **Task** (`tasks/motorcycle_road.py`, `motorcycle_road`): `RoadFollowRural` with `motorcycle_sport`, `vk` with `speed = 25` m/s and `lean = 0.45` rad (full-scale curvature `g·tan(0.45)/v²`, or the steering lock's 0.33 1/m), 60 s episodes, the same goals, reward and end conditions (falls end as `ROLLOVER`/`CRASH_TERRAIN`). `action_mode="raw"` (throttle/brake, steering torque, rider lean) is the variant in which the agent balances. Observation (106): `road`, `route`, `on_road`, the new `road_class`, speed, body velocity and rates, `lean`, `steering` (angle and rate), `rider_lean`, `feet`, last action, LiDAR (2 × 36 beams, 40 m, at 1.2 m).
  - **New observation term `road_class`** (3): one-hot paved/gravel/track of the road under the agent, all 0 off the road. The grip and the speeds that suit each class differ.
  - **`BatchSim.group_info`** gains `full_scale` (speed, reverse, curvature, yaw rate of the ground action map), which the scripted rider needs.
  - **Route planning fix** (`lane.rs`): routes nearer than 10 m to the map's edges are skipped, and road-point destinations (maps without a reachable yard) keep 20 m from them. Roads run along and out of the map, so the motorcycle met `OUT_OF_BOUNDS` while following its lane. Road goldens are unchanged.
  - **Scripted rider** (`MotorcycleRoadRural.scripted`): pure pursuit of the route point `clip(1.5 v, 8, 40)` m ahead. Its curvature is low-passed over 0.5 s, using the last action as the filter state. Speed is capped by the steered curvature and by the lane's curvature 5–40 m ahead (with braking distance at 2.5 m/s²), for 2.5/1.5/1.2 m/s² of lateral acceleration and at most 18/12/8 m/s on paved/gravel/track, and floored at 2.5 m/s.
    - Over 32 routes on two map pools it finishes 27, and 15 of 16 on an unseen pool. It falls in some tight track bends and hairpin junctions (radius about 1 m).
    - Tried and dropped: pure pursuit at 5–10 m (weaves, because the lean lags the steering); a lane-tracking law (curvature feed-forward plus offset and heading feedback), which fell at low speed on kinks and weaved; more smoothing or a longer look-ahead; higher lateral acceleration.
  - **Training**: PPO (256 × 64, 2×256, 3M steps, 7 min at 7–10k SPS) raised the return from −58 to +85. On unseen maps (`map_seed` 1000) it reached 8 % success with about 6 of the goals per episode. That is a pipeline check, still improving when it stopped: `runs/MotorcycleRoadRural-v0__moto__1__1790584603` with `policy.json`.
  - **Viewer**: the exported policy rides in `policy` mode (17.6 m/s on the paved road, 125 fps at 1600×900). Its recording replays with the leaning bike and rider.
  - **Tests** (`tests_py/test_envs.py`):
    - The API and vector suites cover the new env.
    - The scripted rider finishes at least 6 of 8 routes, leans more than 0.2 rad, and the episodes it does not finish end in a fall.
    - In `raw` mode, throttle alone ends every episode, at least one of them in a fall.
    - `sim/tests/roads.rs` checks `road_class` against the road under each car, and that routes keep clear of the map's edges.

## Milestone 6: Aircraft (fixed-wing, helicopter, tiltrotor) and large maps

Planned 2026-09-28. Decided with the user at the start:
- **Aircraft**: all three roadmap types. The VTOL is a **tiltrotor**.
- **Maps**: **tiled large maps** (10–20 km), streamed around the agents, with a floating origin in the viewer.
- **Demos**: **one task per aircraft type**.
- **Training**: as before, only enough to test what was built; the user trains the agents.

The rest of this section is proposed and open to change.

Like M4, M6 is split into sub-milestones. Each ends with tests, its demo, a commit and a push.
- **M6a**: large maps, then the shared aerodynamics, then fixed-wing aircraft. Fixed-wing aircraft need the space most, and the helicopter and the tiltrotor reuse their wing and air-data parts.
- **M6b**: helicopter. Its rotor model (forward flight, flapping) also serves the tiltrotor's proprotors in edgewise flow.
- **M6c**: tiltrotor.

**Already in place**:
- **Rigid bodies**: Free-joint rigid bodies with implicit-midpoint gyroscopics.
- **Multirotor rotors**: first-order motors, thrust and torque ∝ ω², rotor drag, ground effect, battery.
- **Environment**: ISA atmosphere with speed of sound; wind with a log profile, 1−cos gusts and Dryden turbulence (low-altitude form only, floored at 2 m/s airspeed).
- **Sensors**: IMU, GPS, baro, mag, rangefinder, LiDAR.
- **Contacts**: penalty contacts with gear/crash semantics and `LANDED`.
- **Multi-agent and Python stack**: batched and multi-agent worlds, the Python stack, PPO/SAC, policy export, viewer policy mode and MCAP replay.

**Not in place**:
- **Maps**: all maps are single monolithic `HeightGrid`s (2 km at 1 m ≈ 20 MB). At 1 m, 20 km would be about 1.6 GB of heights alone, and erosion runs sequentially over the whole grid. The viewer meshes the whole map at load time and converts f64 to f32 without an origin shift.
- **Aerodynamics**: `AirData` lives in the multirotor module. There are no aerodynamic surfaces or airspeed terms.

### M6a: Large maps and fixed-wing aircraft

#### Design
- **Tiled large maps** (`world::tiles`, `procgen::large`):
  - **Two layers.**
    - A **coarse layer** covers the whole map (e.g. 16 km at 8 m, 2000², about 40 MB). It is generated globally: terrain noise, erosion at the coarse scale, hydrology (lakes, rivers as the flow network), materials, and for rural maps farm sites, the road network and **airstrips** (a flat mown grass or asphalt runway, 400–800 m).
    - **Detail tiles** (256 m at 1 m) are pure functions of (map, tile index). Each is made by bicubic upsampling of the coarse heights plus detail noise, road blending and materials from the coarse network, and scatter from the tile's own seed stream (`map/<gen>/<tile>`). A tile's content is then independent of when, by which thread and in which order it is generated. Seams are continuous because the upsampling and noise are global functions and scatter is assigned by candidate position.
  - **`TiledWorld`** implements `Terrain` and `StaticGeometry`.
    - Queries go to the detail tile when it is loaded or loadable, and to the coarse layer beyond a query's `far` range (long LiDAR rays, AGL of high aircraft).
    - Tiles sit in a shared, thread-safe LRU cache (`Arc`; a few hundred tiles, about 150 MB) used by all worlds of a batch. Obstacles are per tile (one BVH each); ray casts walk the tiles along the ray.
    - `StaticWorld` becomes an enum (`Grid` or `Tiled`), or `HeightGrid` is wrapped behind the trait, so existing maps, hashes and goldens stay unchanged.
  - **Hash**: generator version, config, seed and the coarse layer's content, plus a golden sample of detail tiles (tile hashes checked in tests). The map file caches the coarse layer; tiles are recomputed (or cached on disk under their own keys).
  - **Presets**: `wild` and `rural` gain `large` (16 km). The pool `count` for large maps is small (1–4), since tiles, not whole maps, are the unit of cost.
- **Viewer**:
  - A **floating origin**: render space is re-centred on the camera every 1 km, and transforms are computed in f64 relative to the origin before converting to f32.
  - **Streaming**: detail tiles are meshed as today's chunks, with LOD, near the camera (up to about 1.5 km, with vegetation). Coarse-layer chunks at strides of 16–64 m cover the far field out to 8–12 km, fogged.
  - **An aerial view distance** that grows with the camera's height.
  - **Target**: ≥ 60 fps at 1080p medium on the Iris Xe while flying at 30 m/s at 200 m AGL.
- **Shared aerodynamics** (`vehicles::aero`):
  - `AirData` moves here and gains the air-relative velocity and the speed of sound.
  - **`AeroSurface`**: a lifting surface (area, chord, span, position, incidence, lift curve with stall, drag polar, control surface with effectiveness τ). It has a post-stall flat-plate blend (Beard & McLain's sigmoid) and optional α/β tables, and is used by wings, tails and helicopter fins.
  - **Wind**: Dryden gains the medium/high-altitude form (MIL-F-8785C above 2000 ft) and optional rotational gusts `p_g, q_g, r_g`.
  - **New sensor**: `pitot` (airspeed with noise and lag).
  - **New observation terms**: `air_data` (airspeed, α, β) and `wind_body`.
  - **New state columns**: airspeed, α, β. Goldens are re-blessed after an A/B check of trajectories in a worktree.
- **Fixed-wing aircraft** (`vehicles::fixedwing`, `type = "fixed_wing"`, `Family::FixedWing`):
  - **Structure**: a rigid body with an aerodynamic model, propulsion, control surfaces with servo rate and lag, landing gear, colliders and a battery or fuel.
  - **Aerodynamic model**:
    - Either **stability and control derivatives** (CL, CD polar, CY, Cl, Cm, Cn in α, β, p̂, q̂, r̂, δa, δe, δr, δf, with stall blending), as in Beard & McLain;
    - or **coefficient tables** in α, β, Mach and δ, as JSBSim models are written.
    - A component build-up from `AeroSurface`s is also available, for the tiltrotor's wing.
  - **Propulsion**: a propeller with `C_T(J)`, `C_P(J)` tables on an electric motor (Kv, resistance, current limit, battery) or a piston engine (power map, mixture ignored). It is modelled as a rotor with inertia, torque reaction and gyroscopic moments.
  - **Landing gear**: a light `Gear` force element in the manner of JSBSim's LGear.
    - A strut spring and damper along the gear axis.
    - Rolling friction along the wheel plane, side friction across it, brakes, and a steerable nose or tail wheel.
    - It is not the M2 tyre model, which would need a 1 kHz tyre relaxation for no benefit here.
  - **Spawns**: in the air at trim (heading, airspeed, height AGL), or on a runway on the gear.
  - **Presets**:
    - `aerosonde_like`: a 13.5 kg UAV at 25 m/s, from Beard & McLain's *Small Unmanned Aircraft*, electric.
    - `c172_like`: a Cessna 172, from JSBSim's c172x model data and the POH, with piston engine and tricycle gear. It is used for the JSBSim oracle and runway takeoffs.
  - **Events**:
    - A crash is any non-gear contact, or gear touchdown above a sink-rate limit.
    - `LANDED` means on the gear, slow.
    - A new non-terminal `STALL` bit is raised when α exceeds the stall α.
- **Oracle**: **JSBSim** (PyPI `jsbsim`), run offline like Chrono to generate committed fixtures (`tools/gen_jsbsim_fixtures.py`, `make fixtures-jsbsim`).
  - **Checks**: trim over an airspeed sweep (α, elevator, throttle); linear modes (short period, phugoid, Dutch roll, roll subsidence, spiral) from JSBSim's linearisation; doublet time histories; the c172 takeoff roll.
  - **Also**: Beard & McLain's published Aerosonde trim and transfer functions, and analytic checks (glide ratio `L/D`, turn rate `g·tan φ / V`, energy with the engine off).
- **Control** (`control::fixedwing`, gains scheduled on dynamic pressure and derived from the model, as for the multirotor):
  - **Action modes**:
    - `raw`: aileron, elevator, rudder, throttle, optionally flaps.
    - `rates`: p, q, r and throttle; a rate PI with turn coordination.
    - `attitude` (default): roll, pitch and airspeed; TECS handles throttle.
    - `guidance`: course rate or course, climb rate or altitude, airspeed; TECS plus L1 lateral guidance, as in PX4.
  - **Takeoff and landing**: a scripted runway takeoff and a glide-slope landing helper in the task layer or scripted drivers, not in the controller.
- **Viewer**:
  - **Visuals** built from the definition: fuselage, wings, tail, control surfaces deflecting, prop disc, gear.
  - **HUD**: an artificial horizon, airspeed, altitude and vertical speed, α and β, throttle, surface deflections, stall warning.
  - **Keyboard flight** in `attitude` (and `guidance`), a chase camera suited to high speeds, and replay.
- **Demo**, **`FixedWingWaypoints-v0`**: `aerosonde_like` in `attitude` mode on a large wild map.
  - It starts in the air at trim, 150 m AGL, and flies through waypoints 1–3 km apart around the relief, in wind and turbulence.
  - Terrain contact is a crash. A rangefinder or LiDAR fan looks ahead and down for terrain.
  - **Reward**: progress to the goal, a bonus per waypoint, a penalty below a safe AGL and on stalling, and smoothness.

#### Implementation order
| # | Step | Done when |
|---|---|---|
| 1 ✅ | `world::tiles` + `procgen::large`: coarse layer (terrain, erosion, hydrology, materials, rural roads and airstrips), detail tiles, `TiledWorld` with the tile LRU, hashes, `large` presets | Tiles are identical whatever the access order and thread count; heights and normals are continuous across tile seams; queries agree with a monolithic grid built from the same functions; the coarse 16 km layer generates in ≤ 15 s and a tile in ≤ 50 ms; memory stays under the cache bound; existing goldens are unchanged |
| 2 ✅ | Viewer: floating origin, streamed tiles and far-field coarse chunks, aerial view distance | A drone flies across a 16 km map without jitter at the far edge; ≥ 60 fps at 1080p medium at 30 m/s and 200 m AGL on the Iris Xe |
| 3 ✅ | Shared aero: `AirData` move, `AeroSurface`, medium/high-altitude Dryden, `pitot`, the `air_data`/`wind_body` terms, state columns | Lift, drag and moment of a surface match analytic thin-aerofoil and flat-plate values; Dryden spectra match MIL-F-8785C at altitude; goldens re-blessed after an A/B check |
| 4 ✅ | `FixedWing` family: aero model (derivatives and tables), propeller and motor or engine, gear, presets, wiring through vehicles, sim, recorder and Python | Both presets trim in level flight; engine-off glide conserves energy with drag accounted for; they stand on their gear; scenarios spawn them in the air and on a runway |
| 5 ✅ | JSBSim fixtures and validation | Trim α, elevator and throttle within 5 % (or 0.5°) of JSBSim over the airspeed sweep; mode frequencies and damping within 10 %; doublet responses close; c172 takeoff roll within 10 % of JSBSim and the POH |
| 6 ✅ | Control and action modes (`surfaces`, `rates`, `attitude`, `guidance`) | Rate and attitude steps meet rise and overshoot bounds across the speed range; coordinated turns keep β small; altitude and airspeed hold under wind and turbulence; L1 follows a straight and a circular path |
| 7 ✅ | Viewer: visuals, HUD, keyboard flight, cameras, replay | The Aerosonde flies by keyboard over a large map at ≥ 60 fps; recordings replay |
| 8 ✅ | `FixedWingWaypoints-v0`: task, scripted pilot, short training, export, viewer, replay | The task trains end to end; the exported policy flies in the viewer; a recorded episode replays |

#### As built
- **Step 1 (tiled large maps)**:
  - **`world::tiles`**:
    - `TiledMap` implements both `Terrain` and `StaticGeometry` over a `TileSource` (`tile(tx, ty) -> Tile`, `owner(id)`).
    - A `Tile` is a `HeightGrid` (its core plus a ring of `MAX_SEARCH_CELLS` = 16 cells, so normals and surface searches never cross a tile edge) and an `ObstacleSet`. An obstacle is stored with every tile whose core, grown by `reach` (4 m), it overlaps.
    - Global obstacle ids are `(scatter cell << 8) | (k << 2) | (part << 1) | rock`, so contacts and caches keyed by ids work across tiles.
    - **Queries**:
      - Point queries clamp to the extent.
      - Shape queries within one core ± reach go to that tile; others merge candidates from every overlapped tile, sorted and deduplicated.
      - Rays walk tiles with a DDA up to `detail_range` (2 km) and continue on the coarse grid.
      - `height_bounds` uses the tile where possible, otherwise coarse ± `pad` (a bound on Catmull–Rom overshoot plus noise amplitudes).
    - **Cache**: a shared map of `OnceLock` entries with an LRU stamp and eviction above the capacity, plus a thread-local list of the 4 most recent tiles. Tiles are generated sequentially, with no rayon inside, so pool threads never wait on each other.
  - **`StaticWorld`**:
    - `terrain()` / `obstacles()` return the enums `MapTerrain { Grid, Tiled }` / `MapObstacles { Set, Tiled }`.
    - `grid()` / `obstacle_set()` return the monolithic parts and panic on tiled maps.
    - `tiled_hash` is the map hash of a tiled map.
    - Map files cannot store tiled maps (the map is rebuilt from config and seed; only the coarse layer is cached).
  - **`procgen::large`** (`WildPreset::Large`, `"large"`; `WildConfig.tiles: Option<TilesConfig>`, not serialised when absent):
    - **Coarse layer**: the wild landform at 8 m, eroded and routed through the existing hydrology (lakes, moisture scaled by vertex area), plus coarse materials. It is cached as `{key}.coarse` (postcard + zstd, blake3-checked).
    - **Tile heights**: Catmull–Rom of the coarse heights, plus mid-band fBm (1.2 m at 48 m), plus the wild detail noise.
    - **Tile water**: the highest coarse lake level in the 3×3 neighbourhood wherever a cell's lowest corner lies below it.
    - **Tile materials**: `MaterialRule`, factored out of `wild.rs` with `LineSeeds` and `terrain_seeds`, so the wild goldens are unchanged.
    - **Scatter**:
      - Candidates per 16 m scatter cell from `tree_seed.child_index(cell)` with a fixed number of draws.
      - The conflict pass is a *local priority rule* instead of the monolithic fixed-order pass: a candidate survives if no candidate within `min_spacing` ranks higher. Every tile therefore decides the same trees without a global order.
      - Rocks are kept clear of trees.
    - The map hash covers the generator version, config, seed and coarse layer. The tiles follow from them and are checked by `fixtures/golden_hashes_tiled.toml` (hashes of sample tiles).
  - **Numbers (laptop)**: 16 km coarse layer 1.8 s cold (target ≤ 15 s); one tile 33 ms (≤ 50 ms), about 2.5 MB with about 2300 obstacles. The default cache is 128 tiles (about 320 MB; `AUTONOMOUSIM_TILE_CACHE`).
  - **Sim**: `map = { type = "wild", preset = "large" }` works for aerial groups (spawns sample tiles lazily). Ground groups on tiled maps are rejected at compile time, since drive grids would cover the whole map. The drive grid now uses `query_candidates` instead of iterating the obstacle set.
  - **CLI**: `mapgen --preset large` prints the tile layout; `--preview` renders the coarse grid without obstacles; `--out` fails for tiled maps.
  - **Deferred**: rural roads and airstrips on the coarse layer. Airstrips (a flattened, graded strip with a runway material) come with step 4, where fixed-wing spawns need a runway; large rural maps come later.
  - **Tests**:
    - `world/src/tiles.rs`, with an analytic source: queries against a single grid, a bounded cache, threads seeing identical tiles, and long rays.
    - `procgen/tests/large.rs`:
      - golden hashes independent of access order and threads;
      - neighbouring tiles agree on their overlap;
      - queries agree with a stitched 4×4-tile grid;
      - scatter and water invariants, a bounded cache, and the cached coarse layer;
      - an ignored timing test.
    - `sim/tests/large.rs`: 16 drones hold position on a 4 km tiled map; a ground group is rejected.
- **Step 2 (viewer on large maps)**:
  - **Render origin**:
    - `convert::RenderOrigin` (ENU f64) is subtracted before the f32 conversion of every world position: vehicle roots, the camera, gizmos and map entities.
    - Each map entity carries an `Anchor`; its mesh is built relative to it (`scene::terrain::terrain_mesh` / `water_mesh` and `props::props_grouped` take an anchor).
    - `world_view::recenter` moves the origin to the camera, snapped to 1 km, once the camera is more than 1 km away on either axis, and re-places the anchored entities. The camera eye is kept in ENU (`CameraRig::eye`), so nothing else changes.
    - Monolithic maps keep their meshes in map coordinates (anchor 0).
  - **Streaming** (`world_view::Streamer`):
    - The coarse layer is meshed at start as one chunk per tile (32 × 32 cells of 8 m; about 20 ms for 4096 chunks), with its own LOD bands (8 m out to 1.5 km, then 16, 32 and 64 m).
    - Forest floor is drawn there in canopy green, so distant forest does not turn into bare ground where the detail tiles end.
    - Detail tiles within the tile radius (low 500 m, medium 900 m, high 1400 m) are built nearest first, at most four at a time, on a 2-thread pool. A job generates or fetches the tile (`TiledMap::tile`), meshes its 4 × 4 chunks (terrain at the stride for its distance, water, near and far props) and sends them over a channel.
    - Props are drawn with the tile that owns them (`TileSource::owner`), so obstacles stored in several tiles are drawn once.
    - A detail tile hides the coarse chunk it replaces (`CoarseOf`) and is dropped beyond 1.25 radii. The view holds each shown tile's `Arc<Tile>` for LOD rebuilds, independently of the map's LRU.
    - `MapView::surface_height` answers from shown tiles or the coarse layer, so the camera never generates tiles on the main thread.
  - **View distance**:
    - Base: `Quality::far_view_distance` on tiled maps (2.5/4/6 km), unchanged on monolithic maps.
    - It grows with the camera's height above ground (10 m per metre on tiled maps, 4 on monolithic ones) up to 12 km. Fog and the far plane follow (`update_view_distance`).
    - The HUD shows the detail tiles on screen and being built, and the view distance.
  - **Demo options**: `--demo-speed`, `--demo-agl`, `--demo-turn`, and `--demo-camera`, where the free camera flies the demo so speeds above the multirotor's 12 m/s velocity limit are possible.
  - **Spawns on tiled maps**: random spawn candidates now come in batches of 25 from one tile each. The viewer's 2 m AGL spawn in dense forest had generated a tile per rejected candidate (8.5 s to start; now 0.7 s). Monolithic maps draw as before.
  - **Measured** (Iris Xe, 1080p medium):
    - camera flying 30 m/s at 200 m AGL across the 16 km map: 73–74 fps after warm-up, about 65 detail tiles on screen, none waiting;
    - showcase map unchanged at 69 fps.
    - The jitter-free far edge follows from the origin (checked by the test); a flight by the fixed-wing in step 7 will repeat the check with a real aircraft.
  - **Tests**: `tiled_maps_stream_tiles_around_the_camera` (viewer, on a 2 km tiled map):
    - all 64 coarse chunks are spawned;
    - the tiles within the radius are streamed, and coarse chunks under them hidden;
    - moving away drops the old tiles and brings the new ones;
    - the origin snaps and carries every anchored entity.
- **Step 3 (shared aerodynamics)**:
  - **`vehicles::aero`** (the former `multirotor::aero`, re-exported there):
    - `AirData` gains `speed_of_sound` and `gust_rates` (the air's angular velocity in the body frame). The air-relative velocity is a method rather than a stored field (`relative(v)`, `flow(attitude, v, ω)`): the vehicle moves within the tick, so each model computes it from its own state.
    - `AirFlow`: body-frame air-relative velocity and rates, airspeed, α = atan2(−v_z, v_x), β = asin(−v_y/V) (FRD sign conventions on the FLU body), Mach and dynamic pressure; `at(r)` gives the velocity of a body point.
  - **`AeroSurface`** (`aero/surface.rs`, TOML-loadable, radians):
    - Area, span, chord, aerodynamic centre, `roll` about body x (0 wing, ±π/2 fin), incidence.
    - Lift curve `cl0 + cl_alpha·(α + τδ)`, with `cl_alpha` from Helmbold when absent. Polar `cd0 + cl²/(π·e·A)`.
    - Post-stall flat plate: normal force `cd90·sin α`, centre of pressure moving to mid-chord. Blended with Beard & McLain's sigmoid (stall angle, sharpness M = 50).
    - Optional `Flap`: τ = 1 − (θ_h − sin θ_h)/π and Δc_m = −½·sin θ_h·(1 − cos θ_h)·δ from thin-aerofoil theory, or a given τ.
    - Optional `AlphaTable` (cl, cd, cm against α) replaces the parametric curves.
    - `wrench(flow, δ)` gives force and moment about the centre of mass. Only the chord-plane flow counts (no sweep).
    - β tables are deferred to the fixed-wing body model (step 4), where JSBSim-style coefficient tables live.
  - **Dryden** (`world::environment::wind`):
    - `DrydenScales::at` covers low altitude up to 1000 ft; the medium/high-altitude model from 2000 ft (isotropic, L = 1750 ft, σ from the MIL-HDBK-1797 exceedance table at the light/moderate/severe level of W20, interpolated between levels); and linear interpolation between. Height above ground stands for altitude.
    - Rotational gusts: `Dryden::with_rotational`, `step_rotational(dt, V, span)`, `rates(scales, span)`. `p_g` is first-order with the MIL-F-8785C variance. `q_g = ∂w_g/∂x` and `r_g = −∂v_g/∂x` come through their lags (4b/πV, 3b/πV), discretised with a first-order hold; a zero-order hold gave 10–15 % low spectra. Turbulence axes stand for body axes.
    - Rotational gusts are drawn only for vehicles with `Vehicle::gust_span()` (none yet), so existing turbulence streams are unchanged.
  - **Sensors**:
    - `BodyKinematics.wind` (serde default).
    - `pitot` sensor: axial dynamic pressure `½ρ·max(0, v·x̂)²` at the probe (lever arm included), per-episode offset (2 Pa) and noise (1 Pa); readings are differential pressure, indicated airspeed and true airspeed.
  - **Sim**:
    - Agents fill `speed_of_sound` from the atmosphere and `gust_rates` when the vehicle wants them.
    - Observation terms `air_data` (V, α, β), `wind_body` and `pitot` (indicated airspeed).
    - State column `air_data` (V, α, β; `STATE_DIM` 33).
  - **Goldens**: an A/B run against HEAD (hashing only the first 30 state columns and skipping `/meta`) matched for all four scenarios. The re-blessed hashes change only through the new column and the `/meta` field list.
  - **Tests**:
    - `aero`: flow angles and wind.
    - `aero::surface`: 2-D thin aerofoil (2π, τ and Δc_m of a quarter-chord flap), Helmbold slope and induced drag, flat plate at 45–180°, a bounded and continuous lift curve over ±180°, tail moment and pitch damping, fin weathercocking, tables and validation.
    - `wind`: altitude scales (table values, continuity at 1000/2000 ft); PSDs at 3000 m of u, w, p_g, q_g, r_g against MIL-F-8785C (within 5 %, tolerance 12 %).
    - `sensors`: pitot through wind and crosswind, lever arm, offset statistics.
    - `obs`: `air_terms`.
- **Step 4 (the `FixedWing` family)**:
  - **Aerodynamic model** (`fixedwing::aero`, whole aircraft, `model = "derivatives" | "tables"`):
    - `Derivatives` (Beard & McLain): stability axes; lift blended with the flat plate `2·sgn α·sin²α·cos α` through the stall sigmoid (`alpha0`, M = 50); drag `cd0 + cd_alpha·α +` induced polar, blended to `cd0 + cd90·sin²α`; side force, roll, pitch and yaw in β, p̂, q̂, r̂ and the surfaces.
    - `Tables` (JSBSim-style): per axis a sum of `Term`s, each a scale times a product of variables (`Var`: α, β, |β|, p̂, q̂, r̂, α̇ĉ, surfaces, |δe|, flap, Mach, h/b, stall) and 1-D curves or 2-D tables (`table::Curve`, `table::Table`). Terms use free-stream or slipstream pressure (`Pressure::Slipstream`, ½ρ(v_axial + 2v_i)², JSBSim's `qbar-induced`). Wind-axis forces (−D, Y, −L) go to the body through the wind-to-body rotation; moments are FRD about `aero_reference`. `stall_hysteresis = [lo, hi]` drives the `stall` variable.
    - Both are converted FRD → FLU (F = (Fx, −Fy, −Fz), M = (l, −m, −n)) plus `r_ref × F`. Rate terms use V ≥ 0.5 m/s.
    - Control signs come from the sign of each surface's moment derivative, so +1 rolls right, pitches up and yaws right whatever the source's convention. The stall angles are found by scanning C_L over ±0.6 rad.
  - **Propulsion** (`fixedwing::propulsion`):
    - Propeller `T = ρn²D⁴C_T(J)`, `Q = ρn²D⁵C_Q(J)` (or `C_P/2π`). Polynomials of degree ≤ 2 are expanded in n so they stay finite at n = 0; tables read 0 below 10⁻³ rev/s.
    - Electric motor (Kv, R, i₀, current limit, supply voltage or battery): the current is clamped to [0, i_max], with no regeneration.
    - Piston engine: constant torque with friction fraction f (default 0.2), Gagg–Ferrar altitude factor, and an idle fraction set so the static propeller idles at `idle_rpm`.
    - The rotor speed is integrated semi-implicitly, `Ω' = (JΩ/dt + a)/(J/dt + b + cΩ)` (engine torque a − bΩ, propeller cΩ²). The body gets the reaction torque and the rotor's angular momentum.
  - **Gear** (`fixedwing::gear`): JSBSim-LGear-like strut spring and damper along the terrain normal (optional rebound damping). Tanh friction (0.1 m/s) holds a friction circle, with rolling, side and brake coefficients and a steerable wheel. Wheels are gear-group spheres that only other agents and water see. Gear loads enter as `ContactPoint`s (`HitKind::Terrain`/`Water`), so the crash rule (sinking faster than `crash_speed`), `GROUND_CONTACT` and `LANDED` apply unchanged.
  - **Model** (`fixedwing::model`): the step is split into phases (`begin_step`, `apply_controls`, `apply_gear`, `apply_contacts`, `finish_step`). Servos are a first-order lag plus a rate limit. Ground effect for the aero terms uses h/b up to 1.2 spans (`Vehicle::ground_effect_range`). Also `energy(g)` and `external_power()` for the energy checks, and rotational gusts through `gust_span()`.
  - **Trim** (`FixedWing::trim(V, ρ, γ, flap, g)`): an 8 × 8 Newton iteration with a finite-difference Jacobian and backtracking. Unknowns are α, θ, φ, δa, δe, δr, throttle and Ω; equations are the six body accelerations, the rotor torque balance and the flight-path angle. A Levenberg–Marquardt step takes over where a clamped motor current zeroes the throttle column. Trim fails when a command leaves [−1, 1] or the throttle leaves [0, 1]. The aerodynamics are evaluated on the unstalled hysteresis branch.
  - **Presets**:
    - `aerosonde_like`: Beard & McLain's Aerosonde, with the electric propulsion of the book's supplement, a pusher prop and an estimated tricycle gear. It trims from 18 to about 30 m/s; above that the motor lacks the thrust.
    - `c172_like`: generated by `tools/gen_c172_like.py` from JSBSim's c172p (not c172x, which counts the lift slope twice), with mass properties from JSBSim at 1880 lb.
      - JSBSim's IO-320 model makes about 575 N·m at full throttle at sea level (about 205 hp static, 2535 rpm, 2.1 kN), well above its 160 hp rating, so `max_power` is calibrated to JSBSim's static run.
      - Takeoff at full throttle, rotating at 55 kt: lift-off after 278 m (JSBSim reaches 55 kt in 199 m; the POH gives about 270 m at 2400 lb).
  - **Control** (`control::fixedwing`): the action mode is named `surfaces`, because mode names must be disjoint across families and `raw` belongs to ground vehicles. It has 4 components: aileron, elevator, rudder, and throttle mapped from [−1, 1] to [0, 1]; flaps stay up and the brakes off. `FixedWingSetpoint::Surfaces(FixedWingInput)` passes through `FixedWingController`, which is a placeholder for step 6. `Command::hold(&Vehicle)` holds an aircraft at the input it was reset with (trim, or idle on the brakes).
  - **Sim**:
    - **In the air**: spawns are trimmed for level flight at `spawn.airspeed` (a range; default 1.5 × `FixedWingDef::stall_speed` at the spawn density), turned to the spawn heading, with the steady wind added to the velocity. The `tilt_deg`, `speed` and `rates` perturbations apply on top. An aircraft that cannot be trimmed starts level at full throttle.
    - **On the ground**: `spawn.on_ground` puts the aircraft on its gear at `FixedWingDef::resting_pose`, brakes set.
    - Aerial groups keep the 500 Hz default.
    - **Events**: new non-terminal `STALL` bit (15) while airborne beyond a stall angle.
    - **Recordings**: state messages gain `surfaces`, `throttle`, `rotor_speed`, `airspeed`, `alpha`, `beta` and `gear_loads`.
  - **Viewer**: a placeholder, a wing and fuselage box with wheel spheres, the multirotor chase camera, and keys on elevator and ailerons, until step 7.
  - **Deferred**: airstrips on large maps. Runway starts use flat ground (the flat test world, or open terrain).
  - **Goldens**: unchanged. New fields are optional, and the event names are not recorded.
  - **Tests**:
    - Unit tests: tables and curves; aero axes, signs, stall and slipstream terms; propeller polynomial expansion; the electric steady state and windmilling; piston idle and power; gear strut and friction.
    - `vehicles/tests/fixedwing.rs`:
      - presets consistent;
      - trim over 18–28 m/s (Aerosonde) and 30–60 m/s (C172), faster flight needing less α, climb needing more throttle, flaps reducing α, and trims beyond the envelope failing;
      - the trim held for 5 s (height within 1 m, speed within 0.3 m/s);
      - control signs;
      - engine-off glide, where the energy lost equals the work of the external forces within 2 %, with glide ratio 5–20;
      - standing on the gear, where the wheel loads carry the weight within 2 % and no airframe contact occurs;
      - the C172 takeoff.
    - `sim/tests/fixedwing.rs`: trimmed air spawns hold level flight; trim in an 8 m/s wind (airspeed, not ground speed, at 1.5 × V_s); standing on the brakes (`LANDED`), then taking off with `surfaces` actions; full up elevator raises `STALL`; recorded fields match the live aircraft.
    - `tests_py/test_native.py::test_fixed_wing`.
- **Step 5: JSBSim fixtures and validation**:
  - `tools/gen_jsbsim_fixtures.py` (`make fixtures-jsbsim`, JSBSim 1.3.1 from PyPI in `$(ORACLES)/jsbsim-venv`; the target also regenerates `c172_like.toml`) writes `fixtures/jsbsim/c172.json` for the c172p at 1880 lb:
    - `trim`: level trims at 1000 ft, 60–120 kt;
    - `modes`: eigenvalues of `FGLinearization` at 90 kt, longitudinal (Vt, α, θ, q, rpm) and lateral (β, φ, p, r) blocks, classified;
    - `doublet`: ±0.05 normalised elevator, 1 s each from t = 1 s, at 90 kt;
    - `takeoff`: 10 s full throttle on the brakes (static rpm and thrust), then the roll to 55 kt.
  - **Engine throttle map**: JSBSim's piston engine goes through manifold pressure, so its torque is far from linear in throttle (static: 53 N·m at idle, 236 at 0.6, 578 at 1). A linear map needed throttle 0.39–0.71 where JSBSim trims at 0.60–0.80. `PistonEngineDef.throttle_curve` (optional `Curve`, serde default) maps throttle to the gross torque fraction; `gen_c172_like.py` fits it from 11 static runs (fraction = (τ/T_rated + f·ω/ω_rated)/(1+f)). This is an independent calibration: the trims then agree without tuning. The curve's value at 0 sets the idle: 767 rpm static, as in JSBSim (the `idle_rpm` of 550 now only matters without a curve).
  - `vehicles/tests/jsbsim.rs`:
    - `trim_sweep`: α and elevator within 0.0006 rad, throttle within 2.3 % (bound 5 %), rpm within 0.3 % (bound 2 %).
    - `linear_modes`: central-difference Jacobians of the simulated derivative about the trim (state: body velocity, body rates, attitude error rot₀·exp(δ), rotor speed; dt = 1e-5; the derivative comes from the second of two steps so the α̇ terms act), then eigenvalues by Faddeev–LeVerrier and Durand–Kerner (no linear-algebra dependency). Phugoid 0.268 vs 0.270 rad/s, ζ 0.095 vs 0.099; short period 6.39 vs 6.39, ζ 0.632 vs 0.629; Dutch roll 2.225 vs 2.225, ζ 0.200 vs 0.200; roll −6.41 vs −6.41; spiral −0.0175 vs −0.0176 (bounds 10 %; the spiral only to 0.05).
    - `elevator_doublet`: JSBSim's elevator deflections replayed about our trim with an instant servo (JSBSim's c172p has none; the preset keeps its 50 ms lag). RMS errors: q 1.6 %, θ 0.9 %, α 1.6 % of the peaks (bound 5 %); the worst sample, 15 % of peak q, sits at a step, where the sample phases differ (bound 20 %).
    - `takeoff_roll`: static 2537 vs 2538 rpm and 2106 vs 2107 N; 199.2 m and 13.66 s to 55 kt vs 198.6 m and 13.67 s (bound 10 %). The POH (C172P, 2400 lb, 271 m ground roll) is not directly comparable at 1880 lb: scaled by weight squared it gives about 165 m, the same order.
- **Step 6: control and action modes** (`control/src/fixedwing/{mod,tuning,guidance}.rs`, PX4-style):
  - **Model from the aircraft** (`FlightModel`): level trims from 1.05 V_s up to the top speed at full throttle (the trim table gives surface and throttle feedforward and α), the steepest climb at the design airspeed, and `FixedWing::control_derivatives` (new; central differences of the static loads about a trim: moment per surface, per body rate and per α, lift slope, thrust per throttle; `static_loads` takes body rates) at six `LinearPoint`s over the speed range, interpolated in airspeed and scaled by ρ/ρ_design. Axes are pilot axes (roll right, pitch up, yaw right; `to_pilot`).
  - **Rate loop** (at physics rate): `u = u_trim(V) + B(V)⁻¹·(J·a − D(V)·ω − k_α·M_α(V)·α̂)`, with `a = K_p·e + ∫K_i·e`, `K_p = ω_b = min(10, 0.25/τ_servo)` (yaw half of it), `K_i = 0.2·K_p²`, the integrator limited to 0.3 of the control power and stopped on saturation. The pitch feedforward cancels the α stiffness with an α estimate driven by the reference-filtered pitch-rate setpoint (`α̂̇ = q_ref − L_α/(mV)·α̂`), not by measurement, which kept gust response and stability.
  - **Attitude loop**: P gain `ω_b/3`, turn coordination (the heading rate g·tanφ/V as body rates) and a yaw-rate term of 2·β.
  - **TECS** (energy rates over g·V): throttle = trim + (W/T per throttle)·(demand + P + I on the total rate); pitch = α_trim/cos φ + balance demand + P + I. The acceleration demand is limited to 80 % of the thrust headroom between the trim throttle and its limits: a speed-up the throttle cannot deliver otherwise noses the aircraft over (the C172 dove from 38 to 48 m/s at spawn).
  - **Guidance**: course rate to bank `atan(−χ̇·V_g/g)`, course and altitude P loops (0.5 and 0.4 1/s), L1 (Park; `L1 = ζ·T·V/π`, T = 20 s, ζ = 0.75) on lines and circles (`Path`).
  - **Action modes** (`FixedWingActionMode`, default `attitude`): `surfaces` (4: aileron, elevator, rudder, throttle), `rates` (4: p, q, r, throttle), `attitude` (3: roll, pitch, airspeed), `guidance` (3: course rate, climb rate, airspeed). The airspeed range is [max(1.3 V_s, 1.1 × lowest trim), 0.95 × top speed] (the Aerosonde runs out of elevator at 1.3 V_s); the climb limit defaults to 0.7 of the steepest climb.
  - **Configuration**: `FixedWingConfig` (tuning, limits, design density and airspeed) and `FixedWingActionLimits` are new `GroupSpec` fields `fixed_wing_controller` / `fixed_wing_action_limits` (omitted when default, so `/meta` and goldens are unchanged; rejected for other families).
  - **Stall angles**: a lift peak the lift does not fall from is no stall. JSBSim's C172 table ends at −0.09 rad and is held flat, which raised `STALL` in light turbulence at cruise; its negative stall is now the search limit (−0.6).
  - **Tests** (`control/tests/fixedwing.rs`, both presets):
    - rate steps at 1.3 V_s, design and 0.9 × top speed: rise < 1.5 × 2.2/K_p, overshoot < 20 %;
    - bank 30° and pitch +5° steps: rise < 1.5 × 2.2/k_att, overshoot < 10 % (bank) and 20 % (pitch);
    - a 30° coordinated turn for 30 s: β ≤ 0.005 rad, height within 0.2 m, speed within 0.06 m/s;
    - altitude, airspeed and course hold for 90 s in W20 = 15 turbulence over a 6 m/s crosswind: Aerosonde RMS 1.8 m, max 4.2 m (bound 10 m); C172 RMS 5.3 m, max 14.2 m (bound 20 m: over seeds 1–4 and 7 its worst is 15.6 m, a slow, lightly loaded wing in moderate vertical gusts);
    - L1 on a line entered 100 m off and 40° off course, and on a circle of three minimum turn radii, in a (−2, 4) m/s wind: worst cross-track ≤ 1.9 m after settling (bound 5 m).
  - `sim/tests/fixedwing.rs::guidance_mode_holds_course_and_altitude`: a zero `guidance` action through the scenario keeps the ground track (within 0.1 rad) and height (within 10 m after the 20 s speed-up) in light turbulence over a crosswind. The other sim and Python fixed-wing tests now set `action_mode = "surfaces"`.
- **Step 7: viewer** (`scene::props::fixed_wing`, `viewer/src/{vehicle_view,sim,hud,camera,replay,main}.rs`):
  - **Visuals** from the definition (the presets carry no shape): a fuselage from nose to tail (the frame colliders, and a tractor propeller at the nose) as a cabin hull tapering to a boom, a canopy at the pilot's eye, the wing about the aerodynamic reference (its quarter chord there) with red tips, a stabiliser and fin at the tail, a spinner, and the wheels on struts at the gear contact points. Ailerons (outer 45 %), flaps, elevator and rudder are separate meshes behind their hinge lines (`SurfaceVisual`: hinge, axis, control), turned by the simulated deflection in pilot sense (`surfaces × control_signs`, so positive rolls right, pitches up and yaws right whatever the model's convention). The propeller disc's opacity follows the rotor speed over `FixedWing::full_throttle_speed` (new).
  - **Keyboard flight** (`Sim::flight_setpoint`; aircraft start in `attitude`, M cycles): `attitude` banks with A/D (to the group's roll limit) and pitches with W/S (forward nose down, about the level-trim α at the airspeed setpoint), Space/Shift move the airspeed setpoint (3 m/s per second, within the action map's airspeed range); `guidance` turns with A/D (course rate), climbs with Space/Shift, W/S set the airspeed; `rates` flies body rates (Q/E yaw) with the throttle on Space/Shift. Scales come from the group's `fixed_wing_action_limits` through an `attitude` `FixedWingActionMap` (new accessor `limits()`). `--vehicle aerosonde_like` (or any aircraft) spawns 150 m up at 1.5 V_s; `--demo` flies aircraft in `guidance` at the demo speed, turn rate and height.
  - **HUD**: airspeed (and the keys' setpoint), α (red past 85 % of a stall angle) and β, a STALL warning, throttle and propeller rpm, the four surface deflections, and an artificial horizon (sky and ground split by the horizon rotated by the bank, moved by the pitch, a ladder every 10°, the aircraft symbol); aircraft keys under F1.
  - **Chase camera** for aircraft (`CameraRig::aircraft`): 2.2 spans behind, swinging behind the ground track (0.3 s) instead of the heading and tilting with the smoothed climb angle; first person from the cockpit eye point.
  - **Replay**: `FixedWing::show` (new) puts the recorded deflections, throttle, propeller speed and air data on the placed aircraft, so the surfaces move and the HUD reads as recorded.
  - **Fix**: `attitude` names both a multirotor and an aircraft mode, and the name parsed as the multirotor's, so a recorded fixed-wing scenario (its mode written out) did not compile again. Scenarios now resolve a group's mode name by its vehicle's family (`AgentActionMode::resolve`).
  - **Frame rate**: the Aerosonde's demo over the 16 km `large` map at 25 m/s and 150 m AGL, 1920×1080 medium, no vsync: 83 fps after warm-up (116 over the last frames) on the Iris Xe; the C172 on a training map at 1600×900: 222 fps.
  - **Tests**: `scene` `fixed_wing_visual_matches_its_definition` (span, nose to tail, surfaces per control trailing their hinges, opposite ailerons); viewer `keys_fly_an_aircraft` (hands-off straight and level, A banks to the limit and turns left, wings level when released, Space raises the airspeed setpoint at the key rate and the aircraft follows within 1 m/s, `guidance` climbs, `rates` closes the throttle) and `a_recorded_flight_plays_back` (the replay shows the recorded aileron, throttle, airspeed and bank); sim `attitude_resolves_by_family`.
- **Step 8: `FixedWingWaypoints-v0`** (`tasks/fixed_wing_waypoints.py`):
  - **Task**: `aerosonde_like` in `attitude` on the 16 km `large` map (`map="large"`, a new shortcut; `margin` 1000 m), trimmed at 150 m AGL; 3 waypoints 1–3 km apart at 100–200 m AGL, radius 50 m; mean wind 0–6 m/s, turbulence W20 0–7.7, gusts; 10 Hz policy, 600 s. Observation (46): goal in the heading frame, AGL, air data, body velocity and rates, pitch and roll, last action and a terrain fan (`lidar`: rings −30°, −15°, −6°, 0° × 7 azimuths over 90°, 600 m). Reward: horizontal progress (per 100 m), 10 per waypoint, `−0.5(1 − h/50)²` below 50 m AGL, −0.5 per stalled step, smoothness 0.05, −50 on a crash.
  - **Goal grade** (`goals.grade`, new, aerial `random` goals): draws whose height differs from the previous point by more than `grade` per horizontal metre rank below all others (the task uses 0.08, below the Aerosonde's steepest climb of about 0.125).
  - **Scripted pilot** (`scripted(obs)`): banks 1.5 × the goal bearing; flight-path angle from the goal's height difference or the steepest gradient that keeps 60 m over the fan's terrain points within 400 m and nearer than the goal, within ±0.12; pitch = that + α. Terrain within 300 m that needs more than 0.12 turns it away at 80 % bank towards the more open half of the fan, holding the direction (read from `last_action`); a goal behind within 250 m is left behind straight first. It finishes 55 of 59 episodes on the 16 km map (2 terrain crashes, 2 truncated). Pitfalls met on the way: `pitch_roll` pitch is positive nose down; terrain beyond the goal, or far ahead, kept it climbing past goals in valleys.
  - **Tiled maps under many fast agents** (the task first ran at 22 env-steps/s):
    - Rays use the tiles up to 300 m (`DETAIL_RANGE`, was 2 km) and the coarse layer beyond: a 600 m fan had kept 9+ tiles per aircraft in use and thrashed the 128-tile cache.
    - `BatchSim` reserves 8 tiles per agent in each tiled map's cache (`TiledMap::reserve`; `TILES_PER_AGENT`).
    - `TiledMap::prefetch` generates tiles on background threads (half the cores, at most 8); `WorldInstance` prefetches, every policy step, the tiles within 150 m of each agent's position 3, 6 and 10 s ahead. A batch step no longer waits for the one world whose aircraft enters a new tile (about 40 ms per tile). The cache takes a read-write lock and evicts only on insertion; threads remember their last 8 tiles.
    - Result, 16 aircraft: 1100–1800 env-steps/s with the fan on the 16 km map, limited by tile generation (about 40 ms per 256 m tile; without trees and rocks about twice as fast). On a 4 km version of the preset (`config = {size = 4096}`, 256 tiles that all stay cached) training runs at 9–11k SPS. Physics alone (flat world) is about 90k env-steps/s.
  - **Short training** (`ppo_continuous.py`, 32 worlds × 128 steps, 2M steps, 4 km map with `margin=500`, 3.4 min): the return rises from −190 to −9 and the episodes from 300 to 5000 steps (it learns to stay airborne); success 12.5 % at the end (0.56 waypoints per episode). Not tuned further (the framework's job ends at a pipeline that trains).
  - **Export and viewer**: `policy.json` flies in `viewer policy` on unseen map seed 1000 (87 fps, LiDAR panel with the fan); `eval_record.py` records two episodes that re-simulate bit for bit, and `viewer replay` plays them (89 fps).
  - **Tests**: `world` `prefetched_tiles_are_ready_and_the_same`; `sim/tests/large.rs::aircraft_waypoints_keep_their_grade` (and `grade` rejected for ground vehicles); Python: the generic env tests on a 4 km tiled map, `test_fixed_wing_waypoints_scripted_pilot_reaches_the_goal` (8 aircraft, one waypoint 0.8–1.2 km away: at least 6 reach it) and `test_fixed_wing_waypoints_rewards`.

### M6b: Helicopter

#### Design
- **Rotor model** (`vehicles::rotorcraft::rotor`), shared by the helicopter's main and tail rotors and the tiltrotor's proprotors:
  - **Aerodynamics**: blade-element theory integrated in closed form over a rigid blade (linear twist, lift slope, profile drag, root cut-out, tip loss), with inflow from momentum theory in forward flight (Glauert). Iteration over inflow and thrust gives thrust, torque and power, and the H- and Y-forces.
  - **Flapping**: the tip-path plane follows first-order flapping dynamics `a₁, b₁` with time constant `16/(γΩ)`, from cyclic, rates (gyroscopic and aerodynamic cross-coupling) and advance ratio. A hinge offset or spring gives a hub moment.
  - **Ground effect** and a vortex-ring-state warning (not modelled beyond the momentum-theory limit).
  - **Rotor speed**: a state driven by the engine and governor against the rotor torque, so autorotation is possible.
  - **Optional**: Pitt–Peters dynamic inflow (three states), if the quasi-steady inflow misses the validation.
- **Helicopter** (`type = "helicopter"`, `Family::Rotorcraft`):
  - **Components**: main rotor, tail rotor geared to it, swashplate servos (collective, longitudinal and lateral cyclic; tail collective), fuselage drag areas, horizontal and vertical fins (`AeroSurface`), engine with governor and torque limit, and skids as gear colliders.
  - **Presets**:
    - `xcell60_like`: an 8.2 kg RC helicopter, from Gavrilets, Mettler & Feron's published model.
    - `bo105_like`: a light twin, from Padfield's *Helicopter Flight Dynamics* configuration data.
- **Validation**:
  - **Hover**: power against momentum theory with a figure of merit; thrust and torque coefficients against blade-element formulas.
  - **Bo105 (Padfield)**: the power-versus-speed bucket, trim collective and attitudes, and selected stability derivatives, within about 10–15 %.
  - **X-Cell**: flapping time constants and hover trim.
  - **Behaviour**: autorotation keeps rotor speed with a plausible descent rate.
- **Control**:
  - **Action modes**:
    - `raw`: collective, cyclic, pedal.
    - `rates`: collective and body rates, the counterpart of `ctbr`.
    - `attitude`.
    - `velocity`: horizontal velocity, vertical velocity and yaw rate, as for multirotors.
  - **Gains**: from a numerical linearisation in hover and forward flight, scheduled on airspeed.
- **Viewer**: fuselage, a rotor disc tilted by the flapping, tail rotor, skids; HUD with rotor rpm, collective, torque and power; keyboard flight; replay.
- **Demo**, **`HeliLandingZone-v0`**: the X-Cell (or Bo105) flies from forward flight over a large wild map to a landing zone (flat open ground in a valley or on a ridge) and lands. Success requires a touchdown sink rate and attitude limit, in wind and turbulence.

#### Implementation order
| # | Step | Done when |
|---|---|---|
| 1 ✅ | Rotor model: BEMT with forward-flight inflow, flapping, hub moments, ground effect, rotor speed | Hover and forward-flight thrust, torque and flapping match closed-form blade-element and momentum results; flapping lag matches `16/(γΩ)` |
| 2 ✅ | Helicopter family and presets; engine and governor; tail rotor; fins; skids; wiring | Both presets trim in hover and forward flight; stand on their skids; rotor speed recovers from load steps |
| 3 ✅ | Validation against Padfield (Bo105) and Gavrilets (X-Cell) | Hover power, power curve, trim controls and attitudes within the tolerances above |
| 4 ✅ | Control and action modes | Attitude and velocity steps settle in hover and at 20 m/s; hover holds position in wind |
| 5 ✅ | Viewer: visuals, HUD, keyboard flight, replay | The helicopter flies by keyboard and lands on a large map; recordings replay |
| 6 ✅ | `HeliLandingZone-v0`: task, scripted pilot, short training, export, viewer, replay | As for the other demos |

#### As built
- **Step 1 (rotor model)**, `vehicles::rotorcraft::{Rotor, RotorDef, RotorState, RotorInput, RotorLoads, Spin}`:
  - **Frame**: the shaft frame (z along positive thrust, x the disc reference, azimuth ψ counter-clockwise from −x). A clockwise rotor mirrors y internally (vectors flip y, pseudovectors x and z), so cyclic, flapping, forces and moments keep geometric meanings for either direction.
  - **Blade element**: rigid blades with linear twist (collective at ¾R), root cut-out, tip-loss factor B (lift only), `δ = δ₀ + δ₂C_T²`; small-angle section forces with u_T, u_P including in-plane velocity in any direction, flapping (β′ and the radial-flow term) and shaft pitch/roll rates; yaw rate changes the effective Ω. The integrands are polynomials (degree ≤ 4 in r̄, ≤ 5 in ψ), so 3 Gauss–Legendre points × 8 azimuths evaluate the closed-form results exactly; no reverse flow or blade stall (μ ≲ 0.5).
  - **Inflow**: C_T is affine in λ and independent of flapping; Glauert's uniform momentum inflow through the tip-path plane (`λ_c = μ_z + μ_xβ₁c + μ_yβ₁s`) is solved by Newton from the warm start in `RotorState::inflow`, with bisection on the working-state bracket as the fallback. Ground effect scales the blade-seen inflow by Cheeseman–Bennett `k_G = 1 − (R/4z)²/(1 + (μ/λ)²)` (z clamped ≥ R/2). A `vortex_ring` flag is raised for μ < λ_h and 0.28 < descent/λ_h < 2; the normal working state is kept there.
  - **Flapping**: centre-spring flap equation (`ν² = 1 + K_β/(I_βΩ²)`, `γ = ρacR⁴/I_β` at the local density, gyroscopic `2(p̂ cos ψ + q̂ sin ψ)`), harmonically balanced as a 3×3 system for steady coning and tilt; coning is quasi-steady, the tilt lags with `τ = 16/(γΩ)` (exact exponential over a step), clamped to `flap_limit`. Hub moment `(N_b/2)K_β(−β₁s, β₁c, 0)`. The aerodynamic torque acts on the rotor: `I_Ω·Ω̇ = Q_drive − Q` in `Rotor::advance`; `angular_momentum` for gyroscopics.
  - **Tests** (`crates/vehicles/tests/rotor.rs`): hover λ, C_T, C_Q = λC_T + σδ/8, coning against the closed forms (1e-10); forward flight C_T, Glauert inflow, β₀/β₁c/β₁s against Johnson's centrally hinged formulas (1e-10); a brute-force vector blade-element integration with numerically solved harmonic flapping for both spin directions, cut-out, tip loss, δ₂, spring, rates, sideslip and ground effect (1e-6); rate lag `−τq`, `+τp` with the spin-dependent cross-coupling; cyclic step reaches 63 % at `16/(γΩ)`; ground effect `λ_i = (15/16)√(C_T/2)` at z = R; rotor speed torque balance; windmilling (negative torque) in a descent; vortex-ring flag.
- **Step 2 (helicopter family)**, `vehicles::rotorcraft::{HelicopterDef, Helicopter, HelicopterInput, HelicopterInit, HelicopterTrim}`, `Family::Rotorcraft` (`"rotorcraft"`), `VehicleDef`/`SharedDef`/`Vehicle::Helicopter`:
  - **Definition** (`type = "helicopter"`): rigid body (`fixedwing::AirframeDef`), main and tail `RotorMount { hub, axis, rotor }` (axes normalised), `tail_gear_ratio`, fuselage drag areas per body axis, fin/tailplane `AeroSurface`s (in the free stream, no rotor downwash yet), `PitchChannel`s (`min`/`max` blade pitch, servo rate and lag) for collective, longitudinal and lateral cyclic and pedal, `EngineDef` (max power, rated speed, max torque default 1.2 P/Ω, lag, governor bandwidth), sphere colliders (`gear` = skids), contact constants.
  - **Controls**: inputs in [−1, 1] map linearly to blade pitch; swashplate phasing `θ₁c = −s·lateral`, `θ₁s = −s·longitudinal` (s the main rotor's spin sign) so stick forward/right tilts the disc forward/right for either direction; the pedal sign follows the tail rotor's moment arm so pedal right always yaws right.
  - **Drive train**: one rotor speed Ω (tail at nΩ); `I_d Ω̇ = Q_engine − Q_main − nQ_tail` with `I_d = I_m + n²I_t`; PI governor (`kp = 1.4ω_gI_d`, `ki = ω_g²I_d`, anti-windup), first-order engine lag, limit `min(Q_max, P_max/max(Ω, 0.2Ω_r))`, freewheel (no negative torque); engine can be stopped. Drive reactions `−(Q + IΩ̇)` about each shaft act on the airframe.
  - **Gyroscopics**: only the non-flapping share of each rotor's polar inertia (`RotorDef::hub_inertia` = polar − N_b·I_β, zero by default) enters the rigid-body gyroscopic term; the blades' gyroscopic reaction reaches the hub through the flapping equation and hub spring (counting it twice made the skid contacts ring).
  - **Trim** (`Helicopter::trim(speed, ρ, g)`): Newton with a finite-difference Jacobian over the four inputs (unclamped, linear pitch map), pitch and roll, with settled rotors (`Rotor::settled`: fixed point of flapping and inflow) at the governed speed; residual forces/weight and moments/(weight·R) < 1e-10; error if any input would exceed ±1.
  - **Presets**: `bo105_like` (Padfield's Bo105 data: 2200 kg, R 4.91 m, hingeless K_β 113 330 N·m/rad, 3° shaft tilt, tail rotor gear 5.25) and `xcell60_like` (Gavrilets' 8.2 kg X-Cell: R 0.775 m, 1600 rpm, K_β 54 N·m/rad, effective I_β for the flybar's ~0.1 s flap lag). Both hover at ~4° roll toward the tail-rotor side; Bo105 hover ≈ 321 kW, 30 m/s ≈ 185 kW, 60 m/s ≈ 332 kW; X-Cell hover ≈ 1.1 kW. Skid contact effective mass per collider from the roll inertia over the skid lever arm (60 kg and 0.25 kg), since a quarter of the mass per collider is too stiff in roll at 500 Hz.
  - **Wiring**: control `rotorcraft` module with the `sticks` action mode (the design's `raw`, a name ground vehicles already use; 4 inputs, pass-through controller; the controlled modes come in step 4); sim spawns helicopters trimmed in the air (hover, or `spawn.airspeed` for forward flight, at the spawn altitude's density) or on the skids (collective down, rotor governed); `Command::hold` keeps the reset input; the recorder writes controls, blade pitches, rotor speed, engine power, flapping, coning and airspeed; state hash includes rotor speed, engine torque, pitches, flapping and inflow; the viewer has a placeholder visual (cabin sphere and rotor discs) until step 5.
  - **Tests**: `crates/vehicles/tests/helicopter.rs` (presets round-trip; trims at 0/30/60 m/s (Bo105) and 0/10 m/s (X-Cell) within the control range with hover power within 15 % of momentum theory with tip loss plus profile power and the Bo105 power bucket; zero accelerations released at trim (< 1e-3); standing on the skids for 5 s with only skid contacts; rotor speed droop and recovery (< 0.2 %) after a +0.3 collective step; engine-off decay), `crates/sim/tests/helicopter.rs` (trimmed air spawns hold for 1 s; stands on the skids with `LANDED`, lifts off with collective; recording round trip).
- **Step 3 (validation)**:
  - **Trim API**: `trim` (level), `trim_climb(speed, climb_rate, …)` and `trim_autorotation(speed, …)` (engine off at the governed rotor speed: the seventh unknown is the climb rate, the seventh residual the net rotor torque; seeded from a powered descent at the level power over weight) share an N-dimensional damped Newton solver. `HelicopterTrim` carries the climb rate, density and gravity.
  - **Linearisation**: `Helicopter::linearize(&trim) -> HelicopterLinear { a: 8×8, b: 8×4 }` over `[u, v, w, p, q, r, φ, θ]` and the normalised inputs, by central differences of `trim_derivative` (rigid-body equations with quasi-steady settled rotors at the trim rotor speed); step 4 builds its gains from it.
  - **Results**: Bo105 Lock number 5.07 (Padfield 5.087), λ_β² 1.248; hover figure of merit ≈ 0.7; power 321 kW hover, minimum ≈ 185 kW near 30 m/s (≈ 58 % of hover); nose-down and forward stick grow with speed; autorotation descent at 30 m/s 8.5 m/s with collective ≈ −0.62, descent power within 10 % of level power at 20–50 m/s. X-Cell hover torque 6.29 N·m (Gavrilets ≈ 6.3), effective flapping time constant 0.10 s (0.1), pitch ≈ −9° at 14.5 m/s (−10°). Hover damping `L_p`, `M_q` within 15 % of `−(N_bK_β/2 + T h)·16/(γΩ)`.
  - **Preset change**: X-Cell collective range widened to −0.18…0.30 rad (Gavrilets' ±0.183 rad travel) so it can autorotate. At the governed speed it does so only in steep descent (windmill brake state near hover): Gavrilets' δ₀ = 0.024 costs ≈ 700 W of profile power, and at 12 m/s the shaft power bottoms out near 130 W at about 17 m/s descent.
  - **Rest input**: `HelicopterInit::at_rest(def, pose)` rests with the collective at flat (zero) pitch rather than full down, which with the wider range pressed the X-Cell into its skids.
  - **Not checked**: Padfield's Bo105 trim charts and derivative tables (Appendix 4B.3) were not available online; the Bo105 checks use his configuration data, first principles and published flight-manual magnitudes instead.
  - **Tests**: `crates/vehicles/tests/helicopter_validation.rs` (Bo105 rotor data; power curve and trim trends; autorotation energy balance for Bo105 at 20–50 m/s and X-Cell in vertical descent; X-Cell against Gavrilets; hover damping against the flapping lag; linear-model signs and a perturbation check against the nonlinear derivative within 2 %).
- **Step 4 (control and action modes)**, `control::rotorcraft::{HelicopterController, HelicopterConfig, HelicopterActionMap, HelicopterActionLimits, HelicopterSetpoint}`:
  - **Modes**: `sticks` (pass-through), `rates` (body rates and collective), `attitude` (roll, pitch, yaw rate and collective), `velocity` (heading-frame velocity and yaw rate; the default). `rates`, `attitude` and `velocity` share names with the other families and resolve per group. Internal `Position { position, yaw }` setpoint for scripted hover. `velocity` speeds default to 80 % of the fastest trim forward (Bo105 56 m/s, X-Cell 20 m/s), twice the hover induced velocity sideways and backward, and half of it vertically.
  - **Gain schedule**: at construction the controller trims and linearises the helicopter every half hover induced velocity up to the fastest trim (X-Cell 25 m/s) and interpolates on the body forward airspeed: trim inputs and attitude, the rate, airspeed, cyclic/pedal and collective columns of the angular-acceleration model, and the heave derivatives.
  - **Rate loop**: dynamic inversion of the linear model (cancels the rate, airspeed and collective terms) with PI correction (integral k²/4, frozen at saturation). Bandwidth `0.5/(α·τ_flap + τ_servo)` (Bo105 7.4, X-Cell 11 rad/s).
  - **Flapping lag**: the quasi-steady model omits the tip-path plane's lag (X-Cell flybar 0.1 s). That lag destabilised climbs and accelerations in forward flight on the X-Cell. A lead `(τs + 1)/(ατs + 1)` (α = 0.25) on the demanded roll and pitch acceleration fixes it; the cancellation terms go without, since the rotor's own responses lag alike.
  - **Outer loops** (each 4× slower): the attitude loop holds the heading that the yaw-rate command integrates to, which a rate loop alone cannot do where the fin is directionally unstable (sideways and backward flight). The velocity loop tilts from the trim attitude (`atan(a/g)`, 25° at most) and drives the collective through the heave model with integral action. Trim attitude and collective are scheduled on the *reference* airspeed: on the measured one the X-Cell's steep trim-pitch slope near 20 m/s fed speed back positively. The position loop feeds the velocity loop.
  - **Wiring**: `Controller::new` and `ActionMapping::new` take the helicopter config and limits; groups have `helicopter_controller` and `helicopter_action_limits` fields, foreign to other families.
  - **Tests**: `crates/control/tests/helicopter.rs`:
    - bandwidth ordering;
    - ±10° roll and pitch steps from hover and 20 m/s, within 15 % after 4/k_att, both presets;
    - 2 m/s velocity steps on all axes from hover and 20 m/s, within 0.2 m/s after 15 s, rotor speed within 5 %;
    - hover position hold within 0.3 m in 5 m/s (Bo105) or 3 m/s (X-Cell) wind from three directions;
    - action maps.
    - `crates/sim/tests/helicopter.rs` adds `velocity` mode flying forward and stopping.
    - Full-scale single-axis velocity actions track on both presets; the Bo105's rotor droops 15 % while accelerating to 56 m/s (engine power limit).
- **Step 5 (viewer)**:
  - **Visual** (`scene::props::helicopter(def) -> HelicopterVisual { body, rotors: [RotorVisual; 2], span, eye }`), derived from the definition: a cabin hull over the forward frame colliders with a canopy, an engine cowling and the mast; a boom tapering to the tail rotor; fin and tailplane plates from the `AeroSurface`s; skids along the gear colliders with cross tubes; one blade mesh per rotor.
  - **Rotor heads**: a head per rotor at the hub in the shaft frame, tilted with the simulated tip-path plane (`rotor_tilt([β₁c, β₁s])`) and turned at the rotor speed (at most 9 rad/s shown, against aliasing), with the blades coned up (`blade_rotation`) and a translucent disc whose opacity follows the rotor speed.
  - **Keyboard flight** (`Sim::heli_setpoint`, `heli_map`): `velocity` through the group's action map; `attitude` tilts from the trim attitude at the airspeed; `rates` flies body rates; both with the trim collective ± 0.25 on Space/Shift. The viewer spawns helicopters hovering a metre above their skids in `velocity`; `--demo` flies them in `velocity`.
  - **HUD**: rotor speed in per cent (LOW ROTOR below 90 %), engine power against its limit, airspeed and climb, the four inputs, the disc tilt, an artificial horizon; its own key help. The chase camera follows the heading (not the flight path), and the first-person camera sits behind the canopy.
  - **Replay**: `Helicopter::show(&HelicopterDisplay)` applies the recorded inputs, pitches, rotor speed, engine power, flapping, coning and airspeed.
  - **Ground logic** (controller): the first keyboard landings rolled the X-Cell over on lift-off. The loops wound up against the skids, holding the lateral cyclic at 0.65, and then rolled it over as a skid unloaded (dynamic rollover). While any contact touches, the rate and horizontal velocity integrators are cleared and the attitude loop holds the attitude as it stands. The vertical integrator keeps the helicopter down after a descent until the pilot climbs.
  - **Tests**: `props` (visual covers nose, tail rotor and skids; blade span; shaft frames; tilt and coning signs); viewer `keys_fly_a_helicopter` (hover in place, half-stick forward speed and stop, attitude bank and return to trim, collective climb, yaw in rates); `keys_land_a_helicopter` (both presets descend at 1 m/s, `LANDED` upright and still, lift off level); `rotor_heads_follow_the_flapping`; `replayed_helicopters_show_their_rotors`.
- **Step 6: `HeliLandingZone-v0`** (`tasks/heli_landing_zone.py`):
  - **Task**: `xcell60_like` in `velocity` (speeds 15, 6 and 1.5 m/s, yaw rate 0.5 rad/s: a full-down touchdown stays below the 2 m/s crash sink rate), spawned trimmed in level flight at 4–10 m/s and 35–50 m AGL on the `offroad` 512 m maps (pool of 16); one landing zone 100–300 m away; mean wind 0–3 m/s, turbulence W20 0–3; 20 Hz policy, 120 s. Success: `LANDED` within 2 m of the zone's centre (ends the episode); landing elsewhere is allowed (it can lift off again). Observation (19): goal in the heading frame (1/100), AGL, body velocity and rates, pitch and roll, last action, and the goal again at 1/10 clipped to ±1 for the last metres. Reward: decrease of `φ = √(d² + h²)` per 10 m, −0.005 per step, smoothness 0.02, +20 on success, −20 on a crash.
  - **Landing zones** (`goals.landing_slope`, new, aerial `random` goals): a draw scores the share of `clearance` free of obstacles and foliage around the vertical from 1 to 40 m above the ground (`LANDING_COLUMN`; canopies overhang clearings), less 1.5 when the ground within `clearance` (16 rim points and 8 at half radius) rises or falls more than `landing_slope` per metre, and less 2 over water. The offroad preset gives clean zones every time (clearance 3 m, slope 0.1). The training forest fell back to the best draw 11 times in 24, and the large preset's relief failed on slope, so the task uses offroad rather than the large map the plan named.
  - **Settling on the ground** (controller): after touchdown the velocity loop's collective still carried most of the weight (the vertical integrator stops at 0.3 g). The X-Cell skated backward on its skids and rolled over in 3 of 16 scripted landings. On the ground with no climb asked for, the collective now ramps over `SETTLE_TIME` (1 s) to the trim model's zero-thrust collective; any climb demand restores it at once.
  - **Scripted pilot** (`scripted(obs)`): yaws toward the zone beyond 10 m. Flies at the speed from which it can stop at 0.8 m/s² (at most full forward; 0.5 × distance close in), holding its height. Within 1.5 m of the centre it descends at 1.2 m/s, and at 0.5 m/s below 4 m AGL. It lands 16 of 16 episodes within 0.25 m of the centre, in 55–77 s.
  - **Short training** (`ppo_continuous.py --hidden 256 --bound-coef 0.01 --gamma 0.995`, 64 worlds × 128 steps, 5M steps, 9 min at 9.7k SPS; basic training only):
    - The return rises from −32 to about −1, and deterministic evaluation on unseen maps (`map_seed` 1000) survives 95 % of the episodes. The policy flies to the zone and descends over it, but hovers and circles 15–25 m above it without touching down: no landings in 5M steps.
    - Pitfalls on the way: a first potential `d + h·(1 − d/30)` rose while closing in high inside 30 m, so the policy parked outside the radius. A full-down vertical action of 2 m/s equalled the crash sink rate, so the policy avoided the ground.
    - Left to the user's training: a curriculum on spawn height and zone distance, or a longer run.
  - **Export and viewer**: `policy.json` flies in `viewer policy` on map seed 1000 (167 fps). `eval_record.py` records two episodes that re-simulate bit for bit, and `viewer replay` plays them (181–192 fps).
  - **Tests**: `sim/tests/helicopter.rs::landing_zones_are_flat_dry_and_open` (24 zones on offroad maps: slope, water, obstacles up to 40 m; ground vehicles and zero slopes rejected); Python: the generic env tests, `test_heli_landing_zone_scripted_pilot_lands` (8 of 8 within 2 m) and `test_heli_landing_zone_rewards`.

### M6c: Tiltrotor

#### Design
- **Layout** (proposed): a **quad tiltrotor UAV**, like PX4's tiltrotor airframes, with a wing, a tail with control surfaces and four fixed-pitch rotors on tilting mounts.
  - In hover it is controlled like a multirotor, with differential thrust plus differential tilt for yaw. In cruise it is a fixed-wing with the rotors forward.
  - A twin proprotor layout in the manner of the V-22 would need cyclic pitch on the proprotors from M6b and is harder to control; it stays an option.
- **Model**:
  - Each rotor mount is an actuated revolute (tilt servo with rate limit and lag). Tilt reaction and gyroscopic moments of the spinning rotors act on the airframe.
  - The rotors use M6b's rotor model (fixed pitch, oblique and edgewise inflow in transition).
  - The wing and tail use the fixed-wing surfaces, with an optional simple slipstream factor over the wing behind the rotors.
- **Control**:
  - **Transition**: a transition scheduler gives the tilt as a function of airspeed within the conversion corridor. The control allocation blends multirotor thrust and tilt allocation with the aerodynamic surfaces, weighted by dynamic pressure.
  - **Action modes**:
    - `velocity`: transitions automatically with the commanded speed.
    - `attitude` with airspeed.
    - `raw`: rotor throttles, tilts and surfaces.
- **Validation**:
  - Hover trim like a multirotor (power against momentum theory).
  - Cruise trim like a fixed-wing (against a component build-up of the same wing).
  - The conversion corridor (feasible tilt against airspeed) from trim sweeps, with a plausible shape.
  - A scripted transition in both directions within an altitude band and without stall.
  - Energy bookkeeping.
- **Viewer**: visuals with tilting nacelles; HUD with tilt, transition state and corridor; keyboard flight; replay.
- **Demo**, **`TiltrotorDelivery-v0`**: the tiltrotor takes off vertically from a pad on a large rural map, transitions, cruises 2–4 km to a farm yard, transitions back and lands on the yard's pad.

#### Implementation order
| # | Step | Done when |
|---|---|---|
| 1 ✅ | Tilting rotor mounts, tiltrotor preset, trims | Hover and cruise trims match their references; the corridor is computed; energy balance holds |
| 2 ✅ | Transition control and action modes | Scripted transitions both ways hold altitude within bounds without stall; hover and cruise steps settle |
| 3 ✅ | Viewer: visuals, HUD, keyboard flight, replay | It flies by keyboard through both transitions; recordings replay |
| 4 ✅ | `TiltrotorDelivery-v0`: task, scripted pilot, short training, export, viewer, replay | As for the other demos |

#### As built
- **Step 1 (tilting mounts, preset, trims)**, `vehicles::tiltrotor::{TiltrotorDef, TiltMount, TiltrotorControlsDef, SurfaceMix, Tiltrotor, TiltrotorInput, TiltrotorInit, TiltrotorLoads, TiltrotorTrim, TrimLimits, CorridorPoint}`, `Family::Tiltrotor` (`"tiltrotor"`), `VehicleDef`/`SharedDef`/`Vehicle::Tiltrotor`:
  - **Definition** (`type = "tiltrotor"`): rigid body (`AirframeDef`), fuselage drag areas, `AeroSurface`s, aileron/elevator/rudder `PitchChannel`s mixed onto the flapped surfaces by `SurfaceMix` gains, a `tilt` channel (range and servo of the mounts), 1–4 `TiltMount { pivot, offset, sense, tilting, tilt }` (axis `(sin τ, 0, cos τ)`, hub `pivot + offset·axis`), one propeller and electric motor shared by the rotors, `rotor_drag`, colliders. `AeroSurface` gained an optional `aspect_ratio` (the wing panels are halves of one wing, so span²/area would halve it; serde default, goldens unchanged).
  - **Rotor model, deviation from the design**: the rotors reuse the fixed-wing `Propulsion` (C_T(J), C_Q(J) polynomials on an electric motor, axial inflow `flow(hub)·axis`) rather than M6b's BEMT rotor, which models articulated, cyclic-pitched blades. Edgewise flow adds the in-plane drag `−K_d·ω·v_⊥` (as the multirotor); positive thrust gets Cheeseman–Bennett ground effect. Airframe moments: motor reaction `−axis·sense·Q`, tilt gyroscopic `−sense·I·ω·dAxis/dt`, and the rotors' angular momentum in the rigid-body step (`semi_implicit_euler_with_momentum`). No slipstream over the wing yet.
  - **Preset** `quadtilt_like` (6 kg quad tiltrotor): wing 0.6 m² (two panels, AR 8.07, 1.72° incidence, flaperons), tailplane with elevator, fin with rudder at 1 m behind; four 0.38 m propellers (Aerosonde polynomials) on 400 kV motors at 22.2 V, pivots at ±0.45 m, ±0.6 m, tilt −0.15…π/2 at 1.2 rad/s. Hover throttle 0.58 (616 W shaft, 715 W electric); 20 m/s cruise with the rotors forward at 0.58 throttle, 7.9 N thrust, 277 W; stall ≈ 11.5 m/s.
  - **Trim** (`Tiltrotor::trim(speed, tilt, ρ, g)`, `trim_near`): unknowns pitch, front and rear rotor speed (relative to hover) and elevator; residuals the body-frame force balance with gravity, the pitching moment, and a least-effort condition `(ω_f − ω_r)·∂M/∂e − e·∂M/∂Δω = 0` that picks the split between differential thrust and elevator (a tilt-blended condition failed between 32° and 57°). Throttles follow from the steady motor (`i = Q/k + i₀`, `throttle = (kω + iR)/V`). `feasible(def, limits)`: throttles in [0, 1], currents ≤ max, elevator ≤ 1, no stall, |pitch| ≤ 15°.
  - **Corridor** (`corridor(speeds, tilt_step, limits, ρ, g)`): a tilt sweep per speed, continued from the same speed's last *feasible* trim, then the previous speed's trim at the tilt, then a cold start (continuing from infeasible trims slid onto the stalled branch and left holes). Result: 0 m/s −9…14°, 10 m/s −9…40°, 12 m/s −8…90°, 20 m/s 1…90°, 30 m/s 11…90°, contiguous everywhere.
  - **Sim wiring**: air spawns start trimmed at `spawn.airspeed` (rotors forward from 1.2 × stall, else up, else the first feasible tilt), on the gear otherwise; the recorder writes rotor speeds (`motors`), `throttles`, `tilts`, the surface channels, electric power, airspeed, α and β; the state hash includes rotor speeds, tilts and channels. Control has the `raw` action mode now (2n + 3 components: throttles, tilts over the mount range, surfaces; pass-through) so the family runs end to end; the controlled modes come in step 2. The viewer has a placeholder box and holds the inputs until step 3.
  - **Tests**: `crates/vehicles/tests/tiltrotor.rs`: preset validates; hover trim against momentum theory (figure of merit from the polynomials exactly, 0.6–0.8; motor efficiency; the model holds still); cruise trims at 15/20/25 m/s against a component build-up (Helmbold lift slopes, polar, thin-aerofoil flaps, fuselage drag, rotor normal force): pitch within 0.02°, thrust 0.5 %, elevator 0.01; conversion corridor shape; energy balance within 1 % of ∫|P|; control signs; tilt servo and range; ground effect 16/15. `crates/sim/tests/tiltrotor.rs`: hover and 20 m/s air spawns hold for 1 s; stands on the gear and lifts off; recording round trip.
- **Step 2 (transition control, action modes)**, `control::tiltrotor::{TiltrotorController, TiltrotorConfig, TiltSchedule, SchedulePoint, TiltrotorSetpoint, TiltrotorActionMode, TiltrotorActionLimits, TiltrotorActionMap}`; `GroupSpec.tiltrotor_controller` / `tiltrotor_action_limits` (foreign to the other families and they to it); `Controller::new` / `ActionMapping::new` take the tiltrotor config and limits:
  - **Schedule** (`TiltSchedule`, built once per group at compile, shared by `Arc`): tilt against airspeed `τmax·smoothstep((V − 0.3V_s)/(0.9V_s))` clamped into the corridor (swept at 1 m/s, 2° up to 1.4·V_s) with a 0.05 rad margin from its edges (not from the mount's end stops), kept non-decreasing, with the level trim at each point; rotors forward beyond the sweep while the trim stays feasible. Preset: V_s 11.2 m/s, forward from ~14 m/s, fastest 34 m/s (action speed default 0.9× that).
  - **Model helpers** in `vehicles`: `rotor_load` / `aero_loads` (split out of `airframe_loads`), `rotor_speed_for(thrust)`, `throttle_for(ω, ω_target, lag)` (steady motor plus the spin-up torque), `max_thrust(v_axial)`, `rotor_momentum_body`.
  - **Velocity loop** (heading frame): acceleration-limited reference ([2, 1] m/s²) on a leash of accel/k_vel from the velocity, feedforward, PI (k_vel, k²/4; horizontal integral frozen while the thrust or pitch saturates, every axis while its reference ramps). Mount tilt from the schedule at the airspeed the reference leads to. Total thrust and pitch by a 2-D Newton solve on the model force (rotor thrusts shared as now along the current axes, their in-plane forces, `aero_loads` at the rotated flow); pitch within ±15° and the wing kept 0.05 rad below the stall (when V > 0.5V_s); when the pitch saturates the thrust meets the vertical force first (weighted least squares), when the thrust saturates the pitch holds the vertical force. Roll from the lateral force less (in hover) the airframe's and rotors' side force; on the wing lateral commands turn instead (`a_y = V·r`) with sideslip fed back into the yaw rate (turn coordination) and a flight-path-rate feedforward `a_z/V` on the pitch (without it the steep lift slope and the attitude lag cut the realised climb acceleration about sixfold). Ground: integrators cleared on contact, thrust settled to zero over 1 s when no climb is asked for.
  - **Attitude loop**: Euler targets with heading hold (as the helicopter). **Rate loop**: INDI, `M_d = I·α_d + ω×(Iω + h_rotor)`, the increment over the modelled moment now allocated (`allocation.rs`: weighted least effort with bounds, worst violator fixed first, objectives dropped yaw → thrust → pitch → roll) over the rotor thrusts (moment per newton by finite difference of `rotor_load`, bounds from the full-throttle thrust table against axial speed), differential tilt (left/right mounts, ±0.3 rad within the travel) and aileron/elevator/rudder (finite differences of `aero_loads`), with the total thrust as the fourth objective. Thrusts → rotor speeds → throttles; surfaces centred below half the stall speed. Gains from the model: k_rate = 0.5/(motor lag + slowest servo τ) = 12.5, then ÷4 per loop (3.1, 0.78, 0.2).
  - **Action modes**: `velocity` (default; forward, sideways, vertical, yaw rate; transitions with the forward speed), `attitude` (roll, pitch, yaw rate, climb, airspeed 0…forward speed; the thrust along its axis holds the climb in hover and the airspeed on the wing), `raw`; an internal `Position` setpoint.
  - **Results** (preset, 500 Hz): hover velocity steps of 3 m/s forward/sideways and ±2 m/s vertical settle within 0.1 m/s in 2–3.5 s; at 20 m/s speed steps of ±4 m/s settle in 2–9.5 s, climb/descent of 1.5 m/s in 1.3 s, a 0.15 rad/s turn banks 17° with 0.0003 rad sideslip; hover → 20 m/s and back hold altitude within 1.6 m and 0.7 m, α ≤ 0.18 rad (limit 0.21); position step 10/5/5 m settles in 22 s. The controller costs about 8 µs per update against 1.1 µs for the physics step (≈20 finite-difference model evaluations; the Jacobians could be cached across steps if throughput matters).
  - **Tests**: `crates/control/tests/tiltrotor.rs` (schedule, hover and cruise velocity steps, coordinated turn, both transitions, position step, attitude mode incl. a transition by airspeed), allocation unit tests, action-map test; `crates/sim/tests/tiltrotor.rs` `velocity_mode_flies_a_circuit` (gear → climb → 20 m/s on the wing → hover → descend → settled on the gear, throttles idle).
- **Step 3 (viewer)**:
  - **Visuals** (`scene::props::tiltrotor` → `TiltrotorVisual`): fuselage along the centre-line frame colliders, each surface a plate with its flap (`TiltSurfaceVisual`, hinged on the trailing edge, axis `roll·(−Y)`), booms joining each side's pivots, gear legs, one pod per rotor (`NacelleVisual`: motor can, spinner, disc at `offset` along the thrust axis) posed by `R_y(tilt)` at its pivot. `sync_tiltrotors` turns the flaps by the mixing gains times the channel deflections and the pods by the mount tilts; disc opacity from rotor speed over `full_throttle_speed()`. Chase camera as the helicopter's.
  - **Keyboard flight** (`Sim::tilt_setpoint`, the group's `velocity` action map for scales): `velocity`: W/S move a forward-speed setpoint at 3 m/s² (−sideways…forward limit; the mounts convert with it), A/D fly sideways weighted by `1 − wing_share` and turn on the wing (yaw rate `+ wing·left`), Space/Shift climb, Q/E yaw. `attitude`: W/S move the airspeed setpoint, A/D bank, pitch from the schedule's level trim at that airspeed, Space/Shift climb. `M` skips `rates` for tiltrotors. `TiltSchedule::wing_share(V) = smoothstep((V − 0.9V_s)/(0.4V_s))` is shared with the controller.
  - **HUD** (`tilt_status`): airspeed with the setpoint and climb, α (red past 85 % of the wing's stall margin, STALL past it) and β, electric power, throttle bars per rotor, mount tilts against the scheduled tilt with hover / converting / wing and the wing share, a corridor diagram (feasible band as one unfeathered mesh, stall speed, schedule line, the aircraft's point), horizon. Plots: velocity, position or bank/pitch/airspeed tracking (`Tracking::Attitude`), live and from recorded actions.
  - **Replay**: `Tiltrotor::show(&TiltrotorDisplay)` sets rotor speeds (interpolated), throttles, tilts, channels, electric power and air data from the recording; rotor momentum recomputed.
  - **Tests**: `keys_fly_a_tiltrotor` (hover hands off; W to 1.8·V_s converts with the mounts > 1.5 rad and height within 5 m; A turns on the wing; S back to 0 converts back to a still hover; attitude A banks), `replayed_tiltrotors_show_their_mounts`, `tilt_pods_and_flaps_follow_the_state`, `tiltrotor_visual_matches_its_definition`, `tiltrotor_quantities_match_their_setpoints`. Screenshot run on the training map: ~130 fps on the Iris Xe.
- **Step 4: `TiltrotorDelivery-v0`** (`tasks/tiltrotor_delivery.py`):
  - **Task**: `quadtilt_like` in `velocity` (speeds 25, 4 and 1.5 m/s; yaw rate 0.5 rad/s), spawned on the gear on one farm's pad of the `delivery` map (6 km of farmland at 2 m cells, farms about 400 m apart; about 100 s to generate cold), with the goal on another farm's pad 2–4 km away. Mean wind 0–4 m/s, turbulence W20 0–2; 10 Hz policy, 360 s. Success: `LANDED` within 3 m of the pad (ends the episode).
  - **Observation (22)**: goal in the heading frame (1/1000), AGL, air data, body velocity and rates, pitch and roll, last action, and the goal again at 1/10 clipped to ±1 for the last metres.
  - **Reward**: decrease of `φ = √(d² + h²)` per 100 m, −0.002 per step, smoothness 0.02, −0.2 on a stall step, +20 on success, −20 on a crash, water or leaving the map.
  - **Yard goals** (`GoalKind::Yard`, aerial only; `bay::{pad, trip, YardTrip}`): a trip joins two farm yards, the destination drawn by the pad-to-yard distance within `goals.distance` (else the closest to it). Maps with fewer than two yards are dropped from the pool (an error if none is left). Each yard's pad is the most open point of a 2 m grid (±16 m along, ±12 m across the yard): the smallest obstacle-and-foliage distance over a column 1–40 m up (`PAD_COLUMN`, capped at 15 m), less 0.1 per metre off the centre. `StaticWorld::clearance` could not be used for this, since it includes the terrain. A ground spawn stands on the terrain under the pad.
  - **Map shortcuts** (`tasks/base.py`): `farmland` (2 km rural showcase at 2 m cells, about 16 farms) and `delivery` (the demo map above). Rural maps take a `config` override.
  - **Controller fixes** found by the scripted pilot:
    - **On the gear**: a full-yaw demand saturated the allocation against the gear's friction, so the aircraft never lifted off. While grounded, the attitude and heading are now held with zero yaw rate.
    - **Heading-frame rotation**: in hover and conversion, yawing while flying sideways was not compensated, so a yaw plus side command made the aircraft orbit. The velocity loop now adds `ω × v` (weighted by `1 − wing`).
    - **Weathervane** (`TiltSchedule::vane_share`, 0 below 0.5·V_s, full from 0.9·V_s): in wind at about 10 m/s airspeed, below the wing share, nothing held the nose into the airflow. The sideslip grew to 1 rad and the aircraft departed. Below the wing, a sideslip beyond `MAX_SIDESLIP` (0.25 rad) now yaws the nose back. Full coordination (no deadband) stays with the wing share.
    - The first version removed all sideslip. The velocity loop holds zero sideways ground speed in the heading frame, so the only steady state left was flying straight up- or downwind: the nose turned into the wind, the pilot stopped and turned back, and the loop repeated for minutes. The deadband leaves room to crab.
    - **Leash and integrator**: the velocity reference's leash was symmetric. A crosswind trim in the integrator (up to 0.3 g, frozen while the reference moves) then cancelled the leashed error, and the aircraft stayed stuck at 11 m/s. The leash is now centred on the integrator, so error plus trim always reaches the configured acceleration.
  - **Scripted pilot** (`scripted(obs)`):
    - Climbs straight up to 25 m AGL, yawing toward the pad beyond 10 m.
    - Once aligned within 0.3 rad (or within 50 m), flies toward the pad at the speed from which it can stop at 0.6 m/s² (at most 25 m/s; 0.4 × distance close in), holding 50 m AGL. The controller converts the mounts with the speed.
    - Descends at 1.2 m/s within 1.5 m of the pad, and at 0.5 m/s below 4 m AGL.
    - **Results**: `farmland`, 400–1200 m trips: 8 of 8 within 0.07 m in 115–149 s. `delivery`, 2.1–3.4 km trips: 8 of 8 within 0.07 m in 178–237 s.
  - **Short training** (`ppo_continuous.py --hidden 256 --bound-coef 0.01 --gamma 0.995`, 64 worlds × 128 steps, 3M steps on `farmland` with 300–800 m trips, 20 min at 2.5k SPS; basic training only, to check the pipeline):
    - The return rises from −78 (early crashes) to −12.4, and deterministic evaluation on unseen maps (`map_seed` 1000) survives all episodes.
    - The policy learned to stay parked on its pad: the action is pinned at full down, which keeps the thrust off on the gear, and the time penalty (−7.2 over an episode) costs less than a crash (−20). No trip was flown in 3M steps.
    - Left to the user's training: a curriculum from airborne spawns (as the aircraft tasks), a lift-off shaping term, or a longer run.
  - **Export and viewer**: `policy.json` plays in `viewer policy` on map seed 1000 (160 fps). `eval_record.py` records two episodes that re-simulate bit for bit, and `viewer replay` plays them with plots (105 fps).
  - **Tests**:
    - `control/tests/tiltrotor.rs::slow_flight_in_the_conversion_band`: side and yaw commands at 5–14 m/s keep the sideslip < 0.45 rad above 8 m/s airspeed and track the speed. In a 7 m/s crosswind at ~11 m/s airspeed, the aircraft turns more than 3 rad and stops over the ground.
    - `sim/tests/tiltrotor.rs::yard_goals_join_two_farm_pads`: spawns and goals are pads 300–700 m apart, clear of obstacles 1–30 m up; the aircraft stands still, then lifts off with full climb and yaw.
    - Python: the generic env tests (on a 1 km farm map), `test_tiltrotor_delivery_scripted_pilot_lands` (8 of 8 within 3 m) and `test_tiltrotor_delivery_rewards`.
- **M6c done 2026-09-29.**

**To confirm while building**:
- JSBSim's PyPI package and licence for generating fixtures offline (its aircraft data is LGPL; only derived fixtures are committed);
- access to Beard & McLain's Aerosonde parameters and Padfield's Bo105 data;
- Gavrilets' X-Cell 60 parameter table;
- whether the aerial physics rate stays 500 Hz for fixed-wing aircraft (the short period of a small UAV is several Hz) and moves to 1 kHz for helicopters;
- the tile size and cache bound against memory on the laptop.

## Milestone 7: Camera sensors

Planned 2026-09-29. Decided with the user at the start:
- **Renderer**: a new **`render` crate on plain wgpu**, fed by the `scene` meshes. There is one GPU context per process; the cameras of all worlds are batched into texture atlases with one readback per step. Bevy stays out of the training path.
- **Outputs**: **RGB, depth and semantic** from one pass (multiple render targets). Cameras are ideal pinhole cameras with noise, exposure and latency. Lens distortion, motion blur and rolling shutter are deferred.
- **Determinism**: images are **bit-identical on the same GPU and driver**. Tests and golden image hashes run on **lavapipe** (Mesa's Vulkan software rasterizer, installed on the laptop), so they do not depend on the machine. Images from different GPUs are compared within a tolerance.
- **Python**: a Gymnasium **`Dict` observation `{state, image}`**. `state` is the flat float vector as today; `image` is uint8 `[H, W, C]` per camera, written in place into preallocated numpy buffers. Tasks without cameras keep their plain `Box`. A new `ppo_pixels.py` trains a small CNN encoder.
- **Target**: on the laptop (Iris Xe), **≥ 3,000 camera frames/s at 64×64 RGB + depth** for a batch of 64 worlds, i.e. PPO at ≥ 1.5k SPS. The 7900 XT is measured, with 5–10× expected.
- **Demo**: **a drone lands on a moving car**. The car is driven by a **scripted driver** along rural roads; only the drone learns. The drone sees a **64×64 downward camera** (RGB + depth; semantic for debugging and auxiliary losses) and its own state (velocity, attitude, AGL). It is not given the car's position, and it acts in `velocity` mode.
- **Training**: as before, only enough to test what was built; the user trains the agents.

The rest of this section is proposed and open to change.

**Already in place**:
- **`scene` meshes**, independent of any renderer: terrain chunks with LOD strides, water, road ribbons, obstacles merged per chunk, and visuals for every vehicle family (multirotor, fixed-wing, tiltrotor, helicopter, `wheeled` cars/trucks/trailers, tracks, single-track).
- **wgpu 29** is already in the lockfile, through Bevy 0.19.
- **Sensor timing** (`Timing { divider, latency }`, `DelayLine`) and the LiDAR as a template. The LiDAR sees terrain, obstacles and other agents (`SceneRays`), which gives a reference for depth and semantic images.
- **Agent contacts**: a drone rests on a car's roof (`AgentContacts.supported`, `sim/tests/sim.rs::drone_rides_on_a_car_roof`).
- **Road geometry**: `lane::Follow` (`heading`, `curvature` and `point` ahead). Pure-pursuit drivers exist, but only in tests.
- **Batching**: `BatchSim::step` runs physics per world on rayon, then `gather`s serially. A batched render fits between the two.

**Not in place**:
- No GPU code outside the viewer.
- **No Rust-side scripted agents**: every group takes actions from Python (`BatchSim::step` asserts one action array per group), and `set_action` overwrites `set_command` each step.
- **`LANDED` uses absolute speed**, so it cannot trigger on a moving car. Nothing records which agent an agent rests on.
- **Observations** are one flat `f32` vector per agent; there is no image buffer.
- **The Rust policy runner** (`sim::policy`, used by the viewer) runs MLPs only.
- **The recorder** writes state, pose, action, events and optional LiDAR; there are no image channels.

#### Design
- **`render` crate** (`autonomousim-render`: wgpu, `scene`, `world`; no Bevy):
  - **Device**: headless, with the adapter chosen by preference (discrete → integrated → lavapipe). `AUTONOMOUSIM_RENDER_ADAPTER` names one explicitly; tests force lavapipe. Without a GPU, lavapipe still works, only slowly.
  - **World residency**: each map in the pool is uploaded once (terrain chunks at 2–4 LOD strides, water, roads, merged obstacles) as vertex buffers per chunk. Vertices carry colour and a semantic class. Maps are shared across worlds by `Arc` identity, like `StaticWorld`.
  - **Vehicles**: one mesh set per vehicle definition, drawn instanced with per-agent part transforms (body, wheels, rotors, surfaces, mounts). The poses come from the same functions the viewer uses to pose its visuals.
  - **One pass, three targets**:
    - RGB (`Rgba8Unorm`) with sun and ambient shading, fog and sky colour;
    - linear depth (`R32Float`, metres along the optical axis, 0 for sky);
    - semantic class (`R8Uint`).
    - No MSAA; sensors alias like real ones at 64².
  - **Semantic classes**: sky, terrain (by material group: grass/field, forest floor, rock/scree, sand/mud, snow), water, road, tree trunk, canopy, rock, building/farm, vehicle (own), vehicle (other agents). The table is fixed and versioned.
  - **Batching**: all cameras due in a step render into atlas textures (a 64×64 tile per camera), with one draw list per tile and chunk-level frustum culling. There is one `copy_texture_to_buffer` and one map-read per step, and double-buffered readback lets the next step's physics overlap the copy.
- **Camera sensor** (`sensors::camera`, `SensorConfig::Camera`):
  - **Configuration**: `width`, `height`, `fov_deg` (horizontal), mount pose (FLU position + roll/pitch/yaw), `rate_hz`, `latency`, `outputs` (any of `rgb`, `depth`, `semantic`), `near`/`far`, and noise (Gaussian pixel noise, depth noise ∝ d², exposure jitter). Noise is applied on the CPU from the sensor's seeded stream, so it is deterministic everywhere.
  - **Timing**: a camera's period must be a multiple of the policy period (one frame at most per policy step, rendered from the poses at the end of the step). Latency is a whole number of camera frames, held in a `DelayLine` of images.
  - **The render runs in `BatchSim`, not in `WorldInstance::tick`**: after the parallel physics, the batch collects the due cameras of all worlds, renders once and scatters the images. Single worlds (viewer, `WorldInstance`) use the same path with a batch of one.
- **Image outputs**: per group `[num_envs, count, cameras, H, W, C]` uint8 (depth as float32 metres in its own array, or quantised to uint16 mm), next to the flat obs. `GroupSpec.obs` gains `{ term = "camera", sensor = "down", output = "rgb" }` entries, which go to the image dict instead of the flat vector.
- **Scripted groups** (`GroupSpec.driver`): a group with a Rust driver takes no actions from Python and is left out of the Python action and observation arrays (it stays in the state array for rewards). The first driver is `road`: pure pursuit on `lane::Follow` with a curvature-limited speed profile, a random target speed (`speed = [3, 12]` m/s), optional random stops and a random turn at junctions. It is deterministic from the world's seed stream.
- **Landing on agents**: `LANDED` compares speeds relative to the supporting agent's velocity at the contact. A new state column `support` holds the index of the agent an agent rests on (−1 for none), so tasks can tell "landed on the car" from "landed on the road".
- **Recording and viewer**: recordings do not store images, because replay re-renders them from the recorded state (deterministic on the same machine). An optional `/agent/<id>/camera` channel (PNG, low rate) serves Foxglove. The viewer shows a camera panel (RGB/depth/semantic toggle) from the `render` crate, uploaded as an egui texture, so it shows exactly what the policy sees, plus a frustum gizmo.
- **Rust CNN inference**: `sim::policy` gains conv layers (conv 3×3/stride, ReLU, flatten) matching `ppo_pixels.py`'s encoder, so exported pixel policies fly in `viewer policy`.
- **Demo `DroneLandOnCar-v0`**:
  - **Setup**: an `iris_like` drone in `velocity` mode and a scripted car on `rural` training maps. The drone spawns 15–30 m above the car with the car inside the camera footprint (±10 m offset), and the car drives at 3–12 m/s. There is mild wind.
  - **Observation**: `image` is the 64×64 downward camera (RGB + depth); `state` is body velocity, rates, attitude, AGL and the last action.
  - **Reward**: shaping on the privileged relative position (reward only, not observed); success is `LANDED` with `support` = the car. A crash or losing the car for more than 5 s ends the episode.
  - **Curriculum options**: car speed range, stops on or off, spawn offset.
  - **Scripted pilot**: flies from the semantic image only (centroid and size of the "other vehicle" pixels → a velocity command; depth for height), to prove the camera loop without learning.

#### Implementation order
| # | Step | Done when |
|---|---|---|
| 0 ✅ | `render` crate: headless device and adapter choice (incl. lavapipe), offscreen RGB/depth/semantic of a single mesh, readback, pinhole intrinsics | Depth of an analytic plane and box matches within 1e-4 relative; the image hash on lavapipe is stable across runs and thread counts; golden hash committed |
| 1 ✅ | World and vehicle residency: map upload from `scene` (LOD, culling), semantic classes, instanced vehicles with articulated parts | Depth and class match LiDAR ray casts from the same poses (terrain, trees, rocks, roads, water, a car, a drone) within tolerance on test and procedural maps |
| 2 ✅ | Camera sensor and sim integration: `SensorConfig::Camera`, batched render in `BatchSim`, image buffers, noise, rates and latency, reset | Determinism suite: the same seed gives the same images on one adapter; world 0 is identical at N = 1 and N = 64; latency is exact in frames; state goldens unchanged |
| 3 ✅ | Performance: atlas batching, multiple maps per batch, double-buffered readback, benchmarks | ≥ 3,000 frames/s at 64×64 RGB + depth for 64 worlds on the Iris Xe; 7900 XT measured; results in `benchmarks/results` |
| 4 ✅ | Python: `Dict` observations in the vector, multi-agent and PettingZoo envs; in-place uint8 buffers; `ppo_pixels.py` (CNN encoder) | `check_env` and vector tests pass with images; a small pixel task (hover over a pad from the down camera) trains in minutes |
| 5 ✅ | Scripted groups (`GroupSpec.driver = road`), relative `LANDED` and the `support` column | The car drives 10 min of random routes on rural maps without leaving the road, deterministically; the group is absent from the Python arrays; a drone landed on a car at 8 m/s reports `LANDED` with `support` = the car |
| 6 | Viewer: camera panel (RGB/depth/semantic), frustum gizmo, scripted cars in live and replay; optional camera recording channel | Live and replay show the feed at ≥ 60 fps with one camera on the Iris Xe; replayed images equal the live ones on the same machine |
| 7 | Rust CNN inference and export for pixel policies | Rust outputs match PyTorch within 1e-5 on random inputs; an exported pixel policy flies in `viewer policy` |
| 8 | `DroneLandOnCar-v0`: task, scripted pilot from the semantic image, short training, export, viewer, replay | As for the other demos |

#### As built
- **Step 0 (`render` crate)**, `autonomousim-render` (`crates/render`; wgpu 29 with only the Vulkan backend and WGSL; the lockfile has one wgpu, shared with Bevy's): `GpuContext`, `AdapterChoice`, `Intrinsics`, `CameraPose`, `GpuMesh`, `Draw`, `View`, `Shading`, `Renderer`, `Frame`, `SemanticClass`, `RenderError`:
  - **Device** (`GpuContext::new(&AdapterChoice)`): headless, no surface; `Auto` ranks discrete → integrated → virtual → CPU, `Software` takes the CPU rasterizer (lavapipe), `Named` matches the adapter name; `AUTONOMOUSIM_RENDER_ADAPTER` (`auto`, `software`/`lavapipe`/`cpu`, or a name) via `from_env`; `describe()` gives name, type and driver for benchmark and golden records. Futures are driven by a small `block_on` (no async runtime).
  - **Camera model**: ideal pinhole, square pixels, principal point at the centre; `Intrinsics { width, height, fov_x, near = 0.05, far = 1000 }`, `ray(u, v)` (unit axis depth), `project`. Camera frame FLU with the optical axis +x: `u = cx − f·y/x`, `v = cy − f·z/x`; `CameraPose.orientation` rotates camera → ENU.
  - **Pass**: one pipeline, three colour targets (Rgba8UnormSrgb shaded colour, R32Float depth along the optical axis, R8Uint class) plus a reversed-Z Depth32Float buffer (compare Greater, clear 0). Model-view-projection composed per draw in f64 relative to the camera, then rounded to f32 (no precision loss far from the origin); the linear depth is clip w, interpolated perspective-correctly. Lighting: two-sided Lambert with ambient share, `ambient + (1 − ambient)·max(n·sun, 0)`, the sun turned into each mesh's frame; no culling (back faces lit from their own side). Per-draw uniforms in 256-byte slots with dynamic offsets; targets and read-back buffers are recreated only when the image size changes; rows unpadded on readback. Sky: clear colour, depth 0, class 0.
  - **Meshes**: `GpuMesh::new(ctx, &MeshData, class)` or `with_classes` (per vertex, flat-interpolated); vertex = position, normal, linear colour, class (48 bytes).
  - **Semantic classes** (`SEMANTIC_VERSION` 1, append only): sky, grass, forest floor, rock, soil, snow, water, road, trunk, canopy, boulder, building, own vehicle, vehicle.
  - **Tests** (`crates/render/tests/render.rs`, all on lavapipe): a tilted, rolled, yawed camera over a plane matches the analytic depth within 1e-4 relative on every pixel whose ±0.6 px neighbourhood is all ground (plus class and sky checks); a rotated box on the ground matches slab intersection in depth and class away from edges; shading against the sun (overhead, 60°, below the horizon, a back face from both sides, a flipped mesh, the sky colour) within 1 LSB of the sRGB encoding; the same frame from the same and a new renderer; the golden hash (`fixtures/golden_images.toml`, with the lavapipe/Mesa version in its header); the test binary rerun with `LP_NUM_THREADS` = 1 and 4 gives the same hash.
  - **First numbers** (`cargo run -p autonomousim-render --release --example adapters`): one 64² frame with synchronous readback takes ~240 µs on the Iris Xe and ~170 µs on lavapipe, which is latency-bound; batching many views per submission comes in step 3.
- **Step 1 (world and vehicle residency)**, `render::{GpuWorld, WorldOptions, GpuRig, terrain_class, obstacle_class}`, `View::may_see`, `Draw::{with_scale, with_class}`; `scene::rig::{Rig, Placement}`, `scene::terrain::terrain_vertex_materials`, `scene::props::shown_obstacle`:
  - **Map upload** (`GpuWorld::new(ctx, &StaticWorld, WorldOptions)`): the map cut into 64-cell chunks, each with its terrain at every LOD stride (1/2/4/8; skirts as the viewer's), opaque water, and its obstacles merged at two levels of detail; road ribbons per chunk. Chunk meshes are relative to the chunk's corner (the MVP is composed in f64, so precision holds anywhere on the map). `draws(&View, &mut Vec<Draw>)` culls chunks against the frustum (a box is dropped when all its corners are outside one plane), takes the terrain stride by distance (160/380/800 m, as the viewer), and obstacles in camera detail within 250 m. Tiled maps are refused (`RenderError::Unsupported`) until needed. The wild training map uploads in 0.2 s (1.23M triangles at full detail), the rural one in 0.04 s.
  - **Camera detail**: obstacles at 12 segments, 8 for capsules and icosphere level 2 near the camera (`WorldOptions::props`), the viewer's detail (7/5/1) beyond. At the viewer's detail 4 % of the tree pixels on the wild map were off by more than the tolerance (grazing rays past a faceted crown).
  - **Classes**: terrain by the cell material (`terrain_class`: grass/meadow/crop → grass, forest floor, rock/scree → rock, sand/mud/dirt/plowed → soil, snow, water, asphalt/concrete/gravel → road, wood/metal → building, foliage → canopy), carried as a flat attribute of each triangle's first vertex (`terrain_vertex_materials` gives the south-west cell of every vertex, so at stride 1 a quad is labelled with exactly its cell); obstacles by tag (`obstacle_class`: trunk, conifer and broadleaf crowns and foliage hedges → canopy, rocks → boulder, pillars, walls, fences, buildings and silos → building; untagged by class and material); water surfaces → water; road ribbons → road; vehicles → vehicle, or own vehicle through the per-draw class override.
  - **Vehicle rigs** (`scene::rig`): a vehicle's visual as a flat list of part meshes, and `Rig::place(&Vehicle, &mut Vec<Placement>)` poses each in the vehicle frame from the state (control surfaces by the pilot-sense deflections, tiltrotor flaps by the mixed channels and pods by the mount tilts, helicopter blades by flapping and coning, trailer units, wheels and suspension links, a motorcycle's fork and rider). `Placement` has a scale for the stretched links, and `Draw` passes it to the shader (normals scale inversely). The viewer's helper functions moved into `rig` and the viewer uses them; its scene graph stays. Left out: rotor discs (translucent), blade azimuth (blades stand at 0; there is no azimuth in the state), animated track bands (built once).
  - **Tests** (`crates/render/tests/world.rs`, lavapipe, 96×72 at 90°): every pixel against the static world's ray cast (terrain, water, solid and foliage obstacles, the LiDAR's scene) over pixels whose ±0.5 px corners hit the same thing; road ribbons count as road over the terrain within 0.15 m. Class agreement / depth within tolerance (terrain and water 1 % + 5 cm, obstacles 1 % + 0.1 m / cos incidence) / terrain and water nearer than 100 m within 1e-3 relative: forest patch 0.9994 / 0.9994 / 1.0000, lake 1 / 1 / 1, wild training map 0.9923 / 0.9998 / 0.9999, rural training map 0.9984 / 0.9993 / 1.0000, with every expected class in view (trunks, crowns, boulders, water, roads, buildings). Vehicles (sedan, tractor with trailer, quadrotor, helicopter, tiltrotor, motorcycle with rider) against a CPU ray cast of their placed rig triangles (the LiDAR sees agents as spheres): all clean pixels agree in class and depth within 1e-3 except one on the motorcycle. The own-vehicle override, trailer wheels at the simulated wheel poses and tiltrotor pods at the mount tilts are checked too.
- **Step 2 (camera sensor and sim integration)**, `sensors::camera::{Camera, CameraConfig, CameraNoise, CameraImage}` (`SensorConfig::Camera`, `type = "camera"`), `sim::camera::{Cameras, Capture, gpu, use_adapter, has_cameras}`, `WorldInstance::{deliver, observe_images}`, `BatchSim::{images, image_shape}`, `ObsTerm.output` and `obs::CameraOutput`:
  - **Configuration**: `width`, `height` (default 64×64), `fov_deg` (horizontal, 90), `mount` (position and rotation vector; camera frame FLU with the optical axis +x, so `rotation = [0, π/2, 0]` looks down), `rate_hz` (25), `latency` in frames (0–64), `near`/`far` (0.05/1000 m), `noise { pixel, depth, exposure }` (σ as a fraction of full scale on the encoded values; σ = `depth`·d²; σ of the log of a per-frame gain). No `outputs` list: one pass renders colour, depth and class anyway, and the observation terms pick what they need.
  - **Timing**: the camera period must be a whole number of policy steps (checked when the scenario compiles). A frame is due when the world's tick is a multiple of it, so after every k-th policy step and at every reset (tick 0). The sensor keeps the last `latency + 1` frames (shared by `Arc`, so snapshots stay cheap); the oldest is visible. The first frame after a reset fills the whole line, so images are never empty.
  - **Rendering** (`sim::camera::Cameras`, one per batch): maps uploaded on first use per pool index, one `GpuRig` per group; for each due camera of an active agent, the map's draws plus every active agent's rig (its own labelled own vehicle). Disabled agents are neither drawn nor rendered for. `BatchSim` steps physics in parallel, renders the due cameras world by world, then delivers (noise from the camera's seeded stream) and writes outputs in parallel; without cameras the step is unchanged. `Cameras::update` does the same for a single `WorldInstance`. One GPU context per process (`sim::camera::gpu`, from `AUTONOMOUSIM_RENDER_ADAPTER`; `use_adapter` picks one first, e.g. lavapipe in tests). Render errors panic in `BatchSim` (device loss).
  - **Observations**: `{ term = "camera", sensor, output = "rgb" | "depth" | "semantic" }` are left out of the flat vector and fill a `u8` image per agent `[height, width, channels]`, channels concatenated in term order (rgb 3, depth 1, semantic 1); all camera terms of a group must have one image size. Depth is quantised over the term's `range` (default 100 m) to 0–255, with 255 for sky and beyond; float depth stays available from the sensor in Rust. `BatchSim::images(g)` is `[num_envs, count, H, W, C]`; `refresh` keeps the last frames. Recordings store no images.
  - **State hash**: includes each camera's visible frame (tick and a blake3 digest); scenarios without cameras hash as before, and the state goldens are unchanged.
  - **Tests** (`crates/sim/tests/cameras.rs`, lavapipe): three drones with three cameras over the forest patch give identical images and state hashes for the same seed, with world 0 identical at N = 1 (1 thread) and N = 64 (4 threads), and the same images after a seeded reset; a two-frame-latency camera shows exactly the frame the undelayed one captured two frames (40 ticks) earlier, and the 25 Hz cameras capture every other policy step; noise is seeded, changes colours but not classes; a down camera 9.9 m over flat grass reads 9.9 m (1e-4) at the centre, grass everywhere else, the quantised depth and class in the observation, and a drone placed below as `vehicle` pixels at the right depth; invalid setups (camera faster than the policy or off its period, missing or misplaced `output`, a range on a colour image, scaled images, two image sizes in a group) are refused. Sensor unit tests cover the delay line, seeded noise and validation.
- **Step 3 (performance)**, `Renderer::render_batch(ctx, &[Job], &[Draw])`, `render::Job`, `Cameras::capture_batch`, `crates/sim/examples/camera_bench.rs`:
  - **Batching**: all cameras due in a `BatchSim` step are rendered in one submission. Every image gets its own array layer of layered targets per image size (up to 256 layers per set; more sets when needed), one render pass each, then one copy per target for all layers and one map and wait. Layers instead of a viewport atlas: an image rendered at a pixel offset could round differently, and with layers every image is bit-identical to rendering it alone, whatever its place in the batch (tested with 300 images of two sizes). Draw lists and uniforms for all views are built serially (~0.4 ms for 64 views); multiple maps per batch just work (maps upload on first use).
  - **Fewer targets**: the class id moved into the colour target's alpha (alpha is stored linearly in sRGB formats, so id/255 reads back exactly): two colour targets plus the depth buffer instead of three. The golden image is unchanged.
  - **Smaller chunks**: `WorldOptions::chunk_cells` 64 → 32. A 64² down camera at 10–30 m sees a 20–60 m footprint, and whole 64 m chunks sent twice the triangles through the vertex stage (3.4M → 1.6M triangles per step for 64 views).
  - **Where the time goes** (Iris Xe, 64 views of 64² on rural maps): about 4 ms is fixed cost per render pass in the driver (~65 µs each, measured with empty passes), about 4–5 ms is the draws, and encoding plus readback take under 1 ms. Separate depth buffers per layer did not help (tried). Physics is under 0.3 ms of a 64-world drone step, so double-buffered readback (overlapping physics with the GPU) would gain under 5 % and was left out.
  - **Results** (`cargo run -p autonomousim-sim --release --example camera_bench -- [--adapter …] [--map rural|wild|forest] [--size N] [--envs N] --save`, in `benchmarks/results/2026-09-29-robert-Latitude-7440.json` under `camera`), 64 worlds, one 64² RGB + depth down camera each, drones 10–30 m above ground, 10 threads, Iris Xe: **4,700–5,900 frames/s on 4 rural maps**, 3,600 on 4 wild maps (dense forest), 5,800 on the forest patch; 3,800 at 128²; 6,800 with 256 worlds. Per-frame synchronous rendering before this step: 2,250. Lavapipe: 490 frames/s. **The 7900 XT is not measured yet**: it is on the desktop; run the example there with `--save`.
  - The target of ≥ 3,000 frames/s at 64² for 64 worlds on the Iris Xe is met on every map type.
- **Step 4 (Python)**, `BatchSim.images(group)`, `group_info` `image_shape`/`image_layout`, `set_render_adapter`, `render_adapter`, `SEMANTIC_CLASSES` in `autonomousim._native`; `vector_env.obs_space`/`copy_obs`; `GoalSpec.pad`, `SemanticClass::Marker`, `scene::mesh::landing_pad`; `QuadHoverPad-v0` (`tasks/hover_pad.py`); `examples/ppo_pixels.py`:
  - **Buffers**: per group with camera terms a numpy `uint8 [num_envs, count, H, W, C]` created once and overwritten in place by every step and reset (like obs, state and events); `None` without camera terms. `image_layout` names the channels `<sensor>/<output>` with their first channel and count. Render errors raise `RuntimeError`.
  - **Spaces**: a group with camera terms observes `Dict({"state": Box(obs_dim), "image": Box(0, 255, (H, W, C), uint8)})` in the single, vector, multi-agent (per group, batched `[num_envs, count, …]`) and PettingZoo (per agent) environments; groups without cameras keep their `Box`. `final_obs` is a dict of dense arrays too; `copy=False` returns the buffers themselves.
  - **Adapter**: `AUTONOMOUSIM_RENDER_ADAPTER` as in Rust, or `set_render_adapter` before the first scenario with cameras; `tests_py/conftest.py` sets `software`, so the Python tests render on lavapipe.
  - **Landing pads** (a visible target for pixel tasks; the class table is appended, `SEMANTIC_VERSION` 2): `goals.pad` (radius, m; 0 = none) draws a flat pad (orange disc, white ring and bar) 2 cm above the ground under each active agent's current goal, tilted with the terrain, class `marker`; visual only (no collider, not in LiDAR).
  - **`QuadHoverPad-v0`**: a pad (0.75 m) on flat grass up to 1.5 m from the spawn (2–3.5 m AGL); hold 2.5 m above it. `velocity` actions; observation = a 32×32 RGB down camera at 50 Hz (`depth=True` adds a depth channel) plus rot6d, body velocity, rates, AGL and last action (17), without the pad's position. QuadHover's reward and end conditions around the hover point.
  - **`ppo_pixels.py`**: 3×3 stride-2 convolutions (16/32/32 channels, ReLU) → linear 128 (ReLU), concatenated with the normalised state, separate tanh MLP heads (2×128), shared encoder; images stored as uint8 in the rollout buffer. Checkpoint `algo = "ppo_pixels"` with image shape, channels and state statistics; `load_policy` takes `Dict` observations (the `evaluate` helper works unchanged).
  - **Tests** (`tests_py/test_cameras.py`): the adapter, native buffers and layout (semantic + RGB + depth), `check_env`, vector spaces and views, seeded images reproduce, SAME_STEP `final_obs` and DISABLED reset masks, a multi-agent team of pad drones and a plain drone, PettingZoo `parallel_api_test`; the pad's pixel count and depth in `sim/tests/cameras.rs`. The pad task joins the `check_env` sweep; its vector test lives with the camera tests.
  - **Training** (Iris Xe, 64 worlds, 4 sim and 4 torch threads): 2,400 SPS (sim 30 %); the return rose from ~80 to 330 within 3 minutes and 390 of 500 after 1.5M steps (10.6 min). Evaluation on 64 new seeds: every episode survived, median final distance to the hover point 0.11 m (2.2 m untrained).

- **Step 5 (scripted groups, relative landing)**, `sim::driver::{DriverSpec, RoadDriverSpec, RoadDriver, DriverGeometry, Traffic, Turn}`, `lane::{WalkPos, drive_walk, WALK_EDGE}`, `interaction::Support`, `Agent::{driver, support}`, `CompiledGroup::scripted`:
  - **Spec**: `driver = { type = "road", speed = [3, 12], lateral_accel = 2, decel = 2, lookahead = [3, 0.5], leg = [150, 500], stop_rate = 0, stop_time = [2, 6], roads = ["paved"] }` (all optional). Only single-unit, non-single-track wheeled vehicles spawned `on_road` without `route` goals (checked when the scenario compiles). Spawns lie on the allowed road classes at least `WALK_EDGE` + 10 m inside the map.
  - **Routes**: rural road networks are trees, so the driver walks the network at random: at junctions it takes any allowed road but the one it came on; a leg (`leg`, m) ends at a turning point 10 m before a dead end or 35 m inside the map's edge, else it goes on. The route is the lane polyline (right-hand traffic) of the walk, extended 40 m + the braking distance ahead of the car (points more than 10 m behind dropped); a new cruise speed per leg. The station is searched in a window from 2 m behind to 30 m ahead, so self-crossing routes do not confuse it.
  - **Driving** (`SpeedCurvature` commands at the policy rate): pure pursuit to the lane point `lookahead[0] + lookahead[1]·v` ahead (5 m + 0.6 s cut the inner corner of junction turns off the road; 3 m + 0.5 s does not); speed limited by `√(a_lat/κ + 2·decel·d)` over the bends ahead, by the end of the route, by agents in a corridor ahead (half width + 1.2 m, 60 m, 4 m gap; 12 m when the car's own route ends in a turn within 40 m), and by a clearance check: the footprint swept along the commanded arc (0.5 m steps up to the braking distance, at most 20 m) keeps 0.4 m from the collision spheres of other agents (spheres it already touches only block it if it would press closer). While another agent is within 10 m of its turning point the car waits 16 m before it.
  - **K-turns**: at the turning point at a standstill, forward on full left lock and back on full right lock at 0.8 m/s, settling 1 s at a standstill before each shunt; a shunt ends when a leading corner (front or rear plus 0.3 m, at the sides and centre) would pass the road edge by 0.6 m, reach ground 0.15 m above the road's centre line, or come within 0.3 m of an obstacle or agent; done within 0.35 rad of the reversed heading (at most 40 shunts), then the walk goes on from the reversed end.
  - **Controller fix** (`control::ground`): when stopping (`v_ref = 0`) the combustion powertrain brakes against the motion whichever gear is engaged; before, a car rolling back in forward gear (after a reverse shunt on a slope) got neither throttle nor brake and bumped into banks. No golden trajectory changed.
  - **Scripted groups** have no actions: `BatchSim::step` takes an empty (or a full, ignored) array for them; the Python `BatchSim.step` takes arrays for the learning groups only; `group_info` gives `first_agent`, `scripted` and `driver`. The single and vector envs need one learning group with one agent (`learning_group`), the multi-agent and PettingZoo envs list learning groups in `groups` and scripted ones in `scripted` (added e.g. through `overrides`); scripted rows stay readable with `sim.state(g)`. `state_hash` covers the driver state.
  - **Relative landing**: agent contacts record the supporting agent (a supported contact whose normal is within 60° of vertical) with its contact-point velocity and angular velocity; `LANDED` then uses the velocities relative to it. `STATE_DIM` 33 → 34: the new last column `support` is the supporting agent's index or −1. Golden trajectories are unchanged (checked with an A/B hash of the first 33 columns against the previous commit); the goldens were re-blessed for the new column.
  - **Tests**: `sim/tests/drivers.rs`: two sedans on each map of a rural pool drive 10 min without leaving the road or a terminal event, more than 2 km each, turning round at least once and never standing for a minute (also checked from 6 other seeds, 5–7 K-turns per car); the same seed repeats the state hashes and routes; a drone group among scripted cars in batches (empty or ignored car actions, 1 vs 3 threads); scenario validation. `sim/tests/sim.rs`: a drone pressed onto the roof of a car at 8 m/s reports `LANDED` with `support` = the car. `tests_py/test_scripted.py`: group info, stepping with arrays for the drones only, single/vector/multi-agent/PettingZoo envs without the cars, and the one-learning-group check.
**To confirm while building**:
- ~~whether wgpu 29 can be shared as one workspace dependency with Bevy's~~ (yes, one copy in the lockfile);
- ~~lavapipe's determinism across thread counts (`LP_NUM_THREADS`)~~ (identical at 1 and 4 threads);
- ~~readback latency and atlas limits on the Iris Xe~~ (array layers, 256 per set; one map per step, ~0.4 ms for 64 images of 64²);
- ~~uint16 versus float32 for depth in Python~~ (uint8 over a range in the observation image; float metres from the sensor in Rust);
- the frame rate of large tiled maps (M6a) in the camera path, or restricting cameras to monolithic maps at first.

## Roadmap after M1
| M | Content | Validation |
|---|---|---|
| M2 | (Detailed above.) Ground vehicles I: `KcTravel` joint, MF 6.x tire (`.tir`, combined slip, relaxation length + low-speed damping), steering (prescribed or rack DoF), powertrain (engine map, clutch, gearbox, open/LSD/locked differentials), brakes; Ackermann car, diff-drive, skid-steer; ground action modes (raw, (v, ω), (v, κ)); 1 kHz preset | ISO 4138 constant radius, ISO 7401 step steer, braking, ISO 3888 lane change vs published or Chrono::Vehicle data |
| M3 | (Detailed above.) Multi-agent: PettingZoo ParallelEnv + native group-batched API, mixed air/ground teams, full-shape agent contacts, swarm performance (SoA fast path if needed), neighbor observations | pettingzoo API tests; 256 drones at ≥ 20× real time |
| M4 | (Detailed above; split into M4a/b/c.) Rural maps (spline road graph, terrain blending, fields, farms, dirt tracks) + trucks and trailers (fifth wheel, drawbar, 6×6/8×8, multi-axle steering, lifting the 4-axle limit) + **tracked vehicles** and soft soil (design below) | Offtracking vs analytic results; trailer reversing task; tracked checks below |
| M5 ✅ | (Detailed above; done 2026-09-28.) Bicycles and motorcycles (camber thrust, turn slip) | Whipple benchmark (Meijaard 2007): weave ≈ 4.292 m/s, capsize ≈ 6.024 m/s |
| M6 ✅ | (Detailed above; split into M6a/b/c; done 2026-09-29.) Fixed-wing (coefficient tables), helicopter (BEMT + first-order flapping), VTOL transition; large coarse maps with floating origin | Trim, phugoid/short-period checks; hover power vs momentum theory |
| M7 | (Detailed above.) Cameras: headless wgpu RGB/depth/semantic via `scene` | FPS on Iris Xe and 7900 XT |
| M8 | Urban maps (roads, blocks, lots, buildings, lane graph, traffic lights) + NPCs (IDM + MOBIL traffic, social-force pedestrians) | Traffic sanity checks; no NPC collisions |
| M9 | ROS 2 bridge (`ros2-client`/RustDDS first, zenoh as an option; Lyrical LTS); rosbag2 export; live viewer attach to a running simulation (moved from M3, 2026-09-25) | Round trip with `ros2 topic echo` |

### Tracked vehicles (M4; added 2026-09-24)
- **Scope and presets**:
  - `tracked_apc`: an M113-like armoured carrier from Chrono's `data/vehicle/M113` (BSD-3; Chrono also has the Marder).
  - `rover_tracked`: a rubber-tracked UGV of about 60 kg.
- **Model** (`vehicles::tracked`), built for RL throughput rather than individual track shoes (Chrono's shoe-by-shoe contact is far too slow):
  - **Tree**: chassis (Free) plus, per side, road wheels on trailing-arm suspension. The arms reuse `KcTravel` tables or a revolute arm with a torsion bar, together with the step-3 springs, dampers and stops.
  - **Sprocket and idler**: the driven sprocket and the idler (with a tensioner spring) are fixed to the hull.
  - **Track as a continuous band**: one speed DoF per side carries the inertia of the band, sprocket, idler and road-wheel spin. Road wheels turn kinematically with it.
  - **Internal resistance**: speed-dependent rolling resistance, after Wong. Tension is pretension plus the tensioner deflection, and it feeds the internal resistance.
- **Track–ground force element**:
  - **Patches**: the lower run is sampled as contact patches under and between the road wheels. Each patch takes its normal pressure from the road-wheel loads, spread over the patch length, and follows the terrain.
  - **Shear**: tangential force comes from shear displacement accumulated per patch, an aux state like the tyre's carcass deflection. It follows Janosi–Hanamoto: `τ = (c + p·tan φ)(1 − e^(−j/K))`.
  - **Slip and steering**: longitudinal slip comes from band speed against ground speed, and lateral slip from side-slip and yaw. Skid steering therefore produces its turning-resistance moment on its own (Wong, *Theory of Ground Vehicles*, ch. 7).
- **Obstacles**: the sprocket and idler get colliders, and the front run gets a swept capsule, so steps and logs are climbed through the existing contacts.
- **Soft soil**:
  - Materials gain Bekker–Wong parameters (k_c, k_φ, n, c, φ, K). Sinkage comes from the pressure–sinkage law, plus compaction and bulldozing resistance.
  - Tracks mainly pay off on soft ground, so this comes with them. The same terms can be added to tyres later (Chrono's SCM is the reference).
- **Steering and powertrain**: skid steering by a clutch-brake, a controlled differential, or dual electric or hydrostatic drives, one per side. These reuse the combustion engine and gearbox, the side-coupled electric motors and the bristle couplings. Action modes: `raw` (throttle, brake, steering as the side difference), `vw` and `per_side`.
- **Validation**:
  - **Analytic, drawbar pull**: drawbar pull against slip on soft soil, from the Janosi–Hanamoto integral.
  - **Analytic, turning**: the steady skid-steer turning radius and the sprocket torques against sprocket speed ratio (Wong's turning-resistance model).
  - **Analytic, grades**: gradeability on rigid ground.
  - **Chrono M113**: static loads per road wheel; straight acceleration; the steady turn at a fixed sprocket speed ratio (turn radius and yaw rate within 15 %, given the different track models); a braked hold on a 30 % slope.
- **Performance target**: ≤ 10 µs per tick for an APC with 5 road wheels per side (about 20 patches).

## Key risks and mitigations
- **Penalty contact stability**: stiffness from ω_c·dt, bristle friction, a sub-stepping option. M2 tires don't use this path.
- **Bevy churn** (0.20 expected around Q4 2026): Bevy code stays in `viewer`, `scene` is renderer-independent, versions and lockfile pinned, at most one upgrade per milestone.
- **parry churn**: all parry calls wrapped in `core::contact`/`world`; minor version pinned.
- **Iris Xe rendering budget**: merged vegetation, LOD, fog culling, quality presets; under 2M visible triangles.
- **Python overhead and thread oversubscription**: decimation in Rust, GIL released, preallocated buffers, a dedicated rayon pool, torch limited to 2–4 threads.
- **Determinism leaks**: the rules above plus golden-hash tests in CI.
- **Scope creep**: fixed M1 acceptance list; stretch items tagged; no ground-vehicle work before M1 is done.

## Ecosystem versions (checked 2026-09)
Bevy 0.19.1 (glam 0.32, wgpu 29) · bevy_egui 0.42 (egui 0.36) · parry3d-f64 0.31.1 (glam 0.33 via glamx) · PyO3 0.29.2 + numpy 0.29 · maturin 1.15 · gymnasium 1.3.0 · pettingzoo 1.27 · torch 2.14 · mcap 0.25 (Rust) / 1.4 (Py) · postcard 1.1 · rand_chacha 0.10 · criterion 0.8 · rustc 1.98.
Existing Rust Featherstone crates are immature, so we write our own. Pinocchio and MuJoCo (PyPI) serve as test references.

**Still to confirm during step 0**: whether maturin sets `PYO3_BUILD_EXTENSION_MODULE` (otherwise keep the `extension-module` feature), parry's current BVH type name, the egui_plot version for egui 0.36, the torch ROCm index, and the cf2x/iris parameter sources (done in step 5).
