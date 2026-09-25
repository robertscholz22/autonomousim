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
- **Export**: `uv run python examples/export_policy.py runs/<run>/policy.pt` writes `policy.json` next to the checkpoint (2.3 MB for the 2×256 forest policy). It holds:
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
| 3 | Python: `MultiAgentVectorEnv`, `MultiAgentTask`, per-agent stopping and per-world autoreset; a mixed drone + car scenario | Shape, dtype, masking, autoreset and seeding tests for one group and for a mixed team with different obs/act sizes |
| 4 | PettingZoo `ParallelEnv` | `parallel_api_test` and the seed test pass for a single-group and a mixed-team task |
| 5 | Swarm performance | 256 drones hovering in one world ≥ 20× real time; 128 unchanged or faster; benchmarks recorded |
| 6 | `ppo_multiagent.py` and a quick check task (`SwarmHover-v0`: N drones hold assigned slots in a formation without touching) | Formation error < 0.3 m and no agent contacts in 95 % of episodes after ≤ 15 min of training |
| 7 | `SwarmWaypointForest-v0`, training, viewer | ≥ 80 % of agents finish their waypoints on unseen maps and < 2 % of agents collide with another agent; the exported policy flies the swarm in the viewer; a recorded episode replays |

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

## Roadmap after M1
| M | Content | Validation |
|---|---|---|
| M2 | (Detailed above.) Ground vehicles I: `KcTravel` joint, MF 6.x tire (`.tir`, combined slip, relaxation length + low-speed damping), steering (prescribed or rack DoF), powertrain (engine map, clutch, gearbox, open/LSD/locked differentials), brakes; Ackermann car, diff-drive, skid-steer; ground action modes (raw, (v, ω), (v, κ)); 1 kHz preset | ISO 4138 constant radius, ISO 7401 step steer, braking, ISO 3888 lane change vs published or Chrono::Vehicle data |
| M3 | (Detailed above.) Multi-agent: PettingZoo ParallelEnv + native group-batched API, mixed air/ground teams, full-shape agent contacts, swarm performance (SoA fast path if needed), neighbor observations | pettingzoo API tests; 256 drones at ≥ 20× real time |
| M4 | Rural maps (spline road graph, terrain blending, fields, farms, dirt tracks) + trucks and trailers (fifth wheel, drawbar, 6×6/8×8, multi-axle steering, lifting the 4-axle limit) + **tracked vehicles** and soft soil (design below) | Offtracking vs analytic results; trailer reversing task; tracked checks below |
| M5 | Bicycles and motorcycles (camber thrust, turn slip) | Whipple benchmark (Meijaard 2007): weave ≈ 4.292 m/s, capsize ≈ 6.024 m/s |
| M6 | Fixed-wing (coefficient tables), helicopter (BEMT + first-order flapping), VTOL transition; large coarse maps with floating origin | Trim, phugoid/short-period checks; hover power vs momentum theory |
| M7 | Cameras: headless wgpu RGB/depth/semantic via `scene` | FPS on Iris Xe and 7900 XT |
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
