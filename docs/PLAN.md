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
| Ground scope | Diff-drive/skid-steer robots, Ackermann cars, multi-axle trucks + trailers, motorcycles/bicycles |
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
examples/{ppo_continuous,sac_continuous,eval_record}.py; later export_policy.py
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
| 13 | `QuadWaypointForest` (LiDAR). Stretch: `QuadLanding`, Rust MLP policy playback in the viewer | > 80 % success on unseen maps; end-to-end demo |

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

## Roadmap after M1
| M | Content | Validation |
|---|---|---|
| M2 | Ground vehicles I: `KcTravel` joint, MF 6.x tire (`.tir`, combined slip, relaxation length + low-speed damping), steering (prescribed or rack DoF), powertrain (engine map, clutch, gearbox, open/LSD/locked differentials), brakes; Ackermann car, diff-drive, skid-steer; ground action modes (raw, (v, ω), (v, κ)); 1 kHz preset | ISO 4138 constant radius, ISO 7401 step steer, braking, ISO 3888 lane change vs published or Chrono::Vehicle data |
| M3 | Multi-agent: PettingZoo ParallelEnv + native group-batched API, mixed air/ground teams, full-shape agent contacts, swarm performance (SoA fast path if needed), neighbor observations, live viewer attach over zenoh | pettingzoo API tests; 256 drones at ≥ 20× real time |
| M4 | Rural maps (spline road graph, terrain blending, fields, farms, dirt tracks) + trucks and trailers (fifth wheel, drawbar, 6×6/8×8, multi-axle steering) | Offtracking vs analytic results; trailer reversing task |
| M5 | Bicycles and motorcycles (camber thrust, turn slip) | Whipple benchmark (Meijaard 2007): weave ≈ 4.292 m/s, capsize ≈ 6.024 m/s |
| M6 | Fixed-wing (coefficient tables), helicopter (BEMT + first-order flapping), VTOL transition; large coarse maps with floating origin | Trim, phugoid/short-period checks; hover power vs momentum theory |
| M7 | Cameras: headless wgpu RGB/depth/semantic via `scene` | FPS on Iris Xe and 7900 XT |
| M8 | Urban maps (roads, blocks, lots, buildings, lane graph, traffic lights) + NPCs (IDM + MOBIL traffic, social-force pedestrians) | Traffic sanity checks; no NPC collisions |
| M9 | ROS 2 bridge (`ros2-client`/RustDDS first, zenoh as an option; Lyrical LTS); rosbag2 export | Round trip with `ros2 topic echo` |

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
