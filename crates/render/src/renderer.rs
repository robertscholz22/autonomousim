//! Meshes on the GPU and the camera pass.

use autonomousim_scene::MeshData;
use bytemuck::{Pod, Zeroable};
use glam::{DMat4, DQuat, DVec3, DVec4};
use wgpu::util::DeviceExt;

use crate::camera::{CameraPose, Intrinsics};
use crate::context::{GpuContext, RenderError};
use crate::semantic::SemanticClass;

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const CLASS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Uint;
const Z_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Per-draw uniforms sit at this stride (the largest minimum uniform offset alignment).
const DRAW_STRIDE: u64 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 4],
    class: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawUniform {
    mvp: [[f32; 4]; 4],
    sun: [f32; 4],
    scale: [f32; 4],
    /// x: class for every pixel of the draw, or [`NO_CLASS`] for the mesh's own.
    class: [u32; 4],
}

const NO_CLASS: u32 = u32::MAX;

/// A mesh uploaded to the GPU, in its own frame (placed by each [`Draw`]).
pub struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

impl GpuMesh {
    /// Upload `mesh` with one class for all of it.
    pub fn new(ctx: &GpuContext, mesh: &MeshData, class: SemanticClass) -> Self {
        Self::with_classes(ctx, mesh, &vec![class.id(); mesh.vertex_count()])
    }

    /// Upload `mesh` with a class per vertex (a triangle takes its first vertex's class).
    pub fn with_classes(ctx: &GpuContext, mesh: &MeshData, classes: &[u8]) -> Self {
        assert_eq!(classes.len(), mesh.vertex_count(), "one class per vertex");
        let vertices: Vec<Vertex> = (0..mesh.vertex_count())
            .map(|i| Vertex {
                position: mesh.positions[i],
                normal: mesh.normals[i],
                color: mesh.colors[i],
                class: classes[i] as u32,
            })
            .collect();
        let vertices = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self { vertices, indices, index_count: mesh.indices.len() as u32 }
    }

    pub fn triangle_count(&self) -> u32 {
        self.index_count / 3
    }
}

/// A mesh placed in the world: scaled by `scale` in its own frame, then `rotation` and
/// `position` take it into ENU.
#[derive(Clone, Copy)]
pub struct Draw<'a> {
    pub mesh: &'a GpuMesh,
    pub position: DVec3,
    pub rotation: DQuat,
    pub scale: DVec3,
    /// Class of every pixel of this draw instead of the mesh's own (e.g. the camera's own
    /// vehicle).
    pub class: Option<SemanticClass>,
}

impl<'a> Draw<'a> {
    pub fn new(mesh: &'a GpuMesh, position: DVec3, rotation: DQuat) -> Self {
        Self { mesh, position, rotation, scale: DVec3::ONE, class: None }
    }

    pub fn with_scale(self, scale: DVec3) -> Self {
        Self { scale, ..self }
    }

    pub fn with_class(self, class: SemanticClass) -> Self {
        Self { class: Some(class), ..self }
    }

    /// A mesh already in world coordinates.
    pub fn world(mesh: &'a GpuMesh) -> Self {
        Self::new(mesh, DVec3::ZERO, DQuat::IDENTITY)
    }
}

/// Lighting: a sun and an ambient share, and the sky's colour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shading {
    /// Direction to the sun (ENU, need not be unit).
    pub sun: DVec3,
    /// Share of the light that reaches every surface regardless of its orientation.
    pub ambient: f64,
    /// Linear RGB where nothing is hit.
    pub sky: [f64; 3],
}

impl Default for Shading {
    fn default() -> Self {
        Self { sun: DVec3::new(0.4, -0.3, 0.87), ambient: 0.35, sky: [0.45, 0.62, 0.85] }
    }
}

/// What one camera sees from where.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub pose: CameraPose,
    pub intrinsics: Intrinsics,
    pub shading: Shading,
}

impl View {
    /// Whether anything inside the box `[min, max]` (ENU) may be in view: false only when all
    /// its corners lie outside one plane of the view frustum.
    pub fn may_see(&self, min: DVec3, max: DVec3) -> bool {
        let k = &self.intrinsics;
        let corners = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| {
            let p = DVec3::new(
                if i & 1 == 0 { min.x } else { max.x },
                if i & 2 == 0 { min.y } else { max.y },
                if i & 4 == 0 { min.z } else { max.z },
            );
            self.pose.to_camera(p)
        });
        let (tx, ty) = (0.5 * k.width as f64 / k.focal(), 0.5 * k.height as f64 / k.focal());
        let planes: [&dyn Fn(DVec3) -> f64; 6] = [
            &|c| c.x - k.near,
            &|c| k.far - c.x,
            &|c| c.x * tx - c.y,
            &|c| c.x * tx + c.y,
            &|c| c.x * ty - c.z,
            &|c| c.x * ty + c.z,
        ];
        !planes.iter().any(|plane| corners.iter().all(|&c| plane(c) < 0.0))
    }
}

/// A rendered frame, rows top to bottom.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// sRGB-encoded RGB, 3 bytes per pixel.
    pub rgb: Vec<u8>,
    /// Depth along the optical axis (m); 0 where nothing was hit within the far plane.
    pub depth: Vec<f32>,
    /// [`SemanticClass`] ids.
    pub class: Vec<u8>,
}

impl Frame {
    pub fn index(&self, u: u32, v: u32) -> usize {
        (v * self.width + u) as usize
    }

    pub fn rgb_at(&self, u: u32, v: u32) -> [u8; 3] {
        let i = 3 * self.index(u, v);
        [self.rgb[i], self.rgb[i + 1], self.rgb[i + 2]]
    }

    pub fn depth_at(&self, u: u32, v: u32) -> f32 {
        self.depth[self.index(u, v)]
    }

    pub fn class_at(&self, u: u32, v: u32) -> u8 {
        self.class[self.index(u, v)]
    }
}

/// Render targets of one image size, with their read-back buffers.
struct Targets {
    width: u32,
    height: u32,
    color: wgpu::Texture,
    depth: wgpu::Texture,
    class: wgpu::Texture,
    z: wgpu::Texture,
    /// (buffer, bytes per pixel, padded bytes per row) per colour target.
    readback: [(wgpu::Buffer, u32, u32); 3],
}

impl Targets {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        let texture = |label, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let out = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        let readback = |bytes: u32| {
            let row = (width * bytes).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: (row * height) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            (buffer, bytes, row)
        };
        Self {
            width,
            height,
            color: texture("color", COLOR_FORMAT, out),
            depth: texture("depth", DEPTH_FORMAT, out),
            class: texture("class", CLASS_FORMAT, out),
            z: texture("z", Z_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT),
            readback: [readback(4), readback(4), readback(1)],
        }
    }
}

/// The camera pass: one pipeline, per-draw uniforms and targets sized to the last image.
pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    capacity: u64,
    targets: Option<Targets>,
}

impl Renderer {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("camera shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("draw"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(size_of::<DrawUniform>() as u64),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("camera"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Uint32];
        let target = |format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("camera"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attributes,
                }],
            },
            // Counter-clockwise in normalised device coordinates is counter-clockwise as the
            // camera sees it (x right, y up), i.e. the mesh convention.
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            // Reversed z: 1 at the near plane, 0 at the far plane.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Z_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[target(COLOR_FORMAT), target(DEPTH_FORMAT), target(CLASS_FORMAT)],
            }),
            multiview_mask: None,
            cache: None,
        });
        let (uniforms, bind_group) = Self::uniform_buffer(device, &layout, 64);
        Self { pipeline, layout, uniforms, bind_group, capacity: 64, targets: None }
    }

    fn uniform_buffer(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        draws: u64,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draw uniforms"),
            size: draws * DRAW_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draw"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(size_of::<DrawUniform>() as u64),
                }),
            }],
        });
        (buffer, bind_group)
    }

    /// Render `draws` as `view` sees them and read the images back.
    pub fn render(&mut self, ctx: &GpuContext, view: &View, draws: &[Draw<'_>]) -> Result<Frame, RenderError> {
        let device = &ctx.device;
        let k = &view.intrinsics;
        if self.targets.as_ref().is_none_or(|t| (t.width, t.height) != (k.width, k.height)) {
            self.targets = Some(Targets::new(device, k.width, k.height));
        }
        if draws.len() as u64 > self.capacity {
            self.capacity = (draws.len() as u64).next_power_of_two();
            (self.uniforms, self.bind_group) = Self::uniform_buffer(device, &self.layout, self.capacity);
        }
        // Per-draw uniforms, composed in f64 relative to the camera.
        let projection = projection(k);
        let to_camera = view.pose.orientation.inverse();
        let sun = view.shading.sun.normalize_or_zero();
        let mut bytes = vec![0u8; draws.len() * DRAW_STRIDE as usize];
        for (i, d) in draws.iter().enumerate() {
            let model_view = DMat4::from_scale_rotation_translation(
                d.scale,
                to_camera * d.rotation,
                to_camera * (d.position - view.pose.position),
            );
            let local_sun = d.rotation.inverse() * sun;
            let u = DrawUniform {
                mvp: (projection * model_view).as_mat4().to_cols_array_2d(),
                sun: [local_sun.x as f32, local_sun.y as f32, local_sun.z as f32, view.shading.ambient as f32],
                scale: [d.scale.x as f32, d.scale.y as f32, d.scale.z as f32, 1.0],
                class: [d.class.map_or(NO_CLASS, |c| c.id() as u32), 0, 0, 0],
            };
            let at = i * DRAW_STRIDE as usize;
            bytes[at..at + size_of::<DrawUniform>()].copy_from_slice(bytemuck::bytes_of(&u));
        }
        if !bytes.is_empty() {
            ctx.queue.write_buffer(&self.uniforms, 0, &bytes);
        }
        let t = self.targets.as_ref().expect("targets were just made");
        let views = [&t.color, &t.depth, &t.class].map(|x| x.create_view(&Default::default()));
        let z_view = t.z.create_view(&Default::default());
        let [r, g, b] = view.shading.sky;
        let clear = |c: wgpu::Color| wgpu::Operations { load: wgpu::LoadOp::Clear(c), store: wgpu::StoreOp::Store };
        let attachment = |view, c| {
            Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: clear(c) })
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("camera") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("camera"),
                color_attachments: &[
                    attachment(&views[0], wgpu::Color { r, g, b, a: 1.0 }),
                    attachment(&views[1], wgpu::Color::TRANSPARENT),
                    attachment(&views[2], wgpu::Color::TRANSPARENT),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &z_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            for (i, d) in draws.iter().enumerate() {
                pass.set_bind_group(0, &self.bind_group, &[(i as u64 * DRAW_STRIDE) as u32]);
                pass.set_vertex_buffer(0, d.mesh.vertices.slice(..));
                pass.set_index_buffer(d.mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..d.mesh.index_count, 0, 0..1);
            }
        }
        let size = wgpu::Extent3d { width: t.width, height: t.height, depth_or_array_layers: 1 };
        for (texture, (buffer, _, row)) in [&t.color, &t.depth, &t.class].into_iter().zip(&t.readback) {
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(*row), rows_per_image: None },
                },
                size,
            );
        }
        ctx.queue.submit([encoder.finish()]);
        for (buffer, _, _) in &t.readback {
            buffer.map_async(wgpu::MapMode::Read, .., |r| r.expect("mapping a readback buffer"));
        }
        ctx.wait()?;
        let unpad = |(buffer, bytes, row): &(wgpu::Buffer, u32, u32)| {
            let data = buffer.get_mapped_range(..);
            let line = (t.width * bytes) as usize;
            let mut out = Vec::with_capacity(line * t.height as usize);
            for y in 0..t.height as usize {
                let at = y * *row as usize;
                out.extend_from_slice(&data[at..at + line]);
            }
            drop(data);
            buffer.unmap();
            out
        };
        let rgba = unpad(&t.readback[0]);
        let depth = unpad(&t.readback[1]);
        let class = unpad(&t.readback[2]);
        Ok(Frame {
            width: t.width,
            height: t.height,
            rgb: rgba.as_chunks::<4>().0.iter().flat_map(|&[r, g, b, _]| [r, g, b]).collect(),
            depth: depth.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect(),
            class,
        })
    }
}

/// Camera frame (FLU, optical axis +x) → clip space: x right, y up, reversed z (1 at the near
/// plane, 0 at the far plane), w the depth along the axis.
fn projection(k: &Intrinsics) -> DMat4 {
    let f = k.focal();
    let sx = f / (0.5 * k.width as f64);
    let sy = f / (0.5 * k.height as f64);
    let (n, far) = (k.near, k.far);
    DMat4::from_cols(
        DVec4::new(0.0, 0.0, -n / (far - n), 1.0),
        DVec4::new(-sx, 0.0, 0.0, 0.0),
        DVec4::new(0.0, sy, 0.0, 0.0),
        DVec4::new(0.0, 0.0, n * far / (far - n), 0.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_maps_the_frustum() {
        let k = Intrinsics { near: 0.5, far: 100.0, ..Intrinsics::new(64, 48, 90f64.to_radians()) };
        let ndc = |p: DVec3| {
            let c = projection(&k) * p.extend(1.0);
            (c.truncate() / c.w, c.w)
        };
        // Near plane → z 1, far plane → z 0; w is the axis depth.
        assert!((ndc(DVec3::new(0.5, 0.0, 0.0)).0.z - 1.0).abs() < 1e-12);
        assert!(ndc(DVec3::new(100.0, 0.0, 0.0)).0.z.abs() < 1e-12);
        assert_eq!(ndc(DVec3::new(7.0, 1.0, 2.0)).1, 7.0);
        // A point on the ray through pixel (u, v) lands at that pixel.
        for (u, v) in [(0.0, 0.0), (64.0, 48.0), (10.5, 30.25)] {
            let (p, _) = ndc(k.ray(u, v) * 13.0);
            let (pu, pv) = ((p.x + 1.0) * 32.0, (1.0 - p.y) * 24.0);
            assert!((pu - u).abs() < 1e-9 && (pv - v).abs() < 1e-9, "{u} {v} → {pu} {pv}");
        }
    }
}
