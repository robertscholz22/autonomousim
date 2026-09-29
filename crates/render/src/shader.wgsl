// One pass, two targets: shaded colour with the class id in alpha (alpha is stored linearly,
// so id / 255 comes back as the id), and linear depth along the optical axis.

struct DrawUniform {
    // Mesh-local position → clip space (camera-relative, composed in f64 on the CPU).
    mvp: mat4x4<f32>,
    // Direction to the sun in the mesh's frame (xyz) and the ambient share (w).
    sun: vec4<f32>,
    // Scale of the mesh in its own frame (xyz).
    scale: vec4<f32>,
    // x: class of every pixel of the draw, or 0xffffffff for the mesh's own.
    class_id: vec4<u32>,
};

@group(0) @binding(0) var<uniform> draw: DrawUniform;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) class_id: u32,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    // Clip w is the camera-frame x: the depth along the optical axis (linear in space, so
    // perspective-correct interpolation is exact).
    @location(2) depth: f32,
    @location(3) @interpolate(flat) class_id: u32,
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip = draw.mvp * vec4<f32>(v.position, 1.0);
    // Normals of a scaled mesh scale inversely.
    out.normal = v.normal / draw.scale.xyz;
    out.color = v.color;
    out.depth = out.clip.w;
    out.class_id = select(draw.class_id.x, v.class_id, draw.class_id.x == 0xffffffffu);
    return out;
}

struct FragmentOut {
    @location(0) color: vec4<f32>,
    @location(1) depth: f32,
};

@fragment
fn fs_main(in: VertexOut, @builtin(front_facing) front: bool) -> FragmentOut {
    // Two-sided: a back face is lit as seen from its own side.
    var n = normalize(in.normal);
    if (!front) {
        n = -n;
    }
    let ambient = draw.sun.w;
    let light = ambient + (1.0 - ambient) * max(dot(n, draw.sun.xyz), 0.0);
    var out: FragmentOut;
    out.color = vec4<f32>(in.color.rgb * light, f32(in.class_id) / 255.0);
    out.depth = in.depth;
    return out;
}
