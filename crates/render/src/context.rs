//! The GPU: adapter choice, device and queue.

use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

use thiserror::Error;

/// Environment variable that picks the adapter ([`AdapterChoice::from_env`]).
pub const ADAPTER_ENV: &str = "AUTONOMOUSIM_RENDER_ADAPTER";

#[derive(Debug, Error)]
pub enum RenderError {
    #[error("no suitable GPU adapter ({0}); available: {1}")]
    NoAdapter(String, String),
    #[error("requesting the device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error("waiting for the GPU: {0}")]
    Poll(#[from] wgpu::PollError),
    #[error("reading back an image: {0}")]
    Map(#[from] wgpu::BufferAsyncError),
    #[error("not supported yet: {0}")]
    Unsupported(String),
}

/// Which adapter renders.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum AdapterChoice {
    /// The fastest there is: a discrete GPU, then an integrated one, then the software
    /// rasterizer.
    #[default]
    Auto,
    /// The software rasterizer (lavapipe): slow, but the same images on every machine.
    Software,
    /// The first adapter whose name contains this (case-insensitive).
    Named(String),
}

impl AdapterChoice {
    /// From [`ADAPTER_ENV`]: unset, empty or `auto` → [`Auto`](Self::Auto); `software`,
    /// `lavapipe` or `cpu` → [`Software`](Self::Software); anything else names an adapter.
    pub fn from_env() -> Self {
        match std::env::var(ADAPTER_ENV) {
            Ok(v) => Self::parse(&v),
            Err(_) => Self::Auto,
        }
    }

    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Self::Auto,
            "software" | "lavapipe" | "cpu" => Self::Software,
            name => Self::Named(name.to_string()),
        }
    }

    fn rank(&self, info: &wgpu::AdapterInfo) -> Option<u8> {
        use wgpu::DeviceType as T;
        match self {
            Self::Auto => Some(match info.device_type {
                T::DiscreteGpu => 0,
                T::IntegratedGpu => 1,
                T::VirtualGpu => 2,
                T::Cpu => 3,
                T::Other => 4,
            }),
            Self::Software => (info.device_type == T::Cpu).then_some(0),
            Self::Named(name) => info.name.to_ascii_lowercase().contains(name.as_str()).then_some(0),
        }
    }
}

/// A headless device and its queue. One per process is enough: renderers share it.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
}

impl GpuContext {
    pub fn new(choice: &AdapterChoice) -> Result<Self, RenderError> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN;
        // No debug labels or validation by default, even in debug builds: naming objects through
        // the debug-utils extension crashed the Vulkan loader now and then when two contexts
        // were created in parallel (tests). `WGPU_DEBUG=1` / `WGPU_VALIDATION=1` turn them on.
        desc.flags = wgpu::InstanceFlags::empty().with_env();
        let instance = wgpu::Instance::new(desc);
        let adapters = block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        // The best rank wins; ties keep the enumeration order.
        let adapter = adapters
            .iter()
            .filter_map(|a| choice.rank(&a.get_info()).map(|r| (r, a)))
            .min_by_key(|(r, _)| *r)
            .map(|(_, a)| a.clone());
        let Some(adapter) = adapter else {
            let names: Vec<String> =
                adapters.iter().map(|a| format!("{} ({:?})", a.get_info().name, a.get_info().device_type)).collect();
            return Err(RenderError::NoAdapter(format!("{choice:?}"), names.join(", ")));
        };
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("autonomousim-render"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))?;
        Ok(Self { device, queue, info: adapter.get_info() })
    }

    /// The adapter from [`ADAPTER_ENV`].
    pub fn from_env() -> Result<Self, RenderError> {
        Self::new(&AdapterChoice::from_env())
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    /// Adapter name and driver, e.g. for benchmark results and determinism keys.
    pub fn describe(&self) -> String {
        format!("{} ({:?}, {} {})", self.info.name, self.info.device_type, self.info.driver, self.info.driver_info)
    }

    /// Whether this is the software rasterizer.
    pub fn is_software(&self) -> bool {
        self.info.device_type == wgpu::DeviceType::Cpu
    }

    /// Block until the queue's work is done.
    pub fn wait(&self) -> Result<(), RenderError> {
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        Ok(())
    }
}

/// Drive a future to completion on this thread. wgpu's native futures are ready at once, or
/// once the device has been polled.
pub(crate) fn block_on<F: Future>(f: F) -> F::Output {
    let mut f = pin!(f);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_choices_parse() {
        assert_eq!(AdapterChoice::parse(""), AdapterChoice::Auto);
        assert_eq!(AdapterChoice::parse(" Lavapipe "), AdapterChoice::Software);
        assert_eq!(AdapterChoice::parse("CPU"), AdapterChoice::Software);
        assert_eq!(AdapterChoice::parse("Iris"), AdapterChoice::Named("iris".into()));
    }
}
