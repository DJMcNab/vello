use std::{
    fs::File,
    num::NonZeroU32,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, bail, Result};
use clap::Parser;
use vello::{block_on_wgpu, util::RenderContext, Scene};
use wgpu::{
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, ImageCopyBuffer,
    TextureDescriptor, TextureFormat, TextureUsages,
};

fn main() -> Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let args = Args::parse();

    pollster::block_on(render(scenes::gen_test_scene(), &args))?;

    Ok(())
}

async fn render(scene: Scene, args: &Args) -> Result<()> {
    let mut context = RenderContext::new()
        .or_else(|_| bail!("Got non-Send/Sync error from creating render context"))?;
    let device_id = context
        .device(None)
        .await
        .ok_or_else(|| anyhow!("No compatible device found"))?;
    let device_handle = &mut context.devices[device_id];
    let device = &device_handle.device;
    let queue = &device_handle.queue;
    let mut renderer = vello::Renderer::new(&device)
        .or_else(|_| bail!("Got non-Send/Sync error from creating renderer"))?;
    let width = 1000;
    let height = 1000;
    let size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let target = device.create_texture(&TextureDescriptor {
        label: Some("Target texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    renderer
        .render_to_texture(&device, &queue, &scene, &view, width, height)
        .or_else(|_| bail!("Got non-Send/Sync error from rendering"))?;
    // (width * 4).next_multiple_of(256)
    let padded_byte_width = {
        let w = width as u32 * 4;
        match w % 256 {
            0 => w,
            r => w + (256 - r),
        }
    };
    let buffer_size = padded_byte_width as u64 * height as u64;
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("val"),
        size: buffer_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Copy out buffer"),
    });
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        ImageCopyBuffer {
            buffer: &buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(padded_byte_width),
                rows_per_image: None,
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);
    let buf_slice = buffer.slice(..);

    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buf_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    if let Some(recv_result) = block_on_wgpu(&device, receiver.receive()) {
        recv_result?;
    } else {
        bail!("channel was closed");
    }

    let data = buf_slice.get_mapped_range();
    let mut result_unpadded = Vec::<u8>::with_capacity((width * height * 4).try_into()?);
    for row in 0..height {
        let start = (row * padded_byte_width).try_into()?;
        result_unpadded.extend(&data[start..start + (width * 4) as usize]);
    }
    let out_path = args.out_directory.join("simple_test").with_extension("png");
    let mut file = File::create(&out_path)?;
    let mut encoder = png::Encoder::new(&mut file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&result_unpadded)?;
    writer.finish()?;
    println!("Wrote result ({width}x{height}) to {out_path:?}");
    Ok(())
}

#[derive(Parser, Debug)]
#[command(about, long_about = None, bin_name="cargo run -p headless --")]
struct Args {
    /// Directory to store the result into
    #[arg(long, default_value_os_t = default_directory())]
    pub out_directory: PathBuf,
}

fn default_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("outputs")
}
