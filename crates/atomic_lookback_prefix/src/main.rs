use std::{mem::size_of, path::Path, time::Instant};

use bytemuck::cast_slice;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, BufferDescriptor, BufferUsages, ComputePipelineDescriptor,
    DeviceDescriptor, Limits,
};
use wgpu_profiler::GpuProfiler;

async fn run() {
    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: Some("WGSL atomic lookback prefix sum proof of concept"),
                features: wgpu::Features::default()
                    | wgpu_profiler::GpuProfiler::ALL_WGPU_TIMER_FEATURES,
                limits: Limits::default(),
            },
            None,
        )
        .await
        .unwrap();
    let mut profiler = GpuProfiler::new(Default::default()).unwrap();
    let workgroup_size = 256;
    let num_workgroups = 10_000;
    let total_data = workgroup_size * num_workgroups;
    let input_data: Vec<u32> = (0..(total_data / 32)).flat_map(|_| 0..32).collect();
    let expected_total = input_data.iter().sum::<u32>();
    if expected_total & 1u32 << 31 != 0 {
        panic!("Expected total too big: {expected_total}");
    }
    let module = device.create_shader_module(wgpu::include_wgsl!("./prefix_sum.wgsl"));
    let start = Instant::now();
    let input_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Input buffer"),
        contents: cast_slice(&input_data),
        usage: BufferUsages::STORAGE,
    });

    let output_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Buffer for results"),
        size: total_data as u64 * size_of::<u32>() as u64,
        // COPY_SRC to allow reading back afterwards
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let aggregate_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Buffer for per-workgroup aggregates"),
        size: (num_workgroups * size_of::<u32>() as u32) as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let inclusive_prefix_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Buffer for per-workgroup inclusive prefixes"),
        size: (num_workgroups * size_of::<u32>() as u32) as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Buffer to load results"),
        size: total_data as u64 * size_of::<u32>() as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Prefix sum pipeline"),
        layout: None,
        module: &module,
        entry_point: "main",
    });
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: aggregate_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: inclusive_prefix_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        profiler.begin_scope("Compute", &mut pass, &device);
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
        profiler.end_scope(&mut pass).unwrap();
    }
    profiler.begin_scope("Copy buffer", &mut encoder, &device);
    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &staging_buffer,
        0,
        total_data as u64 * size_of::<u32>() as u64,
    );
    profiler.end_scope(&mut encoder).unwrap();

    profiler.resolve_queries(&mut encoder);
    queue.submit(Some(encoder.finish()));
    profiler.end_frame().unwrap();
    let slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::MaintainBase::Wait);
    let received = receiver.receive().await;
    if let Some(Ok(())) = received {
        println!("Getting data took {:?}", start.elapsed());
        let data = slice.get_mapped_range();
        let result: Vec<u32> = cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        let frame = profiler.process_finished_frame(queue.get_timestamp_period());
        wgpu_profiler::chrometrace::write_chrometrace(
            Path::new("chrome_trace.json"),
            &frame.unwrap(),
        )
        .unwrap();
        let checksum_start = Instant::now();
        let mut agg = 0;
        assert_eq!(input_data.len(), result.len());
        for (v, actual) in input_data.iter().zip(&result) {
            agg += v;
            assert_eq!(*actual, agg);
        }
        println!("Confirming took {:?}", checksum_start.elapsed());
    } else {
        panic!("Couldn't get data, got {received:?}")
    }
}

fn main() {
    pollster::block_on(run());
}
