use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::GpuFuture;
use std::thread::sleep;
use std::time;
use std::sync::Arc;
use vulkano::image::ImageViewAccess;
use vulkano::descriptor::PipelineLayoutAbstract;


macro_rules! prn {
    ($item:expr) => {
        println!("{}", $item)
    }
}



pub fn main() {
    let instance = Instance::new(None, &InstanceExtensions::none(), None)
        .expect("failed to create instance");

    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");

    for family in physical.queue_families() {
        println!("Found a queue family with {:?} queue(s)", family.queues_count());
    }
    let queue_family = physical.queue_families()
        .filter(|&q| q.supports_compute()) // only want a family that supports compute
        .max_by(| &a, &b| {
            a.queues_count().cmp(&b.queues_count()) // get the family with the larges number of queues
        })
        .expect("couldn't find a graphical queue family");

    let (device, mut queues) = {
        Device::new(physical, &Features::none(),  &DeviceExtensions{khr_storage_buffer_storage_class:true, ..DeviceExtensions::none()},
                    [(queue_family, 0.5)].iter().cloned()).expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    let data_iter = 0 .. 65536;
    let data_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false,
                                                     data_iter.clone()).expect("failed to create buffer");

    let data_buffer2 = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false,
                                                      data_iter).expect("failed to create buffer");
    mod cs {
        vulkano_shaders::shader!{
        ty: "compute",
        src: "
            #version 450

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Source {
                uint data[];
            } buffer_a;

            layout(set = 1, binding = 0) buffer Target {
                uint data[];
            } buffer_b;

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                buffer_b.data[idx] += buffer_a.data[idx];
            }
            "
        }
    }

    let shader = cs::Shader::load(device.clone())
        .expect("failed to create shader module");

    let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
        .expect("failed to create compute pipeline"));


    let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();
    let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
        .add_buffer(data_buffer.clone()).unwrap()
        .build().unwrap()
    );

    let layout2 =  compute_pipeline.layout().descriptor_set_layout(1).unwrap();
    let set2 = Arc::new(PersistentDescriptorSet::start(layout2.clone())
        .add_buffer(data_buffer2.clone()).unwrap()
        .build().unwrap()
    );


    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
        .dispatch([1024, 1, 1], compute_pipeline.clone(), vec!(set.clone(), set2.clone()), ()).unwrap()
        .build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();

    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let content = data_buffer2.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 2);
    }

    println!("Everything succeeded!");



}