use vulkano::format::{Format, ClearValue};
use vulkano::image::Dimensions;
use vulkano::image::StorageImage;
use vulkano::instance::{Instance, PhysicalDevice, InstanceExtensions};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::sync::GpuFuture;
use image::{ImageBuffer, Rgba};
use std::sync::Arc;
use vulkano::pipeline::ComputePipeline;
use vulkano::descriptor::PipelineLayoutAbstract;

mod cs {
    vulkano_shaders::shader!{
        ty: "compute",
        src: "
            #version 450

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

            layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

            void main() {
                vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
                vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

                vec2 z = vec2(0.0, 0.0);
                float i;
                for (i = 0.0; i < 1.0; i += 0.005) {
                    z = vec2(
                        z.x * z.x - z.y * z.y + c.x,
                        z.y * z.x + z.x * z.y + c.y
                    );

                    if (length(z) > 3000) {
                        break;
                    }
                }

                vec4 to_write = vec4(vec3(i) * vec3(1, c), 1.0);
                imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
            }

            "
        }
}


pub fn main() {

    // instance is just the "context" for vulkan
    let instance = Instance::new(None, &InstanceExtensions::none(), None)
        .expect("failed to create instance");

    // here i just grab the first device that supports vulkan
    // which is in theory the device
    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");

    // print out the queues available
    for family in physical.queue_families() {
        println!("Found a queue family with {:?} queue(s)", family.queues_count());
    }

    // here I just select the the family that supports the most queues
    // note I am not sure about the performance differences, for example by gpu has 3 different queue families
    // with various queue counts, how do queues relate to processors?
    let queue_family = physical.queue_families()
        .filter(|&q| q.supports_compute()) // only want a family that supports compute
        .max_by(| &a, &b| {
            a.queues_count().cmp(&b.queues_count()) // get the family with the larges number of queues
        })
        .expect("couldn't find a graphical queue family");

    // this device is different from the physical device since its a vulkan device
    // the queues are also I guess "vulkan" queues
    let (device, mut queues) = Device::new(physical, &Features::none(),
                    &DeviceExtensions{khr_storage_buffer_storage_class:true, ..DeviceExtensions::none()},
                    [(queue_family, 1.0)].iter().cloned()).expect("failed to create device");

    // so we got back a list of queues, but we just want one queue to executue on
    // again i have no idea what the advantage of multiple queues are who the f knows
    let queue = queues.next().unwrap();

    // alright now we create an image, we do so on the device, and we also give it a format
    let image = StorageImage::new(device.clone(), Dimensions::Dim2d { width: 1024, height: 1024 },
                                  Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();

    // load in the shader, our shader does the computation of work, in this case it's a simple enough in/out
    // now why does it need the vulkan device? who knows.
    let shader = cs::Shader::load(device.clone())
        .expect("failed to create shader module");


    // create the pipeline, notice that it doesn't need the queue, just the vulkan device
    let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
        .expect("failed to create compute pipeline"));

    // the layout lets us "load" something into the pipeline
    // in this case we're only loading one thing...
    let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();

    // well meow, we add the image into the pipeline
    let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
        .add_image(image.clone()).unwrap()
        .build().unwrap()
    );

    // this what we copy out
    let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false,
                                             (0 .. 1024 * 1024 * 4).map(|_| 0u8))
        .expect("failed to create buffer");

    // great now we create a command buffer, in this case we "dispatch" the pipeline
    // and also so the first parameter tod ispatch is the big mystery, wtf is it doing?
    // is that the number of "work groups needed"?
    let command_buffer = AutoCommandBufferBuilder::new(device.clone(),
                                                       queue.family()).unwrap()
        .dispatch([1024 / 8, 1024 / 8, 1],
                  compute_pipeline.clone(), set.clone(), ()).unwrap()

        .copy_image_to_buffer(image.clone(), buf.clone()).unwrap()

        .build().unwrap();

    // easy enough get a handle to the pipeline
    let finished = command_buffer.execute(queue.clone()).unwrap();

    // alright so now we're not doing anything else, hence we wait for this thing to wrap up
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    // and then actually get stuff back to the cpu
    let buffer_content = buf.read().unwrap();

    // oh yeah, why not also load that into a "cv2" like iamge library
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();

    // and save it as a png
    image.save("image3000.png").unwrap();

}