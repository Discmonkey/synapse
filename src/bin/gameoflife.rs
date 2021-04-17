use vulkano::format::{Format, ClearValue};
use vulkano::image::ImageDimensions as Dimensions;
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
use vulkano::descriptor::{PipelineLayoutAbstract, DescriptorSet};
use std::time::SystemTime;
use vulkano::descriptor::descriptor::DescriptorBufferDesc;
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::WindowBuilder;
use vulkano_win::VkSurfaceBuild;
use winit::dpi::{Size, PhysicalSize};
use vulkano::swapchain::{Swapchain, FullscreenExclusive, ColorSpace, SurfaceTransform, PresentMode};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
        #version 450

        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

        layout(set = 0, binding = 0, rgba8) uniform image2D img;
        layout(set = 1, binding = 0, rgba8) uniform writeonly image2D workspace;

        void main() {
            vec2 location = gl_GlobalInvocationID.xy;
            vec2 img_size = vec2(imageSize(img));
            vec4 new_pixel_value = vec4(vec3(0.0), 1.0);
            ivec2 ilocation = ivec2(gl_GlobalInvocationID.xy);

            vec4 old_pixel_value = imageLoad(img, ilocation);

            int neighbors = 0;
            for (int i = -1; i < 2; i++) {
                for (int j = -1; j < 2; j++) {
                    if (i == 0 && j == 0) {
                        continue;
                    }

                    ivec2 new_location = ilocation;
                    new_location.x += i;
                    new_location.y += j;

                    vec4 pixel = imageLoad(img, new_location);

                    if (pixel.x > 0.0) {
                        neighbors += 1;
                    }
                }
            }

            bool alive = old_pixel_value.x > 0.0;

            if (alive && (neighbors == 2 || neighbors == 3)) {
                new_pixel_value.xyz = vec3(1.0);
            } else if (!alive && neighbors == 3) {
                new_pixel_value.xyz = vec3(1.0);
            }

            imageStore(workspace, ilocation, new_pixel_value);
        }
        "
    }
}

mod rand {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "

        #version 450

        // given an input image, assigns random, integer values to the rgb channels (the a channel defaults to 1)
        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

        // our actual image we are randomizing,
        layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

        float rand(vec2 co){
          return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
        }

        void main() {
            ivec2 loc = ivec2(gl_GlobalInvocationID.xy);

            // pass these in as random seeds in the future
            float chance = rand(vec2(loc));

            vec4 pixel = vec4(vec3(0.0), 1.0);

            if (chance > .9) {
                pixel.x = 1.0;
                pixel.y = 1.0;
                pixel.z = 1.0;
            }

            imageStore(img, loc, pixel);
        }

"
    }
}

fn seconds() -> f32 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs_f32()
}

static WIDTH: u32 = 1280;
static HEIGHT: u32 = 720;

pub fn main() {
    let required_extensions = vulkano_win::required_extensions();
    // instance is just the "context" for vulkan
    let instance = Instance::new(None, &required_extensions, None)
        .expect("failed to create instance");

    // here i just grab the first device that supports vulkan
    // which is in theory the device
    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");

    let event_loop = EventLoop::new();

    let window_size = Size::Physical(PhysicalSize::new(WIDTH, HEIGHT));
    let surface = WindowBuilder::new()
        .with_inner_size(window_size)
        .build(&event_loop, instance.clone()).unwrap();

    // here I just select the the family that supports the most queues
    // note I am not sure about the performance differences, for example by gpu has 3 different queue families
    // with various queue counts, how do queues relate to processors?
    let queue_family = physical.queue_families()
        .filter(|&q| q.supports_compute() && surface.is_supported(q).unwrap_or(false)) // only want a family that supports compute
        .max_by(| &a, &b| {
            a.queues_count().cmp(&b.queues_count()) // get the family with the largest number of queues
        })
        .expect("couldn't find a graphical queue family");

    // this device is different from the physical device since its a vulkan device
    // the queues are also I guess "vulkan" queues
    let (device, mut queues) = Device::new(physical, &Features::none(),
                                           &DeviceExtensions{khr_storage_buffer_storage_class:true, khr_swapchain:true, ..DeviceExtensions::none()},
                                           [(queue_family, 1.0)].iter().cloned()).expect("failed to create device");

    // so we got back a list of queues, but we just want one queue to executue on
    // again i have no idea what the advantage of multiple queues are who the f knows
    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format, dimensions,
                       1, usage, &queue, SurfaceTransform::Identity, alpha, PresentMode::Fifo,
                       FullscreenExclusive::Default, true, ColorSpace::SrgbNonLinear).unwrap()
    };

    // load in the shader, our shader does the computation of work, in this case it's a simple enough in/out
    // now why does it need the vulkan device? who knows.
    let shader = rand::Shader::load(device.clone())
        .expect("failed to create shader module");


    // create the pipeline, notice that it doesn't need the queue, just the vulkan device
    let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
        .expect("failed to create compute pipeline"));

    // the layout lets us "load" something into the pipeline
    // in this case we're only loading one thing...
    let layout_image = compute_pipeline.layout().descriptor_set_layout(0).unwrap();

    // we add the image into the pipeline
    let set_image = Arc::new(PersistentDescriptorSet::start(layout_image.clone())
        .add_image(images[0].clone()).unwrap()
        .build().unwrap()
    );

    // // this what we copy out
    let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false,
                                             (0 .. HEIGHT * WIDTH * 4).map(|_| 0u8))
        .expect("failed to create buffer");

    // create a random image here
    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
        .dispatch([WIDTH / 8, HEIGHT / 8, 1], compute_pipeline.clone(),
                  set_image, ()).unwrap()
        .copy_image_to_buffer(images[0].clone(), buf.clone()).unwrap()
        .build().unwrap();

    // easy enough get a handle to the pipeline
    let finished = command_buffer.execute(queue.clone()).unwrap();

    // alright so now we're not doing anything else, hence we wait for this thing to wrap up
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    // and then actually get stuff back to the cpu
    let buffer_content = buf.read().unwrap();

    // oh yeah, why not also load that into a "cv2" like iamge library
    let random_image = ImageBuffer::<Rgba<u8>, _>::from_raw(WIDTH, HEIGHT, &buffer_content[..]).unwrap();

    // and save it as a png
    random_image.save("random.png").unwrap();

    // okay now we're going to run a single iteration of the "game of life" on it...
    let game_of_life_shader = cs::Shader::load(device.clone())
        .expect("failed to create shader module");


    // create the pipeline, notice that it doesn't need the queue, just the vulkan device
    let game_of_life_pipeline = Arc::new(ComputePipeline::new(
        device.clone(), &game_of_life_shader.main_entry_point(), &())
        .expect("failed to create compute pipeline"));

    // the layout lets us "load" something into the pipeline
    // in this case we're only loading one thing...
    let input_image = game_of_life_pipeline.layout().descriptor_set_layout(0).unwrap();
    let output_image = game_of_life_pipeline.layout().descriptor_set_layout(1).unwrap();
    // we add the image into the pipeline
    let from_image = Arc::new(PersistentDescriptorSet::start(input_image.clone())
        .add_image(images[0].clone()).unwrap()
        .build().unwrap()
    );

    let to_image = Arc::new(PersistentDescriptorSet::start(output_image.clone())
        .add_image(images[1].clone()).unwrap()
        .build().unwrap()
    );

    // // this what we copy out
    let final_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false,
                                             (0 .. WIDTH * HEIGHT * 4).map(|_| 0u8))
        .expect("failed to create buffer");

    let command_buffer_step = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
        .dispatch([WIDTH / 8, HEIGHT / 8, 1], game_of_life_pipeline.clone(),
                  vec!(from_image, to_image), ()).unwrap()
        .copy_image_to_buffer(images[1].clone(), final_buffer.clone()).unwrap()
        .build().unwrap();

    let step_finished = command_buffer_step.execute(queue.clone()).unwrap();

    // alright so now we're not doing anything else, hence we wait for this thing to wrap up
    step_finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    // and then actually get stuff back to the cpu
    let step_content = final_buffer.read().unwrap();

    // oh yeah, why not also load that into a "cv2" like image library
    let step_image = ImageBuffer::<Rgba<u8>, _>::from_raw(WIDTH, HEIGHT, &step_content[..]).unwrap();

    // and save it as a png
    step_image.save("step.png").unwrap();
}