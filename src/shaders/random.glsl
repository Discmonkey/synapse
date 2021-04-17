#version 450

// given an input image, assigns random, integer values to the rgb channels (the a channel defaults to 1)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// our actual image we are randomizing,
layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

void main() {
    ivec2 loc = ivec2(gl_GlobalInvocationID.xy);

    // pass these in as random seeds in the future
    float chance_x = .9; //fract(loc.x * 98384.23);
    float chance_y = .9; //fract(loc.y * 39844.45);
    float chance_z = .9; //frant((loc.x + loc.y) * 983444.22);

    vec4 pixel = vec4(vec3(0.0), 1.0);
//
//    if (chance_x > .8) {
//        pixel.x = 1.0;
//    }
//
//    if (chance_y > .8) {
//        pixel.y = 1.0;
//    }
//
//    if (chance_z > .8) {
//        pixel.z = 1.0;
//    }

    imageStore(img, loc, pixel);
}
