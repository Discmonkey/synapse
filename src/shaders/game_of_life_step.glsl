#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform image2D img;
layout(set = 1, binding = 0, rgba8) uniform writeonly image2D workspace;

void main() {
    vec2 location = gl_GlobalInvocationID.xy;
    vec2 img_size = vec2(imageSize(img));
    vec4 new_pixel_value = vec4(vec3(0), 1.0);
    ivec2 ilocation = ivec2(gl_GlobalInvocationID.xy);

    if (!(location.x == 0 || location.x == img_size.x || location.y == 0 || location.y == img_size.y)) {
        ivec3 neighbors = ivec3(0);

        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                if (i == 0 && j == 0) {
                    continue;
                }

                ivec2 new_location = ilocation;
                new_location.x += i;
                new_location.y += j;

                vec4 pixel = imageLoad(img, new_location);

                for (int i = 0; i < 3; i++) {
                    if (pixel[i] > 0.0) {
                        neighbors[i] += 1;
                    }
                }
            }
        }

        vec4 pixel = imageLoad(img, ilocation);

        for (int i = 0; i < 3; i++) {
            bool alive = pixel[i] > 0.0;
            int alive_neighbors = neighbors[i];

            if (alive && (alive_neighbors == 2 || alive_neighbors == 3)) {
                new_pixel_value[i] = 1.0;
            } else if (!alive && alive_neighbors == 3) {
                new_pixel_value[i] = 1.0;
            }

        }

    }

    imageStore(workspace, ilocation, new_pixel_value);
}