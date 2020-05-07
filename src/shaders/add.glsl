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