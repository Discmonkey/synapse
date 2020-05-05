#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
    uint data_a[];
    uint data_b[];
} buffers;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    buffers.data_b[idx] = buffers.data_a[idx] + buffers.data_b[idx];
}