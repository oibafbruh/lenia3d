@group(0) @binding(0) var<storage, read_write> state: array<u32>;

@compute @workgroup_size(1)
fn main() {
    state[0] = state[0] + 1u;
}
