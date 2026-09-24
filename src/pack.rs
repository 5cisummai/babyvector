pub fn stride_bytes(dim: usize) -> usize {
    let bytes = (dim + 7) / 8;
    ((bytes + 7) / 8) * 8
}

pub fn words_per_row(stride_bytes: usize) -> usize {
    stride_bytes / 8
}

pub fn pack_1bit_row(dim: usize, vector: &[f32], out: &mut [u8]) {
    out.fill(0);
    for i in 0..dim {
        if vector[i] >= 0.0 {
            set_bit(out, i);
        }
    }
}

pub fn pack_1bit_query(dim: usize, query: &[f32], out: &mut [u8]) {
    pack_1bit_row(dim, query, out);
}

fn set_bit(out: &mut [u8], bit_index: usize) {
    let byte = bit_index / 8;
    let bit = bit_index % 8;
    out[byte] |= 1 << bit;
}

#[cfg(test)]
pub(crate) fn get_bit(row: &[u8], bit_index: usize) -> bool {
    let byte = bit_index / 8;
    let bit = bit_index % 8;
    (row[byte] >> bit) & 1 == 1
}
