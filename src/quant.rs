use crate::pack::pack_1bit_row;

pub fn quantize_row(dim: usize, vector: &[f32], out: &mut [u8]) {
    pack_1bit_row(dim, vector, out);
}

#[cfg(test)]
pub(crate) fn naive_score_1bit(dim: usize, query: &[f32], row: &[u8]) -> f32 {
    let mut hamming = 0usize;
    for i in 0..dim {
        let q_pos = query[i] >= 0.0;
        let r_pos = crate::pack::get_bit(row, i);
        if q_pos != r_pos {
            hamming += 1;
        }
    }
    dim as f32 - 2.0 * hamming as f32
}
