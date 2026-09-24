use crate::pack::words_per_row;

#[cfg(test)]
use crate::quant::naive_score_1bit;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub id: [u8; 16],
    pub score: f32,
}

pub fn hamming_similarity(dim: usize, query_row: &[u8], row: &[u8], stride: usize) -> f32 {
    let words = words_per_row(stride);
    let mut hamming = 0u32;
    for w in 0..words {
        let start_bit = w * 64;
        if start_bit >= dim {
            break;
        }
        let q = read_u64(query_row, w);
        let r = read_u64(row, w);
        let mut xor = q ^ r;
        let bits_in_word = (dim - start_bit).min(64);
        if bits_in_word < 64 {
            xor &= (1u64 << bits_in_word) - 1;
        }
        hamming += xor.count_ones();
    }
    dim as f32 - 2.0 * hamming as f32
}

pub fn top_k(
    dim: usize,
    stride: usize,
    count: usize,
    ids: &[[u8; 16]],
    vectors: &[u8],
    query_packed: &[u8],
    k: usize,
) -> Vec<Hit> {
    if count == 0 || k == 0 {
        return Vec::new();
    }
    let k = k.min(count);
    let mut hits: Vec<Hit> = Vec::with_capacity(k);
    for i in 0..count {
        let row_start = i * stride;
        let row = &vectors[row_start..row_start + stride];
        let score = hamming_similarity(dim, query_packed, row, stride);
        let id = ids[i];
        if hits.len() < k {
            hits.push(Hit { id, score });
            if hits.len() == k {
                hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            }
        } else if score > hits[k - 1].score {
            hits[k - 1] = Hit { id, score };
            hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        }
    }
    if hits.len() < k {
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    }
    hits
}

#[cfg(test)]
fn verify_against_naive(dim: usize, stride: usize, query_float: &[f32], query_packed: &[u8], row: &[u8]) -> bool {
    let fast = hamming_similarity(dim, query_packed, row, stride);
    let naive = naive_score_1bit(dim, query_float, row);
    (fast - naive).abs() < 1e-3
}

fn read_u64(row: &[u8], word_index: usize) -> u64 {
    let start = word_index * 8;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&row[start..start + 8]);
    u64::from_le_bytes(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::{pack_1bit_row, stride_bytes};
    use crate::quant::naive_score_1bit;
    use rand::Rng;

    #[test]
    fn one_bit_hamming_matches_naive() {
        let dim = 32;
        let stride = stride_bytes(dim);
        let mut a = vec![0u8; stride];
        let mut b = vec![0u8; stride];
        let va: Vec<f32> = (0..dim).map(|i| if i % 3 == 0 { 1.0 } else { -1.0 }).collect();
        let vb: Vec<f32> = (0..dim).map(|i| if i % 5 == 0 { 1.0 } else { -1.0 }).collect();
        pack_1bit_row(dim, &va, &mut a);
        pack_1bit_row(dim, &vb, &mut b);
        let mut query = vec![0u8; stride];
        pack_1bit_row(dim, &va, &mut query);
        assert!(verify_against_naive(dim, stride, &va, &query, &b));
        assert!((hamming_similarity(dim, &a, &a, stride) - dim as f32).abs() < 1e-5);
        assert!((hamming_similarity(dim, &a, &query, stride) - dim as f32).abs() < 1e-5);
    }

    #[test]
    fn top_k_matches_naive_bruteforce() {
        let mut rng = rand::rng();
        let dim = 16;
        let stride = stride_bytes(dim);
        let n = 40;
        let mut ids = Vec::new();
        let mut vectors = Vec::new();
        let mut floats = Vec::new();
        for i in 0..n {
            ids.push([i as u8; 16]);
            let row_vec: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let mut row = vec![0u8; stride];
            pack_1bit_row(dim, &row_vec, &mut row);
            vectors.extend_from_slice(&row);
            floats.push(row_vec);
        }
        let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let mut query_packed = vec![0u8; stride];
        pack_1bit_row(dim, &query, &mut query_packed);

        let fast = top_k(dim, stride, n, &ids, &vectors, &query_packed, 5);

        let mut naive: Vec<(u8, f32)> = ids
            .iter()
            .enumerate()
            .map(|(i, row_id)| {
                let start = i * stride;
                let row = &vectors[start..start + stride];
                let score = naive_score_1bit(dim, &query, row);
                (row_id[0], score)
            })
            .collect();
        naive.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        naive.truncate(5);

        assert_eq!(fast.len(), naive.len());
        for (hit, (nid, score)) in fast.iter().zip(naive.iter()) {
            assert_eq!(hit.id[0], *nid);
            assert!((hit.score - score).abs() < 1e-4);
        }
    }
}
