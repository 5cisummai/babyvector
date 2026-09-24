use std::cell::RefCell;
use std::ffi::{c_char, c_float, c_int, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::ptr;
use std::sync::Mutex;

use crate::index::Index;
use crate::meta::IndexConfig;
use crate::store::META_FILE;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

pub struct BvHandle {
    index: Mutex<Index>,
}

fn set_error(msg: impl Into<String>) {
    let raw = msg.into().replace('\0', "");
    let c = CString::new(raw).unwrap_or_else(|_| CString::new("invalid error message").unwrap());
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(c));
}

fn clear_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

fn cstr_path(ptr: *const c_char) -> Result<PathBuf, ()> {
    if ptr.is_null() {
        set_error("null path");
        return Err(());
    }
    let s = unsafe { CStr::from_ptr(ptr) };
    match s.to_str() {
        Ok(v) => Ok(PathBuf::from(v)),
        Err(_) => {
            set_error("path is not valid UTF-8");
            Err(())
        }
    }
}

fn with_index<T>(handle: *mut BvHandle, f: impl FnOnce(&mut Index) -> Result<T, String>) -> Result<T, ()> {
    if handle.is_null() {
        set_error("null handle");
        return Err(());
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let h = unsafe { &*handle };
        let mut index = h.index.lock().unwrap_or_else(|poison| poison.into_inner());
        f(&mut index)
    }));
    match result {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(msg)) => {
            set_error(msg);
            Err(())
        }
        Err(_) => {
            set_error("panic in babyvector");
            Err(())
        }
    }
}

/// Returns a pointer valid until the next error on this thread.
#[no_mangle]
pub extern "C" fn bv_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(s) => s.as_ptr(),
        None => ptr::null(),
    })
}

#[no_mangle]
pub extern "C" fn bv_create(dir: *const c_char, dim: u32) -> *mut BvHandle {
    clear_error();
    if dim == 0 {
        set_error("dim must be positive");
        return ptr::null_mut();
    }
    let path = match cstr_path(dir) {
        Ok(p) => p,
        Err(()) => return ptr::null_mut(),
    };
    if path.join(META_FILE).exists() {
        set_error(format!("index already exists: {}", path.display()));
        return ptr::null_mut();
    }
    if let Err(e) = std::fs::create_dir_all(&path) {
        set_error(e.to_string());
        return ptr::null_mut();
    }
    let cfg = IndexConfig { dim: dim as usize };
    match Index::create(path, cfg) {
        Ok(index) => Box::into_raw(Box::new(BvHandle {
            index: Mutex::new(index),
        })),
        Err(e) => {
            set_error(e.to_string());
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn bv_open(dir: *const c_char) -> *mut BvHandle {
    clear_error();
    let path = match cstr_path(dir) {
        Ok(p) => p,
        Err(()) => return ptr::null_mut(),
    };
    match Index::open(path) {
        Ok(index) => Box::into_raw(Box::new(BvHandle {
            index: Mutex::new(index),
        })),
        Err(e) => {
            set_error(e.to_string());
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn bv_close(handle: *mut BvHandle) -> c_int {
    if handle.is_null() {
        set_error("null handle");
        return -1;
    }
    clear_error();
    unsafe {
        drop(Box::from_raw(handle));
    }
    0
}

#[no_mangle]
pub extern "C" fn bv_drop(dir: *const c_char) -> c_int {
    clear_error();
    let path = match cstr_path(dir) {
        Ok(p) => p,
        Err(()) => return -1,
    };
    if !path.exists() {
        return 0;
    }
    if let Err(e) = std::fs::remove_dir_all(&path) {
        set_error(e.to_string());
        return -1;
    }
    0
}

#[no_mangle]
pub extern "C" fn bv_add_vectors(
    handle: *mut BvHandle,
    ids: *const u8,
    vectors: *const c_float,
    n: u32,
) -> c_int {
    clear_error();
    if n == 0 {
        return 0;
    }
    if ids.is_null() || vectors.is_null() {
        set_error("null argument");
        return -1;
    }
    match with_index(handle, |index| {
        let dim = index.meta().dim;
        let n = n as usize;
        let id_slice = unsafe { std::slice::from_raw_parts(ids, n * 16) };
        let mut id_rows = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * 16;
            let mut row = [0u8; 16];
            row.copy_from_slice(&id_slice[start..start + 16]);
            id_rows.push(row);
        }
        let floats = unsafe { std::slice::from_raw_parts(vectors, n * dim) };
        index.add_vectors(&id_rows, floats).map_err(|e| e.to_string())
    }) {
        Ok(()) => 0,
        Err(()) => -1,
    }
}

#[no_mangle]
pub extern "C" fn bv_publish(handle: *mut BvHandle) -> c_int {
    clear_error();
    match with_index(handle, |index| index.publish().map_err(|e| e.to_string())) {
        Ok(()) => 0,
        Err(()) => -1,
    }
}

#[no_mangle]
pub extern "C" fn bv_search(
    handle: *mut BvHandle,
    query: *const c_float,
    k: u32,
    out_ids: *mut u8,
    out_scores: *mut c_float,
) -> u32 {
    clear_error();
    if query.is_null() || out_ids.is_null() || out_scores.is_null() {
        set_error("null argument");
        return 0;
    }
    match with_index(handle, |index| {
        let dim = index.meta().dim;
        let q = unsafe { std::slice::from_raw_parts(query, dim) };
        let hits = index.search_vector(q, k as usize).map_err(|e| e.to_string())?;
        let n = hits.len();
        let out_id_slice = unsafe { std::slice::from_raw_parts_mut(out_ids, n * 16) };
        let out_score_slice = unsafe { std::slice::from_raw_parts_mut(out_scores, n) };
        for (i, hit) in hits.iter().enumerate() {
            out_id_slice[i * 16..(i + 1) * 16].copy_from_slice(&hit.id);
            out_score_slice[i] = hit.score;
        }
        Ok(n as u32)
    }) {
        Ok(n) => n,
        Err(()) => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::path::Path;

    fn err_str() -> String {
        let p = bv_last_error();
        if p.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }

    #[test]
    fn ffi_round_trip_self_is_top1() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("idx");
        std::fs::create_dir_all(&path).unwrap();
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let dim = 8u32;
        let h = bv_create(cpath.as_ptr(), dim);
        assert!(!h.is_null(), "{}", err_str());

        let mut ids = [0u8; 48];
        ids[0] = 1;
        ids[16] = 2;
        ids[32] = 3;
        let mut vectors = vec![0f32; 24];
        for i in 0..3 {
            for j in 0..8 {
                vectors[i * 8 + j] = if j == i { 1.0 } else { -1.0 };
            }
        }
        assert_eq!(
            bv_add_vectors(h, ids.as_ptr(), vectors.as_ptr(), 3),
            0,
            "{}",
            err_str()
        );
        assert_eq!(bv_publish(h), 0, "{}", err_str());

        let query = &vectors[8..16];
        let mut out_ids = [0u8; 16];
        let mut out_scores = [0f32; 1];
        let n = bv_search(h, query.as_ptr(), 1, out_ids.as_mut_ptr(), out_scores.as_mut_ptr());
        assert_eq!(n, 1, "{}", err_str());
        assert_eq!(out_ids[0], 2);

        assert_eq!(bv_close(h), 0);
        assert_eq!(bv_drop(cpath.as_ptr()), 0);
        assert!(!Path::new(&path).exists());
    }
}
