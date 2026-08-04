//! WASM (wasm-bindgen) bindings, split by concern.
//!
//! Only compiled with the `wasm` feature (wasm-pack builds with
//! `--features wasm --no-default-features`). Each submodule mirrors one
//! domain module under `swing/`.

pub mod wasm_swing;
