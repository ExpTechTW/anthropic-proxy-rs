fn main() {
    // `src/main.rs` bakes in `BUILD_VERSION` via `option_env!`. Without this,
    // a cached `target/` (e.g. Swatinem/rust-cache in CI) would keep the binary
    // pinned to the first version it was built with. Declaring the env var as a
    // build input forces a rebuild whenever it changes.
    println!("cargo:rerun-if-env-changed=BUILD_VERSION");
}
