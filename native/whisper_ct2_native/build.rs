//! Lets the unit tests link on Linux.
//!
//! Since rustler 0.39, the NIF links the `enif_*` functions of the BEAM
//! directly on Linux. The BEAM supplies them when it loads the NIF. The test
//! binary runs without the BEAM and calls none of them, so the linker must
//! leave them unresolved there. Cargo has no link argument for unit tests
//! only, so the flag applies to every link of the crate. A shared library
//! leaves unresolved symbols open anyway, so the flag changes only the test
//! binary.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo::rustc-link-arg=-Wl,--unresolved-symbols=ignore-in-object-files");
    }
}
