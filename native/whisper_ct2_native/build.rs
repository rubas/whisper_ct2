//! Links the KleidiAI archive of oneDNN into the NIF.

use std::env;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    // TODO(revert: when onednn-src links libkleidiai_ops.a itself): on
    // aarch64, oneDNN builds its KleidiAI matmul kernels into a second static
    // archive next to libdnnl.a, and libdnnl.a calls into it. onednn-src and
    // ct2rs link only libdnnl.a, and a shared library may keep undefined
    // symbols, so the NIF links but fails to load. rustc puts a link
    // argument after every library, so the archive follows libdnnl.a, as a
    // static link needs.
    if env::var_os("CARGO_FEATURE_DNNL").is_some()
        && env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|arch| arch == "aarch64")
    {
        let lib_dir =
            env::var("DEP_DNNL_LIBRARY_PATH").expect("onednn-src sets DEP_DNNL_LIBRARY_PATH");
        println!("cargo::rustc-link-arg={lib_dir}/libkleidiai_ops.a");
    }
}
