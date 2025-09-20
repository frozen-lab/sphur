fn main() {
    // Re-run if any headers or wrapper source change
    println!("cargo:rerun-if-changed=wrapper/wrapper.c");
    println!("cargo:rerun-if-changed=wrapper/wrapper.h");
    println!("cargo:rerun-if-changed=../clib/include/sphur.h");

    cc::Build::new()
        .file("wrapper/wrapper.c")
        .include("wrapper")
        .include("../clib/include")
        .flag_if_supported("-O2")
        .compile("sphur_wrapper");
}
