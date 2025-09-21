fn main() {
    // Re-run if any headers or wrapper source change
    println!("cargo:rerun-if-changed=wrapper/wrapper.c");
    println!("cargo:rerun-if-changed=wrapper/sphur.h");

    let mut build = cc::Build::new();

    build
        .file("wrapper/wrapper.c")
        .include("wrapper")
        .flag_if_supported("-O3")
        .warnings(false);

    if build.get_compiler().is_like_msvc() {
        build.define("SPHUR_USE_INTRIN", None);
        build.flag_if_supported("/Oi");
    }

    build.compile("sphur_wrapper");
}
