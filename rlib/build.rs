use std::process::Command;

fn main() {
    // re-run if asm code has changed
    println!("cargo:rerun-if-changed=../asm/linux_x86.asm");

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let obj_path = format!("{}/linux_x86.o", out_dir);

    // build asm
    let status = Command::new("nasm")
        .args(&["-f", "elf64", "../asm/linux_x86.asm", "-o", &obj_path])
        .status()
        .expect("failed to run nasm");

    assert!(status.success(), "nasm failed");

    // link object file
    println!("cargo:rustc-link-arg={}", obj_path);
}
