use std::process::Command;

#[test]
fn precompile_accepts_a_single_file() {
    // A DXBC container with no parts: exercise input selection without requiring DXC.
    let mut blob = [0u8; 32];
    blob[..4].copy_from_slice(b"DXBC");
    blob[20..24].copy_from_slice(&1u32.to_le_bytes());
    blob[24..28].copy_from_slice(&32u32.to_le_bytes());
    let file = std::env::temp_dir().join(format!("xcl-single-shader-{}.dxbc", std::process::id()));
    std::fs::write(&file, blob).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_bathyal"))
        .arg("shaders")
        .arg(&file)
        .args(["--precompile", "--split", "0"])
        .env("XCL_DXC_THREADS", "1")
        .output();
    std::fs::remove_file(file).unwrap();

    let output = output.unwrap();
    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("precompiling 1 "), "{stdout}");
    assert!(stdout.contains("compiled 0, already cached 0, errors 0"), "{stdout}");
}
