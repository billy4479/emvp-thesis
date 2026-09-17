#![cfg(feature = "gpu")]

//! Device-independent validation of the `gpu_online` bench's float32 GEMV
//! shader.
//!
//! The `gpu_online` bench compiles this kernel only on a machine with a
//! compute adapter, so a WGSL syntax or validation error would otherwise
//! surface there alone. This test pins the shader's well-formedness on
//! every machine, mirroring the answer kernel's parse-and-validate unit
//! test in `emvp/src/gpu`: only the execution needs hardware.

const GEMV_F32_WGSL: &str = include_str!("../benches/gemv_f32.wgsl");

/// The bench's float32 GEMV shader must parse as WGSL and pass naga's full
/// validation without any adapter.
#[test]
fn gemv_f32_wgsl_parses_and_validates_device_independently() {
    let module = naga::front::wgsl::parse_str(GEMV_F32_WGSL).expect("the GEMV shader must parse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("the GEMV shader must validate");
}
