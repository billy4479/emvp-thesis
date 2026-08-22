#![expect(
    clippy::unwrap_used,
    reason = "test inputs establish that plans and transforms must succeed"
)]

#[path = "ntt/backends.rs"]
mod backends;
#[path = "ntt/convolution.rs"]
mod convolution;
#[path = "ntt/pretransformed.rs"]
mod pretransformed;
#[path = "ntt/properties.rs"]
mod properties;
#[path = "ntt/support.rs"]
mod support;
#[path = "ntt/transforms.rs"]
mod transforms;
