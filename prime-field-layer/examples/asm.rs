//! Assembly-extraction driver. Never run; built with `--emit asm` only.
#![expect(clippy::unwrap_used, reason = "driver uses fixed valid inputs")]

use std::hint::black_box;

use prime_field_layer::{FieldElement, NttPlan, PrimeField};

const P1: u32 = 1_073_479_681;
const P2: u32 = 2_013_265_921;
const P3: u32 = 2_281_701_377;
const P4: u32 = 65_537;
const PSEUDO_MERSENNE: u32 = 4_294_967_291;

#[inline(never)]
fn forward_auto(plan: &NttPlan<P1>, values: &mut [FieldElement<P1>]) {
    black_box(plan.forward(black_box(values))).unwrap();
}

#[inline(never)]
fn inverse_auto(plan: &NttPlan<P1>, values: &mut [FieldElement<P1>]) {
    black_box(plan.inverse(black_box(values))).unwrap();
}

#[inline(never)]
fn forward_scalar(plan: &NttPlan<P1>, values: &mut [FieldElement<P1>]) {
    black_box(plan.forward(black_box(values))).unwrap();
}

#[inline(never)]
fn inverse_scalar(plan: &NttPlan<P1>, values: &mut [FieldElement<P1>]) {
    black_box(plan.inverse(black_box(values))).unwrap();
}

#[inline(never)]
fn pointwise(plan: &NttPlan<P1>, lhs: &mut [FieldElement<P1>], rhs: &[FieldElement<P1>]) {
    black_box(plan.pointwise_mul_assign(black_box(lhs), black_box(rhs))).unwrap();
}

#[inline(never)]
fn forward_reduced(plan: &NttPlan<P2>, values: &mut [FieldElement<P2>]) {
    black_box(plan.forward(black_box(values))).unwrap();
}

#[inline(never)]
fn mul_elements(lhs: &mut [FieldElement<P1>], rhs: &[FieldElement<P1>]) {
    black_box(PrimeField::<P1>::new().mul_elements_assign(black_box(lhs), black_box(rhs))).unwrap();
}

#[inline(never)]
fn scalar_mul_elements(values: &mut [FieldElement<P1>], scalar: FieldElement<P1>) {
    PrimeField::<P1>::new().scalar_mul_elements_assign(black_box(values), black_box(scalar));
}

#[inline(never)]
fn add_elements(lhs: &mut [FieldElement<P1>], rhs: &[FieldElement<P1>]) {
    black_box(PrimeField::<P1>::new().add_elements_assign(black_box(lhs), black_box(rhs))).unwrap();
}

#[inline(never)]
fn forward_wide(plan: &NttPlan<P3>, values: &mut [FieldElement<P3>]) {
    black_box(plan.forward(black_box(values))).unwrap();
}

#[inline(never)]
fn dot_u64_fitting(lhs: &[u32], rhs: &[u32]) -> u32 {
    black_box(PrimeField::<P4>::new().dot_canonical(black_box(lhs), black_box(rhs))).unwrap()
}

#[inline(never)]
fn dot_pseudo_mersenne(lhs: &[u32], rhs: &[u32]) -> u32 {
    black_box(PrimeField::<PSEUDO_MERSENNE>::new().dot_canonical(black_box(lhs), black_box(rhs)))
        .unwrap()
}

fn main() {
    let auto = black_box(NttPlan::<P1>::new(1024).unwrap());
    let scalar = black_box(NttPlan::<P1>::new_scalar(1024).unwrap());
    let reduced = black_box(NttPlan::<P2>::new(1024).unwrap());

    let mut values = auto.elements(&black_box((0..1024).collect::<Vec<u32>>()));
    let rhs = values.clone();

    forward_auto(&auto, &mut values);
    inverse_auto(&auto, &mut values);
    forward_scalar(&scalar, &mut values);
    inverse_scalar(&scalar, &mut values);
    pointwise(&auto, &mut values, &rhs);
    let reduced_input: Vec<u32> = values.iter().map(|value| value.value()).collect();
    forward_reduced(&reduced, &mut black_box(reduced.elements(&reduced_input)));
    mul_elements(&mut values, &rhs);
    scalar_mul_elements(&mut values, black_box(rhs[0]));
    add_elements(&mut values, &rhs);
    let wide = black_box(NttPlan::<P3>::new(1024).unwrap());
    let mut wide_values = wide.elements(&black_box((0..1024).collect::<Vec<u32>>()));
    forward_wide(&wide, &mut wide_values);
    let dot_lhs: Vec<u32> = (0..4096).map(|index| index % 1_000).collect();
    let dot_rhs: Vec<u32> = (0..4096).map(|index| (index * 7 + 3) % 1_000).collect();
    black_box(dot_u64_fitting(&dot_lhs, &dot_rhs));
    black_box(dot_pseudo_mersenne(&dot_lhs, &dot_rhs));
    black_box(wide_values);
    black_box(values);
}
