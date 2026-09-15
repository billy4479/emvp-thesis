use prime_field_layer::{NttBackend, NttPerformanceWarning, NttPlan};

#[test]
fn diagnostics_report_the_actual_modulus_tier() {
    let lazy = NttPlan::<1_073_479_681>::new(256).unwrap();
    assert_eq!(lazy.backend(), NttBackend::ScalarShoupLazy);
    assert_eq!(lazy.performance_warning(), None);

    let tight = NttPlan::<2_013_265_921>::new(256).unwrap();
    assert_eq!(tight.backend(), NttBackend::ScalarShoup);
    assert_eq!(tight.performance_warning(), None);

    let wide = NttPlan::<2_281_701_377>::new(256).unwrap();
    assert_eq!(wide.backend(), NttBackend::ScalarMontgomery);
    assert_eq!(
        wide.performance_warning(),
        Some(NttPerformanceWarning::MontgomeryFallback)
    );

    let below_boundary = NttPlan::<1_053_818_881>::new(16).unwrap();
    assert_eq!(below_boundary.backend(), NttBackend::ScalarShoupLazy);
    let above_boundary = NttPlan::<1_107_296_257>::new(16).unwrap();
    assert_eq!(above_boundary.backend(), NttBackend::ScalarShoup);

    let short = NttPlan::<1_073_479_681>::new(16).unwrap();
    assert_eq!(short.backend(), NttBackend::ScalarShoupLazy);
    assert_eq!(short.performance_warning(), None);
}
