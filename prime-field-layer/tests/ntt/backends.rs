use prime_field_layer::{FieldError, NttBackend, NttPerformanceWarning, NttPlan};

#[test]
fn diagnostics_report_the_actual_modulus_tier() {
    let lazy = NttPlan::<998_244_353>::new_scalar(256).unwrap();
    assert_eq!(lazy.backend(), NttBackend::ScalarShoupLazy);
    assert_eq!(
        lazy.performance_warning(),
        Some(NttPerformanceWarning::ScalarRequested)
    );

    let tight = NttPlan::<2_013_265_921>::new(256).unwrap();
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        assert_eq!(tight.backend(), NttBackend::Avx2Shoup);
        assert_eq!(tight.performance_warning(), None);
    } else {
        assert_eq!(tight.backend(), NttBackend::ScalarShoup);
        assert_eq!(
            tight.performance_warning(),
            Some(NttPerformanceWarning::Avx2Unavailable)
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        assert_eq!(tight.backend(), NttBackend::ScalarShoup);
        assert_eq!(
            tight.performance_warning(),
            Some(NttPerformanceWarning::Avx2Unavailable)
        );
    }

    let wide = NttPlan::<2_281_701_377>::new(256).unwrap();
    assert_eq!(wide.backend(), NttBackend::ScalarMontgomery);
    assert_eq!(
        wide.performance_warning(),
        Some(NttPerformanceWarning::MontgomeryFallback)
    );
    assert!(matches!(
        NttPlan::<2_281_701_377>::new_avx2(256),
        Err(FieldError::Avx2Unavailable)
    ));

    let below_boundary = NttPlan::<1_053_818_881>::new_scalar(16).unwrap();
    assert_eq!(below_boundary.backend(), NttBackend::ScalarShoupLazy);
    let above_boundary = NttPlan::<1_107_296_257>::new_scalar(16).unwrap();
    assert_eq!(above_boundary.backend(), NttBackend::ScalarShoup);

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        assert_eq!(
            NttPlan::<1_053_818_881>::new_avx2(16).unwrap().backend(),
            NttBackend::Avx2ShoupLazy
        );
        assert_eq!(
            NttPlan::<1_107_296_257>::new_avx2(16).unwrap().backend(),
            NttBackend::Avx2Shoup
        );
    }

    let short_auto = NttPlan::<998_244_353>::new(256).unwrap();
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            assert_eq!(short_auto.backend(), NttBackend::Avx2ShoupLazy);
            assert_eq!(short_auto.performance_warning(), None);
        } else {
            assert_eq!(short_auto.backend(), NttBackend::ScalarShoupLazy);
            assert_eq!(
                short_auto.performance_warning(),
                Some(NttPerformanceWarning::Avx2Unavailable)
            );
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    assert_eq!(
        short_auto.performance_warning(),
        Some(NttPerformanceWarning::Avx2Unavailable)
    );
}
