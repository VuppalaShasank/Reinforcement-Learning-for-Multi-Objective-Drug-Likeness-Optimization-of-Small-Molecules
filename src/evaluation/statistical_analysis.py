"""
Statistical Analysis Module
Computes parametric (paired Student's t-test) and non-parametric (Wilcoxon signed-rank test)
hypothesis tests for comparing molecular optimization methods.
"""

import numpy as np
import pandas as pd
from scipy import stats

def compute_paired_tests(series_a, series_b, method_a_name="Method A", method_b_name="Method B", metric_name="Delta-QED"):
    """
    Computes two-sided paired t-test and Wilcoxon signed-rank test.
    """
    a = np.array(series_a, dtype=np.float64)
    b = np.array(series_b, dtype=np.float64)
    
    diff = a - b
    
    # Paired Student's t-test
    t_stat, p_t = stats.ttest_rel(a, b)
    
    # Wilcoxon signed-rank test
    w_stat, p_w = stats.wilcoxon(diff)
    
    return {
        "metric": metric_name,
        "method_a": method_a_name,
        "method_b": method_b_name,
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "t_statistic": float(t_stat),
        "t_pvalue": float(p_t),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(p_w),
        "significant_p05": bool(p_w < 0.05),
        "significant_p001": bool(p_w < 0.001)
    }

def print_test_summary(test_result):
    print(f"=== Statistical Test: {test_result['method_a']} vs {test_result['method_b']} ===")
    print(f"Metric: {test_result['metric']}")
    print(f"Mean Difference: {test_result['mean_diff']:+.4f} (Median: {test_result['median_diff']:+.4f})")
    print(f"Paired t-test: t = {test_result['t_statistic']:.4f}, p = {test_result['t_pvalue']:.3e}")
    print(f"Wilcoxon signed-rank: W = {test_result['wilcoxon_statistic']:.1f}, p = {test_result['wilcoxon_pvalue']:.3e}")
    sig_str = "Statistically significant at p < 0.001" if test_result['significant_p001'] else "Not significant"
    print(f"Significance: {sig_str}\n")

