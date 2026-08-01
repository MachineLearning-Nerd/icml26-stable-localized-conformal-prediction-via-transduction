"""Published values transcribed from the paper, for cell-by-cell comparison.

Source: ar5iv HTML of arXiv 2605.01452, SHA-256
d07bc37a6a81e0c74aef488fd566dfe5ebf4e0b94ad526c3449334000c2741a2.
Method order in every row: base, SDCP, PPI, ours, ours-sel, oracle, DP.
`marks` records the paper's own -/+ marginal-coverage superscripts.
"""

METHODS = ["base", "SDCP", "PPI", "ours", "ours-sel", "oracle", "DP"]

# Table 1 -- Std block, then Marginal block.
TABLE1 = {
    "CRIME": {
        "n_over_m": "30/500",
        "GLCP": {
            "std": [0.75, 1.42, 0.78, 0.50, 0.58, 0.16, 0.86],
            "pct": {"ours": 42.9, "ours-sel": 27.7},
            "marginal": [0.896, 0.904, 0.901, 0.895, 0.898, 0.895, 0.911],
            "marks": ["", "", "", "", "", "", ""],
        },
        "CQR": {
            "std": [0.55, 0.53, 0.53, 0.42, 0.44, 0.10, 0.86],
            "pct": {"ours": 25.0, "ours-sel": 19.4},
            "marginal": [0.896, 0.895, 0.898, 0.903, 0.900, 0.898, 0.909],
            "marks": ["", "", "", "", "", "", ""],
        },
    },
    "BIO": {
        "n_over_m": "30/1000",
        "GLCP": {
            "std": [0.53, 0.66, 0.58, 0.37, 0.49, 0.07, 0.31],
            "pct": {"ours": 35.7, "ours-sel": 8.3},
            "marginal": [0.902, 0.915, 0.906, 0.930, 0.921, 0.900, 0.956],
            "marks": ["", "", "", "", "", "", "+"],
        },
        "CQR": {
            "std": [0.42, 0.39, 0.41, 0.32, 0.36, 0.14, 0.31],
            "pct": {"ours": 29.3, "ours-sel": 12.3},
            "marginal": [0.896, 0.913, 0.907, 0.916, 0.914, 0.901, 0.959],
            "marks": ["", "", "", "", "", "", "+"],
        },
    },
    "STAR": {
        "n_over_m": "30/1000",
        "GLCP": {
            "std": [8.78, 13.18, 10.42, 5.25, 5.65, 1.48, 9.13],
            "pct": {"ours": 48.4, "ours-sel": 42.9},
            "marginal": [0.899, 0.911, 0.902, 0.919, 0.918, 0.902, 0.939],
            "marks": ["", "", "", "", "", "", "+"],
        },
        "CQR": {
            "std": [6.45, 7.87, 6.25, 5.90, 5.93, 1.42, 8.80],
            "pct": {"ours": 7.3, "ours-sel": 6.7},
            "marginal": [0.892, 0.905, 0.899, 0.895, 0.895, 0.896, 0.935],
            "marks": ["", "", "", "", "", "", "+"],
        },
    },
    "DERMA": {
        "n_over_m": "30/1000",
        "GLCP": {
            "std": [0.68, 0.72, 0.78, 0.49, 0.46, 0.06, 0.12],
            "pct": {"ours": 30.4, "ours-sel": 35.8},
            "marginal": [0.930, 0.915, 0.937, 0.933, 0.932, 0.900, 0.976],
            "marks": ["", "", "+", "", "", "", "+"],
        },
        "CQR": {
            "std": [0.15, 0.22, 0.15, 0.14, 0.13, 0.09, 0.09],
            "pct": {"ours": 22.1, "ours-sel": 29.3},
            "marginal": [0.925, 0.925, 0.931, 0.924, 0.930, 0.901, 0.964],
            "marks": ["", "", "", "", "", "", "+"],
        },
    },
    "TISSUE": {
        "n_over_m": "30/1000",
        "GLCP": {
            "std": [0.83, 1.23, 1.02, 0.73, 0.79, 0.12, 0.07],
            "pct": {"ours": 13.5, "ours-sel": 4.8},
            "marginal": [0.909, 0.926, 0.923, 0.928, 0.923, 0.905, 0.998],
            "marks": ["", "", "", "", "", "", "+"],
        },
        "CQR": {
            "std": [0.72, 1.12, 0.97, 0.62, 0.64, 0.10, 0.09],
            "pct": {"ours": 15.4, "ours-sel": 11.5},
            "marginal": [0.905, 0.917, 0.920, 0.925, 0.920, 0.905, 0.994],
            "marks": ["", "", "", "", "", "", "+"],
        },
    },
}

# Table 2 -- LogAbs, m = 500.
TABLE2 = {
    30: {
        "GLCP": {
            "std": [1.12, 1.65, 1.09, 0.77, 0.88, 0.36, 0.85],
            "pct": {"ours": 31.2, "ours-sel": 20.7},
            "marginal": [0.899, 0.935, 0.909, 0.911, 0.910, 0.904, 0.921],
            "marks": ["", "+", "", "", "", "", ""],
        },
        "CQR": {
            "std": [0.98, 0.88, 0.94, 0.82, 0.83, 0.31, 0.81],
            "pct": {"ours": 16.3, "ours-sel": 15.4},
            "marks": ["", "", "", "", "", "", ""],
        },
    },
    100: {
        "GLCP": {
            "std": [0.50, 0.67, 0.46, 0.38, 0.38, 0.15, 0.41],
            "pct": {"ours": 23.2, "ours-sel": 24.4},
            "marks": ["", "-", "", "", "+", "", "+"],
        },
        "CQR": {
            "std": [0.45, 0.46, 0.47, 0.38, 0.38, 0.14, 0.38],
            "pct": {"ours": 16.7, "ours-sel": 15.8},
            "marks": ["", "-", "", "", "", "", "+"],
        },
    },
    500: {
        "GLCP": {
            "std": [0.25, 0.21, 0.21, 0.18, 0.16, 0.14, 0.21],
            "pct": {"ours": 26.6, "ours-sel": 33.9},
            "marks": ["", "-", "", "", "+", "", "+"],
        },
        "CQR": {
            "std": [0.18, 0.17, 0.18, 0.17, 0.17, 0.13, 0.20],
            "pct": {"ours": 6.3, "ours-sel": 6.3},
            "marks": ["", "-", "", "", "", "", "+"],
        },
    },
}

# Repeats behind every Table 2 setting (Section 5.2).
SIM_REPEATS = 50

# Repeats behind every Table 1 column (Section 5.1). A reproduction assembled
# from shards is only comparable to the printed cells at this exact count.
REAL_REPEATS = 50

# The claim set under test, verbatim from the challenge record.
CLAIM4_GLCP_BAND = (20.0, 48.0)
CLAIM4_CQR_BAND = (6.0, 29.0)
CLAIM5_TARGETS = {"GLCP": 31.2, "CQR": 16.3}

# The real-data experiments all run at the calibration size Table 1 reports.
REAL_N = 30
# Theorem 4.7 guarantee band at alpha=0.1, alpha_tol=0.02, n=30.
THM47_BAND = (0.88, 0.9 + 0.02 + 1.0 / (REAL_N + 1))
# Annotation bands used for the -/+ superscripts. The two tables do NOT agree:
#   Table 1 (RealAnalysis/sum_tab.py:34):  [0.9-0.01, 0.901 + 1/n]   -> [0.89, 0.934333]
#   Table 2 (SimuAnalysis/sum_tab.py:56):  [0.9-0.01, 0.9   + 1/(n+1)] -> [0.89, 0.932258]
# The captions state the second form for both. Matching each table's own code is
# what reproduces its published marks and hence its reference baselines.
REAL_ANNOTATION_BAND = (0.89, 0.901 + 1.0 / 30.0)
SIM_ANNOTATION_BAND = (0.89, 0.9 + 1.0 / 31.0)
TABLE_ANNOTATION_BAND = SIM_ANNOTATION_BAND
