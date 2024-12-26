# SemScore Characterization Report
Generated on: 2024-12-24 18:26:26

## Descriptive Statistics

### Identical
- Samples: 107
- Mean: 1.000 (95% CI: [1.000, 1.000])
- Median: 1.000 (95% CI: [1.000, 1.000])
- Std Dev: 0.000 (95% CI: [0.000, 0.000])
- IQR: 0.000 (95% CI: [0.000, 0.000])

### Paraphrase
- Samples: 104
- Mean: 0.837 (95% CI: [0.818, 0.854])
- Median: 0.860 (95% CI: [0.839, 0.871])
- Std Dev: 0.094 (95% CI: [0.077, 0.110])
- IQR: 0.115 (95% CI: [0.080, 0.136])

### Similar Content
- Samples: 90
- Mean: 0.551 (95% CI: [0.516, 0.587])
- Median: 0.545 (95% CI: [0.524, 0.570])
- Std Dev: 0.171 (95% CI: [0.148, 0.193])
- IQR: 0.250 (95% CI: [0.175, 0.293])

### Related Topic
- Samples: 90
- Mean: 0.340 (95% CI: [0.298, 0.383])
- Median: 0.334 (95% CI: [0.295, 0.382])
- Std Dev: 0.206 (95% CI: [0.175, 0.233])
- IQR: 0.308 (95% CI: [0.195, 0.351])

### Different Domain
- Samples: 97
- Mean: 0.149 (95% CI: [0.126, 0.172])
- Median: 0.127 (95% CI: [0.098, 0.180])
- Std Dev: 0.119 (95% CI: [0.101, 0.134])
- IQR: 0.173 (95% CI: [0.142, 0.210])

### Unrelated
- Samples: 90
- Mean: 0.015 (95% CI: [-0.001, 0.032])
- Median: 0.004 (95% CI: [-0.012, 0.013])
- Std Dev: 0.078 (95% CI: [0.063, 0.092])
- IQR: 0.090 (95% CI: [0.068, 0.110])

### Contradiction
- Samples: 93
- Mean: 0.781 (95% CI: [0.742, 0.819])
- Median: 0.859 (95% CI: [0.799, 0.890])
- Std Dev: 0.187 (95% CI: [0.157, 0.212])
- IQR: 0.248 (95% CI: [0.166, 0.344])

## Pairwise Comparisons

| Comparison | CLES | Cohen's d | Mean Difference (95% CI) | KW p-value |
|------------|------|-----------|-------------------------|------------|
| Identical_vs_Paraphrase | 1.000 | 2.464 | 0.163 ([-0.028, 0.028]) | 1.489e-37 |
| Identical_vs_Similar Content | 1.000 | 3.875 | 0.449 ([-0.073, 0.070]) | 3.055e-35 |
| Identical_vs_Related Topic | 1.000 | 4.749 | 0.660 ([-0.101, 0.101]) | 3.054e-35 |
| Identical_vs_Different Domain | 1.000 | 10.409 | 0.851 ([-0.120, 0.121]) | 1.947e-36 |
| Identical_vs_Unrelated | 1.000 | 18.578 | 0.985 ([-0.141, 0.138]) | 3.055e-35 |
| Identical_vs_Contradiction | 1.000 | 1.720 | 0.219 ([-0.046, 0.047]) | 9.169e-36 |
| Paraphrase_vs_Similar Content | 0.924 | 2.103 | 0.285 ([-0.055, 0.056]) | 2.868e-24 |
| Paraphrase_vs_Related Topic | 0.977 | 3.179 | 0.496 ([-0.082, 0.083]) | 2.184e-30 |
| Paraphrase_vs_Different Domain | 1.000 | 6.441 | 0.688 ([-0.099, 0.098]) | 1.896e-34 |
| Paraphrase_vs_Unrelated | 1.000 | 9.402 | 0.822 ([-0.120, 0.121]) | 3.553e-33 |
| Paraphrase_vs_Contradiction | 0.521 | 0.383 | 0.056 ([-0.041, 0.042]) | 6.166e-01 |
| Similar Content_vs_Related Topic | 0.789 | 1.115 | 0.211 ([-0.063, 0.064]) | 2.080e-11 |
| Similar Content_vs_Different Domain | 0.973 | 2.750 | 0.403 ([-0.072, 0.069]) | 6.850e-29 |
| Similar Content_vs_Unrelated | 0.998 | 4.025 | 0.536 ([-0.086, 0.087]) | 8.471e-31 |
| Similar Content_vs_Contradiction | 0.181 | -1.280 | -0.230 ([-0.061, 0.062]) | 9.125e-14 |
| Related Topic_vs_Different Domain | 0.786 | 1.152 | 0.191 ([-0.055, 0.055]) | 1.544e-11 |
| Related Topic_vs_Unrelated | 0.935 | 2.091 | 0.325 ([-0.067, 0.066]) | 6.639e-24 |
| Related Topic_vs_Contradiction | 0.069 | -2.245 | -0.441 ([-0.086, 0.083]) | 6.794e-24 |
| Different Domain_vs_Unrelated | 0.843 | 1.322 | 0.134 ([-0.035, 0.034]) | 5.820e-16 |
| Different Domain_vs_Contradiction | 0.006 | -4.057 | -0.632 ([-0.100, 0.099]) | 6.816e-32 |
| Unrelated_vs_Contradiction | 0.000 | -5.314 | -0.766 ([-0.119, 0.117]) | 1.798e-31 |