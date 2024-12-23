# SemScore Characterization Report
Generated on: 2024-12-22 16:52:08

## Descriptive Statistics

### Identical
- Samples: 84
- Mean: 1.000 (95% CI: [1.000, 1.000])
- Median: 1.000 (95% CI: [1.000, 1.000])
- Std Dev: 0.000 (95% CI: [0.000, 0.000])
- IQR: 0.000 (95% CI: [0.000, 0.000])

### Paraphrase
- Samples: 81
- Mean: 0.822 (95% CI: [0.801, 0.842])
- Median: 0.839 (95% CI: [0.813, 0.867])
- Std Dev: 0.098 (95% CI: [0.078, 0.116])
- IQR: 0.119 (95% CI: [0.088, 0.144])

### Similar Content
- Samples: 70
- Mean: 0.556 (95% CI: [0.514, 0.597])
- Median: 0.553 (95% CI: [0.507, 0.587])
- Std Dev: 0.178 (95% CI: [0.150, 0.202])
- IQR: 0.268 (95% CI: [0.181, 0.321])

### Related Topic
- Samples: 70
- Mean: 0.338 (95% CI: [0.287, 0.389])
- Median: 0.323 (95% CI: [0.281, 0.386])
- Std Dev: 0.217 (95% CI: [0.181, 0.249])
- IQR: 0.325 (95% CI: [0.220, 0.382])

### Different Domain
- Samples: 70
- Mean: 0.134 (95% CI: [0.106, 0.162])
- Median: 0.119 (95% CI: [0.075, 0.178])
- Std Dev: 0.119 (95% CI: [0.097, 0.139])
- IQR: 0.177 (95% CI: [0.139, 0.195])

### Unrelated
- Samples: 70
- Mean: -0.011 (95% CI: [-0.023, 0.001])
- Median: -0.012 (95% CI: [-0.030, 0.004])
- Std Dev: 0.050 (95% CI: [0.042, 0.057])
- IQR: 0.069 (95% CI: [0.050, 0.094])

### Contradiction
- Samples: 78
- Mean: 0.752 (95% CI: [0.708, 0.793])
- Median: 0.822 (95% CI: [0.739, 0.867])
- Std Dev: 0.190 (95% CI: [0.160, 0.216])
- IQR: 0.299 (95% CI: [0.196, 0.364])

## Pairwise Comparisons

| Comparison | CLES | Cohen's d | Mean Difference (95% CI) | KW p-value |
|------------|------|-----------|-------------------------|------------|
| Identical_vs_Paraphrase | 1.000 | 2.597 | 0.178 ([-0.034, 0.035]) | 1.038e-29 |
| Identical_vs_Similar Content | 1.000 | 3.696 | 0.444 ([-0.079, 0.079]) | 7.093e-28 |
| Identical_vs_Related Topic | 1.000 | 4.519 | 0.662 ([-0.113, 0.116]) | 7.092e-28 |
| Identical_vs_Different Domain | 1.000 | 10.825 | 0.866 ([-0.138, 0.136]) | 7.092e-28 |
| Identical_vs_Unrelated | 1.000 | 29.915 | 1.011 ([-0.160, 0.160]) | 7.093e-28 |
| Identical_vs_Contradiction | 1.000 | 1.880 | 0.248 ([-0.055, 0.056]) | 3.105e-29 |
| Paraphrase_vs_Similar Content | 0.900 | 1.886 | 0.266 ([-0.061, 0.062]) | 2.692e-17 |
| Paraphrase_vs_Related Topic | 0.969 | 2.944 | 0.484 ([-0.095, 0.092]) | 3.342e-23 |
| Paraphrase_vs_Different Domain | 1.000 | 6.371 | 0.688 ([-0.114, 0.117]) | 3.743e-26 |
| Paraphrase_vs_Unrelated | 1.000 | 10.491 | 0.833 ([-0.139, 0.137]) | 3.743e-26 |
| Paraphrase_vs_Contradiction | 0.562 | 0.466 | 0.070 ([-0.047, 0.049]) | 1.768e-01 |
| Similar Content_vs_Related Topic | 0.783 | 1.098 | 0.218 ([-0.077, 0.076]) | 7.638e-09 |
| Similar Content_vs_Different Domain | 0.974 | 2.792 | 0.423 ([-0.085, 0.085]) | 3.622e-22 |
| Similar Content_vs_Unrelated | 1.000 | 4.334 | 0.567 ([-0.101, 0.105]) | 1.779e-24 |
| Similar Content_vs_Contradiction | 0.222 | -1.059 | -0.196 ([-0.067, 0.067]) | 5.291e-09 |
| Related Topic_vs_Different Domain | 0.786 | 1.167 | 0.204 ([-0.068, 0.068]) | 5.002e-09 |
| Related Topic_vs_Unrelated | 0.953 | 2.212 | 0.349 ([-0.078, 0.080]) | 2.119e-20 |
| Related Topic_vs_Contradiction | 0.087 | -2.034 | -0.414 ([-0.094, 0.093]) | 4.853e-18 |
| Different Domain_vs_Unrelated | 0.886 | 1.586 | 0.145 ([-0.039, 0.040]) | 3.041e-15 |
| Different Domain_vs_Contradiction | 0.007 | -3.853 | -0.618 ([-0.113, 0.113]) | 4.696e-25 |
| Unrelated_vs_Contradiction | 0.000 | -5.358 | -0.763 ([-0.131, 0.132]) | 1.014e-25 |