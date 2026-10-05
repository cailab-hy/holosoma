# Fixed-probe Q levels (probe 4096 rows, seed 12345; mean ± std over seeds)

Q_level = mean min(Q1,Q2)(s_probe, a_probe); LSE = mean 0.5(LSE1+LSE2)(s_probe); bracket = LSE - Q_D twin mean.

| Method | n | peak step | Q_level @peak | Q_level @final | Q_level final-k | LSE @peak | LSE @final | bracket @peak | bracket @final | Q_pi @final | Q_rand @final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CQL (alpha=5) | 5 | 49400 ± 8764 | 19.3 ± 1.8 | 13.0 ± 0.2 | 13.4 ± 0.1 | 11.6 ± 1.9 | 5.1 ± 0.2 | -8.2 ± 0.1 | -8.4 ± 0.1 | 16.0 ± 0.2 | -133.4 ± 4.6 |
| B-arm (LSE - wQ_D) | 5 | 89200 ± 9524 | 12.7 ± 0.9 | 11.5 ± 0.2 | 12.0 ± 0.2 | 6.1 ± 0.9 | 4.9 ± 0.1 | -7.5 ± 0.1 | -7.4 ± 0.1 | 15.5 ± 0.2 | -189.3 ± 11.9 |
| AW-CQL (H50) | 5 | 73000 ± 16688 | 20.8 ± 1.9 | 18.2 ± 0.4 | 18.5 ± 0.2 | 13.7 ± 2.0 | 10.9 ± 0.4 | -7.7 ± 0.1 | -7.8 ± 0.1 | 21.9 ± 0.4 | -151.7 ± 11.1 |
