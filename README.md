# hwo-characterize

This repository documents an exploratory analysis of 164 provisional target stars for NASA’s Habitable Worlds Observatory. We combined publicly available data from SIMBAD, Gaia, the TESS Input Catalog, and the HWO Activity and Rotation Catalog to examine stellar magnitudes, colors, temperatures, metallicities, rotation periods, and ages. We also tested whether the gyro-interp package could reproduce compiled ages for eligible FGK dwarfs. The initial results show expected broad stellar trends but poor agreement between gyrochronology-derived and catalog ages. Because the reference ages and rotation measurements come from heterogeneous sources, this result should be interpreted as a motivation for further quality-controlled analysis rather than as a definitive test of gyrochronology.

Contents:
1. Research Paper v2: The second draft, including gyro-interp, ages, rotations (look here first for the important information)
2. Project2.ipynb: The code used for the data gathering for the second draft
3. run_tss_arc_young_gyrointerp.py: The code used to run gyro-interp on the TSS_ARC dataset
4. Characterization_of_Stars...: The original draft, without gyro-interp, ages, or rotations
5. Project.ipynb: The original code used for the first draft
