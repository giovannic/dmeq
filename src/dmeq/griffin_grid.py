"""The canonical Griffin age grid.

Transcribed from ``Griffin-2016-R-code/age.txt``: ~2-week resolution through
infancy, coarsening to 2-year steps in adulthood, 165 classes total. Each entry
is the lower edge of a model age class in years; class ``i`` spans
``[ages[i], ages[i+1])`` and the terminal class is an absorbing
``[ages[-1], inf)`` compartment.

``solve`` defaults to a uniform 100 x 1-year grid, which under-resolves the
immunity recursions through infancy. Pass this grid as ``age_bins_years`` to
solve on the grid Griffin fitted.
"""

GRIFFIN_AGE_GRID = (
    0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48,
    0.52, 0.56, 0.6, 0.64, 0.68, 0.72, 0.76, 0.8, 0.84, 0.88, 0.92, 0.96, 1.0,
    1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7,
    1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.06667, 2.13333, 2.2, 2.26667, 2.33333,
    2.4, 2.46667, 2.53333, 2.6, 2.66667, 2.73333, 2.8, 2.86666, 2.93334, 3.0,
    3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.12501, 4.25, 4.37501,
    4.5, 4.62501, 4.75, 4.87501, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.25, 6.5, 6.75,
    7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0,
    10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0,
    16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 23.0, 24.0,
    25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 42.0,
    44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0, 66.0,
    68.0, 70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0, 84.0, 86.0, 88.0, 90.0,
    92.0, 94.0, 96.0, 98.0, 100.0,
)
