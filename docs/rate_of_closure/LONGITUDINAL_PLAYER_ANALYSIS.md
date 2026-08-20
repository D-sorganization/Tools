# Longitudinal Player and Population Analysis

Issue: [Tools #4230](https://github.com/D-sorganization/Tools/issues/4230)

The longitudinal workbench requires explicit, user-attested player identity,
session identity, and numeric session order. It never derives identity or time
from a filename, club, source, layout, or row order.

Each session contributes one equal-weight metric mean. The backing data retains
the session row count, sample standard deviation, standard error, and cumulative
mean. With at least three uniquely ordered sessions, each player receives an
ordinary least-squares slope per session, Student-t interval, p-value, `R²`, and
first-to-last observed change. Players with insufficient sessions or constant
inputs remain visible with an explicit unavailable status.

Eligible player slopes are synthesized using inverse-variance fixed effects and
a DerSimonian-Laird random-effects estimate. The result reports `Q`, `tau²`,
`I²`, the random-effect interval, and a normal-approximation probability that
the slope points in the user-declared improvement direction. This probability
is conditional on the model and declared higher/lower-is-better direction; it
is not the probability that practice caused improvement.

These analyses are observational. Equipment, intent, monitor, environment,
selection, fatigue, injuries, coaching, and regression to the mean can explain
apparent change. Confirmation requires prospectively defined outcomes,
consistent measurement, and an appropriate comparison or causal design.
