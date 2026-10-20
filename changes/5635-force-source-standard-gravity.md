---
issue: 5635
summary: "Regenerate force-source comparison artifacts and Colab notebook at standard gravity"
---

#5668 moved every pendulum-web preset to `STANDARD_GRAVITY` (9.80665 m/s^2)
but the checked-in force-source comparison artifacts, generated from
`PRESETS[0]`, still recorded g = 9.81. Rerun `npm run generate:force-source`,
update the Colab notebook default, and add a regression test that the
artifacts record `STANDARD_GRAVITY`.
