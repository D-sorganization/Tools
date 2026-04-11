# DbC / DRY / TDD Project Audit (2026-02-27)

Generated automatically from `src/*` and `tests/*`.

| Project                   | Py Files | Test Files | Density | >50 | >80 | Params>4 | pre | post | inv | print | bare except |
| ------------------------- | -------: | ---------: | ------: | --: | --: | -------: | --: | ---: | --: | ----: | ----------: |
| shared                    |      282 |        107 |   0.379 | 252 |  51 |      208 |  62 |   16 |   0 |     0 |           0 |
| data_processing           |      104 |         24 |   0.231 | 122 |  22 |       96 |   0 |    0 |   0 |     0 |           0 |
| electrode_advisor         |       44 |          1 |   0.023 |  51 |  18 |       57 |   0 |    0 |   0 |     0 |           0 |
| tools                     |       54 |          5 |   0.093 |  35 |   8 |       13 |   0 |    0 |   0 |     0 |           0 |
| rotation_converter        |       18 |          9 |   0.500 |  26 |   8 |       29 |   0 |    0 |   0 |     0 |           0 |
| scientific_modeling       |       44 |          8 |   0.182 |  25 |   4 |       21 |   0 |    0 |   0 |     0 |           0 |
| python                    |       52 |         98 |   1.885 |  15 |   3 |       11 |   0 |    0 |   0 |     0 |           0 |
| function_generator        |       11 |          1 |   0.091 |   6 |   3 |        0 |   0 |    0 |   0 |     0 |           0 |
| document_processing       |       34 |          8 |   0.235 |  14 |   2 |       10 |   0 |    0 |   0 |     0 |           0 |
| web_applications          |       28 |         15 |   0.536 |   9 |   2 |        8 |   0 |    0 |   0 |     0 |           0 |
| gasification_equilibrium  |       20 |          6 |   0.300 |   7 |   2 |       12 |   0 |    0 |   0 |    14 |           0 |
| humanoid_builder_gui      |        9 |          1 |   0.111 |   4 |   2 |        0 |   0 |    0 |   0 |     0 |           0 |
| flow_rate_converter       |        9 |          1 |   0.111 |   3 |   2 |        0 |   0 |    0 |   0 |     0 |           0 |
| glass_bath_fea            |       26 |         11 |   0.423 |  10 |   1 |        3 |   0 |    0 |   0 |     0 |           0 |
| steam_engine_calculator   |        9 |          1 |   0.111 |   5 |   1 |        0 |   0 |    0 |   0 |     0 |           0 |
| multi_param_analysis      |        9 |          1 |   0.111 |   4 |   1 |        1 |   0 |    0 |   0 |     0 |           0 |
| optimizer_gui             |        9 |          1 |   0.111 |   4 |   1 |        1 |   0 |    0 |   0 |     0 |           0 |
| urdf_builder_gui          |        9 |          1 |   0.111 |   3 |   1 |        0 |   0 |    0 |   0 |     0 |           0 |
| financial_calculator      |       10 |          1 |   0.100 |   2 |   1 |        1 |   0 |    0 |   0 |     0 |           0 |
| wgs_reactor               |        9 |          1 |   0.111 |   2 |   1 |        1 |   0 |    0 |   0 |     0 |           0 |
| trc_vessel_designer       |       11 |          1 |   0.091 |   2 |   1 |        0 |   0 |    0 |   0 |     0 |           0 |
| ode_solver                |        9 |          1 |   0.111 |   2 |   1 |        0 |   0 |    0 |   0 |     0 |           0 |
| flare_calculator          |        9 |          1 |   0.111 |   5 |   0 |        1 |   0 |    0 |   0 |     0 |           0 |
| inertia_calculator        |        9 |          1 |   0.111 |   3 |   0 |        2 |   0 |    0 |   0 |     0 |           0 |
| acid_gas_dewpoint         |       11 |          1 |   0.091 |   3 |   0 |        0 |   0 |    0 |   0 |     0 |           0 |
| c3d_viewer                |        9 |          1 |   0.111 |   3 |   0 |        0 |   0 |    0 |   0 |     0 |           0 |
| scrubber_calculator       |        9 |          1 |   0.111 |   3 |   0 |        0 |   0 |    0 |   0 |     0 |           0 |
| pressure_drop_calculator  |       11 |          1 |   0.091 |   2 |   0 |        0 |   0 |    0 |   0 |     0 |           0 |
| thermal_profile_predictor |        9 |          1 |   0.111 |   2 |   0 |        0 |   0 |    0 |   0 |     0 |           0 |
| baghouse_calculator       |       10 |          1 |   0.100 |   1 |   0 |        1 |   0 |    0 |   0 |     0 |           0 |
| media_processing          |       11 |          2 |   0.182 |   1 |   0 |        0 |   0 |    0 |   0 |     0 |           0 |
| signal_processing_studio  |        8 |          1 |   0.125 |   0 |   0 |        1 |   0 |    0 |   0 |     0 |           0 |
| hcl_reactor               |        0 |          0 |   0.000 |   0 |   0 |        0 |   0 |    0 |   0 |     0 |           0 |
| matlab                    |        0 |          0 |   0.000 |   0 |   0 |        0 |   0 |    0 |   0 |     0 |           0 |
| psa_package               |        4 |          0 |   0.000 |   0 |   0 |        0 |   0 |    0 |   0 |     0 |           0 |
| verification              |        3 |          0 |   0.000 |   0 |   0 |        0 |   0 |    0 |   0 |     0 |           0 |
| syngas_compression        |       10 |          1 |   0.100 |   0 |   0 |        0 |   0 |    0 |   0 |     0 |           0 |
| syngas_water_calculator   |        9 |          1 |   0.111 |   0 |   0 |        0 |   0 |    0 |   0 |     0 |           0 |
