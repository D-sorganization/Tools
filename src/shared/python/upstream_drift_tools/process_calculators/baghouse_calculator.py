"""Baghouse Calculator

        outlet_temp_c, flow_acfm, flow_scfm = self._calculate_outlet_thermal(
            gas_flow_kg_s,
            inlet_temp_k,
            pressure_pa,
            composition,
            heat_loss_w,
        )

        (
            carbon_removed,
            ash_removed,
            total_solids,
            fill_hrs,
            fill_days,
            c_fill,
            a_fill,
        ) = self._calculate_drum_sizing(
            solid_carbon_in_kg_hr,
            ash_in_kg_hr,
            carbon_removal_efficiency,
            ash_removal_efficiency,
            drum_volume_m3,
            solid_density_kg_m3,
        )

        air_to_cloth = flow_acfm / bag_area_ft2 if bag_area_ft2 > 0 else 0.0

        ash_stream_comp = {
            "carbon_fraction": (
                carbon_removed / total_solids if total_solids > 0 else 0.0
            ),
            "ash_fraction": (ash_removed / total_solids if total_solids > 0 else 0.0),
        }

        return BaghouseResult(
            carbon_removed_rate=carbon_removed,
            ash_removed_rate=ash_removed,
            total_solids_removed_rate=total_solids,
            drum_fill_time_hours=fill_hrs,
            drum_fill_time_days=fill_days,
            carbon_only_fill_time_hours=c_fill,
            ash_only_fill_time_hours=a_fill,
            clean_gas_flow_rate=gas_flow_kg_s * SECONDS_PER_HOUR,
            flow_acfm=flow_acfm,
            flow_scfm=flow_scfm,
            air_to_cloth_ratio=air_to_cloth,
            outlet_temperature_c=outlet_temp_c,
            ash_stream_composition=ash_stream_comp,
            removal_efficiency={
                "carbon": carbon_removal_efficiency * 100.0,
                "ash": ash_removal_efficiency * 100.0,
            },
        )
