"""Flare Calculator
        )
        return result

    def calculate_radiation_zones(self, flare_design: FlareDesign) -> dict[str, float]:
        """Calculate radiation zones around the flare.

        Args:
            flare_design: Flare design parameters

        Returns:
            Dictionary with zone distances (m)

        Preconditions:
            flare_design must not be None
            flare_design.heat_release must be non-negative
        """
        require(flare_design is not None, "flare_design must be provided")
        require(
            flare_design.heat_release >= 0,
            "flare_design.heat_release must be non-negative",
            flare_design.heat_release,
        )
        zones = {
            "lethal": 0.0,  # 37.5 kW/m²
            "damage": 0.0,  # 12.5 kW/m²
            "safe": 0.0,  # 1.6 kW/m²
            "comfort": 0.0,  # 0.5 kW/m²
        }

        emissivity = FLARE_FLAME_EMISSIVITY
        heat_release = flare_design.heat_release

        # Calculate distances for each radiation level
        radiation_levels = {
            "lethal": RADIATION_LETHAL,
            "damage": RADIATION_DAMAGE,
            "safe": RADIATION_SAFE,
            "comfort": RADIATION_COMFORT,
        }

        for zone, level in radiation_levels.items():
            if level > 0:
                # Distance based on point source model
                distance = math.sqrt(emissivity * heat_release / (4 * math.pi * level))
                zones[zone] = distance

        return zones

    def calculate_combustion_efficiency(
        self,
        gas_composition: dict[str, float],
        temperature: float,
        pressure: float,
    ) -> float:
        """Calculate combustion efficiency.

        Args:
            gas_composition: Gas composition (mol%)
            temperature: Gas temperature (K)
            pressure: Gas pressure (bar)

        Returns:
            Combustion efficiency (0-1)

        Preconditions:
            gas_composition must not be empty
            temperature > 0 K
            pressure > 0 bar
        """
        require(len(gas_composition) > 0, "gas_composition must not be empty")
        check_temperature(temperature, "temperature")
        check_pressure(pressure, "pressure")
        efficiency = FLARE_BASE_EFFICIENCY  # Base efficiency

        # Normalize factors
        total = sum(gas_composition.values()) or 1.0

        # Factors based on mole fractions
        h2_frac = gas_composition.get("H2", 0) / total
        co_frac = gas_composition.get("CO", 0) / total
        h2s_frac = gas_composition.get("H2S", 0) / total

        if h2_frac > FLARE_H2_THRESHOLD:
            efficiency += FLARE_H2_EFFICIENCY_BOOST

        if co_frac > FLARE_CO_THRESHOLD:
            efficiency -= FLARE_CO_EFFICIENCY_PENALTY

        if h2s_frac > FLARE_H2S_THRESHOLD:
            efficiency -= FLARE_H2S_EFFICIENCY_PENALTY

        # Temperature effects
        if temperature < FLARE_COLD_TEMP_K:
            efficiency -= FLARE_COLD_TEMP_PENALTY
        elif temperature > FLARE_HOT_TEMP_K:
            efficiency += FLARE_HOT_TEMP_BOOST

        return max(FLARE_MIN_EFFICIENCY, min(FLARE_MAX_EFFICIENCY, efficiency))
