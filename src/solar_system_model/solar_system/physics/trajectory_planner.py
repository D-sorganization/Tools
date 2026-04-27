# TRACKED_TASK: see #2310 — architecture debt extraction schedule

"""
Trajectory Planner

        # Get orbital radii at departure
        origin_state = origin.get_state_at_time(departure_date)
        r1 = float(np.linalg.norm(origin_state.position))

        # Estimate destination radius (use semi-major axis for planning)
        r2 = (
            destination.orbital_elements.semi_major_axis * AU
            if destination.orbital_elements
            else 0.0
        )

        if transfer_type == TransferType.HOHMANN:
            return self._calculate_hohmann(origin, destination, departure_date, r1, r2)
        elif transfer_type == TransferType.BI_ELLIPTIC:
            return self._calculate_bi_elliptic(
                origin, destination, departure_date, r1, r2
            )
        elif transfer_type == TransferType.GRAVITY_ASSIST:
            raise ValueError("Use calculate_gravity_assist to specify an assist body")
        else:
            # Default to Hohmann
            return self._calculate_hohmann(origin, destination, departure_date, r1, r2)

    def _calculate_hohmann(
        self,
        origin: CelestialBody,
        destination: CelestialBody,
        departure_date: float,
        r1: float,
        r2: float,
    ) -> TransferTrajectory:
        """Calculate a Hohmann transfer trajectory."""
        # Get transfer parameters
        if not (origin is not None):
            raise ValueError("origin must be provided")
        dv1, dv2, tof, a_transfer = self.hohmann_transfer(r1, r2)
        tof_days = tof / SECONDS_PER_DAY
        arrival_date = departure_date + tof_days

        # Get states
        origin_state = origin.get_state_at_time(departure_date)
        dest_state = destination.get_state_at_time(arrival_date)

        # Calculate phase angle
        dest_state_at_departure = destination.get_state_at_time(departure_date)
        phase = OrbitalMechanics.phase_angle(
            origin_state.position, dest_state_at_departure.position
        )

        # Calculate delta-v direction (prograde at departure)
        v_unit = origin_state.velocity / np.linalg.norm(origin_state.velocity)
        delta_v1_vec = v_unit * dv1 if r2 > r1 else -v_unit * dv1

        # Create maneuver nodes
        maneuvers = [
            ManeuverNode(
                time=departure_date,
                position=origin_state.position,
                delta_v=delta_v1_vec,
                delta_v_magnitude=dv1,
                description=f"Trans-{destination.name} Injection burn",
            ),
            ManeuverNode(
                time=arrival_date,
                position=dest_state.position,
                delta_v=-v_unit * dv2,  # Retrograde at arrival
                delta_v_magnitude=dv2,
                description=f"{destination.name} Orbit Insertion burn",
            ),
        ]

        # Generate trajectory points
        trajectory_points = self._generate_trajectory_points(
            origin_state, a_transfer, r1, r2, departure_date, tof_days
        )

        return TransferTrajectory(
            origin=origin.name,
            destination=destination.name,
            transfer_type=TransferType.HOHMANN,
            departure_time=departure_date,
            arrival_time=arrival_date,
            time_of_flight=tof_days,
            total_delta_v=dv1 + dv2,
            maneuvers=maneuvers,
            trajectory_points=trajectory_points,
            phase_angle=math.degrees(phase),
        )

    def _calculate_bi_elliptic(
        self,
        origin: CelestialBody,
        destination: CelestialBody,
        departure_date: float,
        r1: float,
        r2: float,
    ) -> TransferTrajectory:
        """Calculate a bi-elliptic transfer trajectory."""
        # Use intermediate radius 1.5x the larger orbit
        if not (origin is not None):
            raise ValueError("origin must be provided")
        r_intermediate = max(r1, r2) * 1.5

        dv1, dv2, dv3, tof = self.bi_elliptic_transfer(r1, r2, r_intermediate)
        tof_days = tof / SECONDS_PER_DAY
        arrival_date = departure_date + tof_days

        origin_state = origin.get_state_at_time(departure_date)

        # Calculate intermediate maneuver time
        a1 = (r1 + r_intermediate) / 2
        t_intermediate = (
            OrbitalMechanics.orbital_period(a1, self.mu) / 2 / SECONDS_PER_DAY
        )
        intermediate_date = departure_date + t_intermediate

        v_unit = origin_state.velocity / np.linalg.norm(origin_state.velocity)

        maneuvers = [
            ManeuverNode(
                time=departure_date,
                position=origin_state.position,
                delta_v=v_unit * dv1,
                delta_v_magnitude=dv1,
                description="First transfer burn",
            ),
            ManeuverNode(
                time=intermediate_date,
                position=np.array([r_intermediate, 0, 0]),  # Approximate
                delta_v=v_unit * dv2,
                delta_v_magnitude=dv2,
                description="Intermediate plane change",
            ),
            ManeuverNode(
                time=arrival_date,
                position=np.array([r2, 0, 0]),  # Approximate
                delta_v=-v_unit * dv3,
                delta_v_magnitude=dv3,
                description="Orbit insertion",
            ),
        ]

        return TransferTrajectory(
            origin=origin.name,
            destination=destination.name,
            transfer_type=TransferType.BI_ELLIPTIC,
            departure_time=departure_date,
            arrival_time=arrival_date,
            time_of_flight=tof_days,
            total_delta_v=dv1 + dv2 + dv3,
            maneuvers=maneuvers,
            trajectory_points=[],
            phase_angle=0.0,
        )

    def calculate_gravity_assist(
        self,
        origin: CelestialBody,
        assist_body: CelestialBody,
        destination: CelestialBody,
        departure_date: float,
        periapsis_altitude_km: float = 300.0,
    ) -> TransferTrajectory:
        """Plan a patched-conic gravity assist sequence."""

        if not (origin is not None):
            raise ValueError("origin must be provided")
        first_leg = self.calculate_transfer(
            origin, assist_body, departure_date, TransferType.HOHMANN
        )
        assist_arrival = first_leg.arrival_time

        flyby_radius = (assist_body.radius + periapsis_altitude_km) * 1000.0
        flyby_speed = (
            math.sqrt(max(assist_body.gm, 0.0) / flyby_radius)
            if assist_body.gm > 0
            else 0.0
        )
        assist_heliocentric_speed = float(
            np.linalg.norm(assist_body.get_state_at_time(assist_arrival).velocity)
        )

        second_leg = self.calculate_transfer(
            assist_body,
            destination,
            assist_arrival + 0.5,
            TransferType.HOHMANN,
        )

        assist_bonus = flyby_speed + assist_heliocentric_speed * 0.3
        total_delta_v = max(
            first_leg.total_delta_v + second_leg.total_delta_v - assist_bonus, 0.0
        )

        maneuvers = first_leg.maneuvers + second_leg.maneuvers
        trajectory_points = first_leg.trajectory_points + second_leg.trajectory_points

        return TransferTrajectory(
            origin=origin.name,
            destination=destination.name,
            transfer_type=TransferType.GRAVITY_ASSIST,
            departure_time=departure_date,
            arrival_time=second_leg.arrival_time,
            time_of_flight=second_leg.arrival_time - departure_date,
            total_delta_v=total_delta_v,
            maneuvers=maneuvers,
            trajectory_points=trajectory_points,
            phase_angle=second_leg.phase_angle,
        )

    def _generate_trajectory_points(
        self,
        initial_state: StateVector,
        a_transfer: float,
        r_start: float,
        r_end: float,
        start_date: float,
        duration_days: float,
        num_points: int = 100,
    ) -> list[StateVector]:
        """Generate points along a transfer trajectory."""
        if not (initial_state is not None):
            raise ValueError("initial_state must be provided")
        points = []

        # Determine orbit parameters
        r_min = min(r_start, r_end)
        r_max = max(r_start, r_end)
        e = OrbitalMechanics.eccentricity_from_apsides(r_min, r_max)

        # Determine anomaly range
        if r_start < r_end:
            # Outward: Periapsis -> Apoapsis
            nu_start = 0.0
            nu_end = math.pi
        else:
            # Inward: Apoapsis -> Periapsis
            nu_start = math.pi
            nu_end = 2 * math.pi

        # Get initial position angle
        pos = initial_state.position
        initial_angle = math.atan2(pos[1], pos[0])

        # Optimization: Precalculate constants
        mu = self.mu
        h = math.sqrt(mu * a_transfer * (1 - e**2))
        n = math.sqrt(mu / a_transfer**3)
        sqrt_1_minus_e = math.sqrt(1 - e)
        sqrt_1_plus_e = math.sqrt(1 + e)
        parameter = a_transfer * (1 - e**2)  # Semi-latus rectum

        # Calculate initial Mean Anomaly
        sin_nu_start_2 = math.sin(nu_start / 2)
        cos_nu_start_2 = math.cos(nu_start / 2)
        E_start = 2 * math.atan2(
            sqrt_1_minus_e * sin_nu_start_2, sqrt_1_plus_e * cos_nu_start_2
        )
        M_start = E_start - e * math.sin(E_start)

        # Precalculate angle offset constants
        # angle = nu + (initial_angle - nu_start)
        phi = initial_angle - nu_start
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)

        nu_start_2 = nu_start / 2
        nu_range_2 = (nu_end - nu_start) / 2

        # Generate points
        for i in range(num_points + 1):
            fraction = i / num_points

            # Use half-angle for iteration
            nu_2 = nu_start_2 + nu_range_2 * fraction
            sin_nu_2 = math.sin(nu_2)
            cos_nu_2 = math.cos(nu_2)

            # Double angle formulas for nu
            sin_nu = 2 * sin_nu_2 * cos_nu_2
            cos_nu = cos_nu_2**2 - sin_nu_2**2

            # 1. Radius
            r = parameter / (1 + e * cos_nu)

            # 2. Velocity components
            v_r = mu * e * sin_nu / h
            v_t = h / r

            # 3. Position and Velocity in inertial frame
            # angle = nu + phi
            cos_angle = cos_nu * cos_phi - sin_nu * sin_phi
            sin_angle = sin_nu * cos_phi + cos_nu * sin_phi

            x = r * cos_angle
            y = r * sin_angle
            z = 0.0  # Assuming coplanar

            vx = v_r * cos_angle - v_t * sin_angle
            vy = v_r * sin_angle + v_t * cos_angle
            vz = 0.0

            # 4. Time
            E = 2 * math.atan2(sqrt_1_minus_e * sin_nu_2, sqrt_1_plus_e * cos_nu_2)
            M = E - e * math.sin(E)

            delta_M = M - M_start
            if delta_M < 0:
                delta_M += 2 * math.pi

            dt = delta_M / n
            time = start_date + dt / SECONDS_PER_DAY

            points.append(
                StateVector(
                    position=np.array([x, y, z]),
                    velocity=np.array([vx, vy, vz]),
                    time=time,
                )
            )

        return points

    def create_spacecraft_from_transfer(
        self, trajectory: TransferTrajectory, name: str = "Spacecraft"
    ) -> Spacecraft:
        """
        Create a Spacecraft object from a transfer trajectory.

        Args:
            trajectory: The calculated transfer trajectory
            name: Name for the spacecraft

        Returns:
            Spacecraft with the trajectory set
        """
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        spacecraft = Spacecraft(name=name, trajectory=trajectory.trajectory_points)
        return spacecraft

    def get_transfer_summary(
        self, origin: CelestialBody, destination: CelestialBody
    ) -> dict[str, Any]:
        """
        Get a summary of transfer options between two bodies.

        Args:
            origin: Origin body
            destination: Destination body

        Returns:
            Dictionary with transfer information
        """
        if not (origin is not None):
            raise ValueError("origin must be provided")
        r1 = (
            origin.orbital_elements.semi_major_axis * AU
            if origin.orbital_elements
            else 0.0
        )
        r2 = (
            destination.orbital_elements.semi_major_axis * AU
            if destination.orbital_elements
            else 0.0
        )

        # Hohmann transfer
        dv1, dv2, tof, a_transfer = self.hohmann_transfer(r1, r2)
        phase = self.hohmann_phase_angle(r1, r2)

        # Synodic period
        synodic = self.synodic_period_planets(origin, destination)

        summary = {
            "Route": f"{origin.name} → {destination.name}",
            "Distance": f"{abs(r2 - r1) / AU:.2f} AU",
            "Hohmann Transfer": {
                "Departure Δv": f"{dv1:.1f} m/s",
                "Arrival Δv": f"{dv2:.1f} m/s",
                "Total Δv": f"{dv1 + dv2:.1f} m/s",
                "Time of Flight": f"{tof / SECONDS_PER_DAY:.1f} days",
                "Phase Angle": f"{phase:.1f}°",
            },
            "Synodic Period": f"{synodic:.1f} days ({synodic / 365.25:.2f} years)",
        }

        return summary
