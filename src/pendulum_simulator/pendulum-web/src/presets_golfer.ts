/**
 * Preset configurations for Golfer Upper-Body (8-DOF) model.
 *
 * Realistic defaults:
 *   Hub (torso): m_hub = 3.0 kg, L_hub = 0.25 m
 *   Right arm: m_r_upper = 3.0 kg, L_r_upper = 0.30 m
 *              m_r_fore = 1.5 kg, L_r_fore = 0.25 m
 *   Left arm: m_l_upper = 3.0 kg, L_l_upper = 0.30 m
 *             m_l_fore = 1.5 kg, L_l_fore = 0.25 m
 *   Club: m_club = 0.30 kg, L_club = 1.10 m, m_clubhead = 0.20 kg
 */

import type { GolferParams, TorqueFuncGolfer } from "./physics_golfer";
import {
  makeGolferParams,
  makePolynomialTorque_golfer,
} from "./physics_golfer";

export interface PresetGolfer {
  name: string;
  params: GolferParams;
  // Angles in degrees
  theta_hub_deg: number;
  alpha_rs_deg: number;
  alpha_re_deg: number;
  alpha_rh_deg: number;
  alpha_ls_deg: number;
  alpha_le_deg: number;
  alpha_lh_deg: number;
  theta_club_deg: number;
  // Angular velocities
  dtheta_hub: number;
  dalpha_rs: number;
  dalpha_re: number;
  dalpha_rh: number;
  dalpha_ls: number;
  dalpha_le: number;
  dalpha_lh: number;
  dtheta_club: number;
  // Torque coefficients
  torqueFunc: TorqueFuncGolfer;
  coeff_hub: number[];
  coeff_rs: number[];
  coeff_re: number[];
  coeff_rh: number[];
  coeff_ls: number[];
  coeff_le: number[];
  coeff_lh: number[];
  tEnd: number;
  description: string;
}

const _preset_golfer = (
  name: string,
  // Masses
  m_hub: number,
  m_r_upper: number,
  m_r_fore: number,
  m_l_upper: number,
  m_l_fore: number,
  m_club: number,
  m_clubhead: number,
  // Lengths
  L_hub: number,
  L_r_upper: number,
  L_r_fore: number,
  L_l_upper: number,
  L_l_fore: number,
  L_club: number,
  // Offsets
  d_rs: number,
  d_ls: number,
  grip_right: number,
  grip_left: number,
  // Damping
  b_hub: number,
  b_rs: number,
  b_re: number,
  b_rh: number,
  b_ls: number,
  b_le: number,
  b_lh: number,
  // Initial angles (deg)
  theta_hub_deg: number,
  alpha_rs_deg: number,
  alpha_re_deg: number,
  alpha_rh_deg: number,
  alpha_ls_deg: number,
  alpha_le_deg: number,
  alpha_lh_deg: number,
  theta_club_deg: number,
  // Initial velocities
  dtheta_hub: number,
  dalpha_rs: number,
  dalpha_re: number,
  dalpha_rh: number,
  dalpha_ls: number,
  dalpha_le: number,
  dalpha_lh: number,
  dtheta_club: number,
  // Torque coefficients
  c_hub: number[],
  c_rs: number[],
  c_re: number[],
  c_rh: number[],
  c_ls: number[],
  c_le: number[],
  c_lh: number[],
  tEnd: number,
  description: string,
): PresetGolfer => {
  const params = makeGolferParams({
    m_hub,
    m_r_upper,
    m_r_fore,
    m_l_upper,
    m_l_fore,
    m_club,
    L_hub,
    L_r_upper,
    L_r_fore,
    L_l_upper,
    L_l_fore,
    L_club,
    d_rs,
    d_ls,
    grip_right,
    grip_left,
    m_clubhead,
    g: 9.81,
    b_hub,
    b_rs,
    b_re,
    b_rh,
    b_ls,
    b_le,
    b_lh,
  });

  return {
    name,
    params,
    theta_hub_deg,
    alpha_rs_deg,
    alpha_re_deg,
    alpha_rh_deg,
    alpha_ls_deg,
    alpha_le_deg,
    alpha_lh_deg,
    theta_club_deg,
    dtheta_hub,
    dalpha_rs,
    dalpha_re,
    dalpha_rh,
    dalpha_ls,
    dalpha_le,
    dalpha_lh,
    dtheta_club,
    torqueFunc: makePolynomialTorque_golfer(
      c_hub,
      c_rs,
      c_re,
      c_rh,
      c_ls,
      c_le,
      c_lh,
    ),
    coeff_hub: c_hub,
    coeff_rs: c_rs,
    coeff_re: c_re,
    coeff_rh: c_rh,
    coeff_ls: c_ls,
    coeff_le: c_le,
    coeff_lh: c_lh,
    tEnd,
    description,
  };
};

export const PRESETS_GOLFER: PresetGolfer[] = [
  _preset_golfer(
    "Golfer Upper Body (symmetric swing)",
    // Masses
    3.0,
    3.0,
    1.5,
    3.0,
    1.5,
    0.3,
    0.2,
    // Lengths
    0.25,
    0.3,
    0.25,
    0.3,
    0.25,
    1.1,
    // Offsets
    0.15,
    0.15,
    0.1,
    0.1,
    // Damping
    0.1,
    0.08,
    0.06,
    0.04,
    0.08,
    0.06,
    0.04,
    // Initial angles (deg)
    -30,
    -45,
    30,
    0,
    -45,
    30,
    0,
    0,
    // Initial velocities
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    // Torque coefficients
    [-15, 5],
    [5, -2],
    [0],
    [0],
    [5, -2],
    [0],
    [0],
    2.0,
    "Symmetric golfer swing with equal arm torques, passive hand release.",
  ),
  _preset_golfer(
    "Golfer Upper Body (asymmetric swing)",
    // Masses
    3.0,
    3.0,
    1.5,
    3.0,
    1.5,
    0.3,
    0.2,
    // Lengths
    0.25,
    0.3,
    0.25,
    0.3,
    0.25,
    1.1,
    // Offsets
    0.15,
    0.15,
    0.1,
    0.1,
    // Damping
    0.1,
    0.08,
    0.06,
    0.04,
    0.08,
    0.06,
    0.04,
    // Initial angles (deg)
    -30,
    -45,
    30,
    0,
    -35,
    25,
    0,
    0,
    // Initial velocities
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    // Torque coefficients (right arm stronger)
    [-15, 5],
    [8, -3],
    [0],
    [0],
    [3, -1],
    [0],
    [0],
    2.0,
    "Asymmetric golfer swing with right arm dominance.",
  ),
  _preset_golfer(
    "Free Golfer Body (no torques)",
    // Masses
    3.0,
    3.0,
    1.5,
    3.0,
    1.5,
    0.3,
    0.2,
    // Lengths
    0.25,
    0.3,
    0.25,
    0.3,
    0.25,
    1.1,
    // Offsets
    0.15,
    0.15,
    0.1,
    0.1,
    // Damping
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    // Initial angles (deg)
    0,
    45,
    -30,
    0,
    45,
    -30,
    0,
    0,
    // Initial velocities
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    // Torque coefficients (all zero)
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    3.0,
    "No torques, no damping — pure constraint-driven dynamics.",
  ),
];
