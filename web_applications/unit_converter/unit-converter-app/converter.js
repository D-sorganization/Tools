/**
 * Unit Converter - Enhanced Core Logic
 * Ported from Python gasification model unit converter
 * NIST-compliant conversion factors with custom unit support
 */

// ============================================================================
// CONSTANTS AND REFERENCE DATA
// ============================================================================

const CELSIUS_OFFSET = 273.15;
const RANKINE_RATIO = 5.0 / 9.0;
const SCFM_TO_CU_METER_PER_HOUR_AT_60F = 1.699010795;

// Standard conditions for gas flow
const StandardConditions = {
  STP: { temp: 273.15, pressure: 101325.0, label: 'STP (0°C, 101.325 kPa)' },
  SCFM_60F: { temp: 288.706, pressure: 101325.0, label: 'SCFM at 60°F, 14.696 psia' },
  SCFM_70F: { temp: 294.261, pressure: 101325.0, label: 'SCFM at 70°F, 14.696 psia' },
  NTP: { temp: 293.15, pressure: 101325.0, label: 'NTP (20°C, 101.325 kPa)' },
  SATP: { temp: 298.15, pressure: 100000.0, label: 'SATP (25°C, 1 bar)' }
};

// Gas database with physical properties
const GAS_DATABASE = {
  air: { name: 'Air', mw: 28.97, density_stp: 1.2922, k: 1.4 },
  nitrogen: { name: 'Nitrogen', mw: 28.014, density_stp: 1.2506, k: 1.4 },
  oxygen: { name: 'Oxygen', mw: 31.999, density_stp: 1.4289, k: 1.395 },
  hydrogen: { name: 'Hydrogen', mw: 2.016, density_stp: 0.08988, k: 1.405 },
  methane: { name: 'Methane', mw: 16.043, density_stp: 0.7168, k: 1.321 },
  co: { name: 'Carbon Monoxide', mw: 28.01, density_stp: 1.25, k: 1.4 },
  co2: { name: 'Carbon Dioxide', mw: 44.01, density_stp: 1.9768, k: 1.289 },
  h2o: { name: 'Water Vapor', mw: 18.015, density_stp: 0.00485, k: 1.33 }
};

// Conversion factors - all NIST-standard values
const CONVERSION_FACTORS = {
  length: {
    m: 1.0,
    cm: 0.01,
    mm: 0.001,
    um: 1.0e-6,
    µm: 1.0e-6,
    nm: 1.0e-9,
    Å: 1.0e-10,
    mil: 2.54e-5,
    km: 1000.0,
    in: 0.0254,
    ft: 0.3048,
    yd: 0.9144,
    mi: 1609.344
  },

  mass: {
    kg: 1.0,
    g: 0.001,
    mg: 1.0e-6,
    µg: 1.0e-9,
    tonne: 1000.0,
    lb: 0.45359237,
    oz: 0.028349523125,
    ton: 907.18474,
    long_ton: 1016.0469088,
    slug: 14.59390294,
    grain: 6.479891e-5
  },

  volume: {
    m3: 1.0,
    'm³': 1.0,
    L: 0.001,
    mL: 1.0e-6,
    cm3: 1.0e-6,
    'cm³': 1.0e-6,
    mm3: 1.0e-9,
    'mm³': 1.0e-9,
    ft3: 0.028316846592,
    'ft³': 0.028316846592,
    in3: 1.6387064e-5,
    'in³': 1.6387064e-5,
    gal: 0.003785411784,
    imp_gal: 0.00454609,
    qt: 0.000946352946,
    pt: 0.000473176473,
    fl_oz: 2.95735295625e-5,
    bbl: 0.158987294928
  },

  pressure: {
    Pa: 1.0,
    kPa: 1000.0,
    MPa: 1.0e6,
    GPa: 1.0e9,
    bar: 100000.0,
    atm: 101325.0,
    mbar: 100.0,
    psi: 6894.757293168,
    torr: 133.322387415,
    mmHg: 133.322387415,
    inHg: 3386.389,
    inH2O: 249.082,
    ftH2O: 2989.07,
    cmH2O: 98.0665
  },

  energy: {
    J: 1.0,
    kJ: 1000.0,
    MJ: 1.0e6,
    GJ: 1.0e9,
    BTU: 1055.05585262,
    cal: 4.184,
    kcal: 4184.0,
    Wh: 3600.0,
    kWh: 3.6e6,
    MWh: 3.6e9,
    therm: 105505585.262,
    erg: 1.0e-7,
    eV: 1.602176634e-19
  },

  power: {
    W: 1.0,
    kW: 1000.0,
    MW: 1.0e6,
    GW: 1.0e9,
    hp: 745.69987158227022,
    metric_hp: 735.49875,
    'BTU/hr': 0.29307107017222,
    'cal/s': 4.184,
    'kcal/hr': 1.163,
    'ft·lbf/s': 1.3558179483314004
  },

  mass_flow: {
    'kg/s': 1.0,
    'kg/min': 1.0 / 60.0,
    'kg/hr': 1.0 / 3600.0,
    'kg/day': 1.0 / 86400.0,
    'g/s': 0.001,
    'g/min': 0.001 / 60.0,
    'g/hr': 0.001 / 3600.0,
    'g/day': 0.001 / 86400.0,
    'lb/hr': 0.45359237 / 3600.0,
    'lb/min': 0.45359237 / 60.0,
    'lb/s': 0.45359237,
    'lb/day': 0.45359237 / 86400.0,
    'tonne/hr': 1000.0 / 3600.0,
    'tonne/day': 1000.0 / 86400.0,
    'ton/hr': 907.18474 / 3600.0,
    'ton/day': 907.18474 / 86400.0
  },

  area: {
    m2: 1.0,
    'm²': 1.0,
    cm2: 1.0e-4,
    'cm²': 1.0e-4,
    mm2: 1.0e-6,
    'mm²': 1.0e-6,
    km2: 1.0e6,
    'km²': 1.0e6,
    in2: 6.4516e-4,
    'in²': 6.4516e-4,
    ft2: 0.09290304,
    'ft²': 0.09290304,
    yd2: 0.83612736,
    'yd²': 0.83612736,
    acre: 4046.8564224,
    hectare: 10000.0
  },

  time: {
    s: 1.0,
    min: 60.0,
    hr: 3600.0,
    day: 86400.0
  },

  volumetric_flow: {
    'm3/s': 1.0,
    'm³/s': 1.0,
    'm3/min': 1.0 / 60.0,
    'm³/min': 1.0 / 60.0,
    'm3/hr': 1.0 / 3600.0,
    'm³/hr': 1.0 / 3600.0,
    'm3/day': 1.0 / 86400.0,
    'm³/day': 1.0 / 86400.0,
    'L/s': 0.001,
    'L/min': 0.001 / 60.0,
    'L/hr': 0.001 / 3600.0,
    'L/day': 0.001 / 86400.0,
    'ft3/s': 0.028316846592,
    'ft³/s': 0.028316846592,
    'ft3/min': 0.028316846592 / 60.0,
    'ft³/min': 0.028316846592 / 60.0,
    'ft3/hr': 0.028316846592 / 3600.0,
    'ft³/hr': 0.028316846592 / 3600.0,
    'ft3/day': 0.028316846592 / 86400.0,
    'ft³/day': 0.028316846592 / 86400.0,
    'gal/min': 0.003785411784 / 60.0,
    gpm: 0.003785411784 / 60.0,
    'gal/hr': 0.003785411784 / 3600.0,
    'gal/day': 0.003785411784 / 86400.0,
    'imp_gal/min': 0.00454609 / 60.0,
    'imp_gal/hr': 0.00454609 / 3600.0,
    'imp_gal/day': 0.00454609 / 86400.0,
    'bbl/day': 0.158987294928 / 86400.0,
    'bbl/hr': 0.158987294928 / 3600.0
  },

  density: {
    'kg/m3': 1.0,
    'kg/m³': 1.0,
    'g/L': 1.0,
    'g/cm3': 1000.0,
    'g/cm³': 1000.0,
    'lb/ft3': 16.01846337396,
    'lb/ft³': 16.01846337396,
    'lb/gal': 119.8264273,
    'kg/L': 1000.0
  },

  dynamic_viscosity: {
    'Pa·s': 1.0,
    'Pa.s': 1.0,
    'mPa·s': 0.001,
    cP: 0.001,
    P: 0.1,
    'lb/ft·s': 1.4881639436
  },

  kinematic_viscosity: {
    'm2/s': 1.0,
    'm²/s': 1.0,
    cSt: 1.0e-6,
    'cm2/s': 1.0e-4,
    'cm²/s': 1.0e-4,
    St: 1.0e-4,
    'ft2/s': 0.09290304,
    'ft²/s': 0.09290304
  },

  thermal_conductivity: {
    'W/m·K': 1.0,
    'W/mK': 1.0,
    'BTU/(ft·hr·°F)': 1.7307346664,
    'cal/(cm·s·°C)': 418.4
  },

  heat_transfer: {
    'W/m2·K': 1.0,
    'W/m²K': 1.0,
    'BTU/(ft2·hr·°F)': 5.6782633411,
    'BTU/(ft²·hr·°F)': 5.6782633411
  },

  specific_heat: {
    'J/kg·K': 1.0,
    'J/kgK': 1.0,
    'kJ/kg·K': 1000.0,
    'kJ/kgK': 1000.0,
    'BTU/lb·°F': 4186.8,
    'cal/g·°C': 4186.8
  }
};

// Heating value conversions (mass-based units convert to MJ/kg, volumetric need density)
const HEATING_VALUE_FACTORS = {
  'MJ/kg': 1.0,
  'kJ/kg': 0.001,
  'J/kg': 1e-6,
  'cal/g': 0.004184,
  'kcal/kg': 0.004184,
  'BTU/lb': 0.00232444,
  'kWh/kg': 3.6,
  // Volumetric units require density
  'MJ/Nm3': null,
  'MJ/Nm³': null,
  'BTU/scf': null,
  'kWh/Nm3': null,
  'kWh/Nm³': null
};

// Unit aliases for user-friendly input
const UNIT_ALIASES = {
  meter: 'm',
  meters: 'm',
  metre: 'm',
  metres: 'm',
  centimeter: 'cm',
  centimeters: 'cm',
  centimetre: 'cm',
  millimeter: 'mm',
  millimeters: 'mm',
  micrometer: 'um',
  micron: 'um',
  nanometer: 'nm',
  kilometer: 'km',
  inch: 'in',
  inches: 'in',
  foot: 'ft',
  feet: 'ft',
  yard: 'yd',
  yards: 'yd',
  mile: 'mi',
  miles: 'mi',
  kilogram: 'kg',
  kilograms: 'kg',
  gram: 'g',
  grams: 'g',
  milligram: 'mg',
  microgram: 'µg',
  pound: 'lb',
  pounds: 'lb',
  lbs: 'lb',
  ounce: 'oz',
  ounces: 'oz',
  liter: 'L',
  liters: 'L',
  litre: 'L',
  litres: 'L',
  milliliter: 'mL',
  gallon: 'gal',
  gallons: 'gal',
  quart: 'qt',
  pint: 'pt',
  pascal: 'Pa',
  kilopascal: 'kPa',
  megapascal: 'MPa',
  atmosphere: 'atm',
  joule: 'J',
  joules: 'J',
  kilojoule: 'kJ',
  megajoule: 'MJ',
  gigajoule: 'GJ',
  calorie: 'cal',
  calories: 'cal',
  kilocalorie: 'kcal',
  watt: 'W',
  watts: 'W',
  kilowatt: 'kW',
  megawatt: 'MW',
  horsepower: 'hp',
  second: 's',
  seconds: 's',
  minute: 'min',
  minutes: 'min',
  hour: 'hr',
  hours: 'hr',
  kelvin: 'K',
  celsius: 'C',
  fahrenheit: 'F',
  rankine: 'R'
};

// Optimization: Cache for reverse alias lookup
let _REVERSE_ALIASES_CACHE = null;
// Optimization: Cache for flattened searchable units
let _SEARCH_CACHE = null;

// Security: Prevent prototype pollution
function isValidKey(key) {
  return (
    key !== '__proto__' && key !== 'constructor' && key !== 'prototype' && typeof key === 'string'
  );
}

function getReverseAliases() {
  if (_REVERSE_ALIASES_CACHE) {
    return _REVERSE_ALIASES_CACHE;
  }

  _REVERSE_ALIASES_CACHE = {};
  for (const [alias, unit] of Object.entries(UNIT_ALIASES)) {
    if (!_REVERSE_ALIASES_CACHE[unit]) {
      _REVERSE_ALIASES_CACHE[unit] = [];
    }
    _REVERSE_ALIASES_CACHE[unit].push(alias);
  }
  return _REVERSE_ALIASES_CACHE;
}

// ============================================================================
// CUSTOM UNIT STORAGE
// ============================================================================

class CustomUnitManager {
  constructor() {
    this.customUnits = {}; // { category: { unit: factor } }
    this.customAliases = {}; // { unit: [aliases] }
    this.loadFromStorage();
  }

  addUnit(category, unit, referenceUnit, factorToReference, aliases = []) {
    // Security validation
    if (!isValidKey(category) || !isValidKey(unit)) {
      throw new Error('Invalid category or unit name');
    }

    // Validation
    if (!CONVERSION_FACTORS[category]) {
      throw new Error(`Unknown category: ${category}`);
    }

    const factors = CONVERSION_FACTORS[category];
    if (!factors[referenceUnit]) {
      throw new Error(`Unknown reference unit '${referenceUnit}' in category '${category}'`);
    }

    if (factors[unit]) {
      throw new Error(`Unit '${unit}' already exists in category '${category}'`);
    }

    if (factorToReference <= 0) {
      throw new Error('Conversion factor must be positive');
    }

    // Add to custom units
    if (!this.customUnits[category]) {
      this.customUnits[category] = {};
    }

    const baseFactor = factors[referenceUnit] * factorToReference;
    this.customUnits[category][unit] = baseFactor;

    // Add to the main conversion factors
    factors[unit] = baseFactor;

    // Store aliases
    if (aliases.length > 0) {
      this.customAliases[unit] = aliases;
      aliases.forEach(alias => {
        UNIT_ALIASES[alias.toLowerCase()] = unit;
      });
      _REVERSE_ALIASES_CACHE = null; // Invalidate cache
    }

    _SEARCH_CACHE = null; // Invalidate search cache
    this.saveToStorage();
    return { success: true, message: `Custom unit '${unit}' added to ${category}` };
  }

  removeUnit(category, unit) {
    if (!this.customUnits[category] || !this.customUnits[category][unit]) {
      throw new Error(`Custom unit '${unit}' not found in category '${category}'`);
    }

    // Remove from factors
    delete CONVERSION_FACTORS[category][unit];
    delete this.customUnits[category][unit];

    // Remove aliases
    if (this.customAliases[unit]) {
      this.customAliases[unit].forEach(alias => {
        delete UNIT_ALIASES[alias.toLowerCase()];
      });
      delete this.customAliases[unit];
      _REVERSE_ALIASES_CACHE = null; // Invalidate cache
    }

    _SEARCH_CACHE = null; // Invalidate search cache
    this.saveToStorage();
    return { success: true, message: `Custom unit '${unit}' removed` };
  }

  getCustomUnits(category = null) {
    if (category) {
      return this.customUnits[category] || {};
    }
    return this.customUnits;
  }

  isCustomUnit(category, unit) {
    return (
      this.customUnits[category] &&
      Object.prototype.hasOwnProperty.call(this.customUnits[category], unit)
    );
  }

  saveToStorage() {
    try {
      localStorage.setItem('customUnits', JSON.stringify(this.customUnits));
      localStorage.setItem('customAliases', JSON.stringify(this.customAliases));
    } catch {
      // Silent fail for localStorage errors
    }
  }

  loadFromStorage() {
    try {
      const saved = localStorage.getItem('customUnits');
      const savedAliases = localStorage.getItem('customAliases');

      if (saved) {
        const parsed = JSON.parse(saved);
        this.customUnits = {};

        // Restore to main conversion factors with validation
        Object.keys(parsed).forEach(category => {
          if (!isValidKey(category)) { return; }

          if (!this.customUnits[category]) {
            this.customUnits[category] = {};
          }

          Object.keys(parsed[category]).forEach(unit => {
            if (!isValidKey(unit)) { return; }

            const value = parsed[category][unit];
            this.customUnits[category][unit] = value;

            // Only add to CONVERSION_FACTORS if category exists
            if (CONVERSION_FACTORS[category]) {
              CONVERSION_FACTORS[category][unit] = value;
            }
          });
        });
      }

      if (savedAliases) {
        const parsedAliases = JSON.parse(savedAliases);
        this.customAliases = {};

        // Restore aliases with validation
        Object.keys(parsedAliases).forEach(unit => {
          if (!isValidKey(unit)) { return; }

          const aliases = parsedAliases[unit];
          if (Array.isArray(aliases)) {
            this.customAliases[unit] = aliases;
            aliases.forEach(alias => {
              if (isValidKey(alias)) {
                UNIT_ALIASES[alias.toLowerCase()] = unit;
              }
            });
          }
        });
        _REVERSE_ALIASES_CACHE = null; // Invalidate cache
        _SEARCH_CACHE = null; // Invalidate search cache
      }
    } catch {
      // Silent fail for localStorage errors
    }
  }

  clearAll() {
    // Remove all custom units
    Object.keys(this.customUnits).forEach(category => {
      Object.keys(this.customUnits[category]).forEach(unit => {
        delete CONVERSION_FACTORS[category][unit];
      });
    });

    // Remove all custom aliases
    Object.keys(this.customAliases).forEach(unit => {
      this.customAliases[unit].forEach(alias => {
        delete UNIT_ALIASES[alias.toLowerCase()];
      });
    });

    _REVERSE_ALIASES_CACHE = null; // Invalidate cache
    _SEARCH_CACHE = null; // Invalidate search cache
    this.customUnits = {};
    this.customAliases = {};
    localStorage.removeItem('customUnits');
    localStorage.removeItem('customAliases');
  }
}

// Global instance
const customUnitManager = new CustomUnitManager();

// ============================================================================
// CONVERSION FUNCTIONS
// ============================================================================

// Temperature conversion
function convertTemperature(value, fromUnit, toUnit) {
  fromUnit = fromUnit.toUpperCase();
  toUnit = toUnit.toUpperCase();

  if (fromUnit === toUnit) {
    return value;
  }

  // Convert to Kelvin first
  let kelvin;
  switch (fromUnit) {
    case 'K':
      kelvin = value;
      break;
    case 'C':
      kelvin = value + CELSIUS_OFFSET;
      break;
    case 'F':
      kelvin = (value - 32) * RANKINE_RATIO + CELSIUS_OFFSET;
      break;
    case 'R':
      kelvin = value * RANKINE_RATIO;
      break;
    default:
      throw new Error(`Unknown temperature unit: ${fromUnit}`);
  }

  // Convert from Kelvin to target
  switch (toUnit) {
    case 'K':
      return kelvin;
    case 'C':
      return kelvin - CELSIUS_OFFSET;
    case 'F':
      return (kelvin - CELSIUS_OFFSET) / RANKINE_RATIO + 32;
    case 'R':
      return kelvin / RANKINE_RATIO;
    default:
      throw new Error(`Unknown temperature unit: ${toUnit}`);
  }
}

// Gas flow conversion helpers
function standardToActualFlow(scfmValue, tempK, pressurePa, standard) {
  const { temp: stdTemp, pressure: stdPressure } = standard;
  return scfmValue * (stdPressure / pressurePa) * (tempK / stdTemp);
}

function actualToStandardFlow(acfmValue, tempK, pressurePa, standard) {
  const { temp: stdTemp, pressure: stdPressure } = standard;
  return acfmValue * (pressurePa / stdPressure) * (stdTemp / tempK);
}

function scfmToStandardM3PerHour(scfmValue, standard, referenceStd) {
  let m3HrStd = scfmValue * SCFM_TO_CU_METER_PER_HOUR_AT_60F;
  const { temp: stdTemp, pressure: stdPressure } = standard;
  const { temp: refTemp, pressure: refPressure } = referenceStd;

  if (stdTemp !== refTemp || stdPressure !== refPressure) {
    m3HrStd = m3HrStd * (refTemp / stdTemp) * (stdPressure / refPressure);
  }
  return m3HrStd;
}

function standardM3PerHourToScfm(m3HrAtRef, referenceStd, standard) {
  const { temp: refTemp, pressure: refPressure } = referenceStd;
  const { temp: stdTemp, pressure: stdPressure } = standard;

  let m3HrAtScfmStd;
  if (refTemp !== stdTemp || refPressure !== stdPressure) {
    m3HrAtScfmStd = m3HrAtRef * (stdTemp / refTemp) * (refPressure / stdPressure);
  } else {
    m3HrAtScfmStd = m3HrAtRef;
  }

  return m3HrAtScfmStd / SCFM_TO_CU_METER_PER_HOUR_AT_60F;
}

// Gas flow conversion with T/P corrections
function convertGasFlow(value, fromUnit, toUnit, options = {}) {
  const {
    temperature = null,
    pressure = null,
    gasType = 'air',
    standardCondition = 'SCFM_60F'
  } = options;

  const gasProps = GAS_DATABASE[gasType.toLowerCase()] || GAS_DATABASE['air'];
  const standard = StandardConditions[standardCondition] || StandardConditions.SCFM_60F;

  fromUnit = fromUnit.toUpperCase();
  toUnit = toUnit.toUpperCase();

  // Validate ACFM requires T/P
  if ((fromUnit === 'ACFM' || toUnit === 'ACFM') && (temperature === null || pressure === null)) {
    throw new Error('Temperature and pressure are required for ACFM conversions');
  }

  // Convert to standard m³/hr (intermediate unit)
  let m3HrStd;

  if (fromUnit === 'SCFM') {
    m3HrStd = scfmToStandardM3PerHour(value, standard, StandardConditions.STP);
  } else if (fromUnit === 'ACFM') {
    const scfm = actualToStandardFlow(value, temperature, pressure, standard);
    m3HrStd = scfmToStandardM3PerHour(scfm, standard, StandardConditions.STP);
  } else if (fromUnit === 'NM3/HR' || fromUnit === 'NM³/HR') {
    m3HrStd = value;
  } else if (CONVERSION_FACTORS.mass_flow[fromUnit.toLowerCase()]) {
    // Convert mass flow to volumetric
    const kgS = value * CONVERSION_FACTORS.mass_flow[fromUnit.toLowerCase()];
    const kgHr = kgS * 3600.0;
    m3HrStd = kgHr / gasProps.density_stp;
  } else {
    throw new Error(`Unknown gas flow unit: ${fromUnit}`);
  }

  // Convert from standard m³/hr to target
  if (toUnit === 'SCFM') {
    return standardM3PerHourToScfm(m3HrStd, StandardConditions.STP, standard);
  } else if (toUnit === 'ACFM') {
    const scfm = standardM3PerHourToScfm(m3HrStd, StandardConditions.STP, standard);
    return standardToActualFlow(scfm, temperature, pressure, standard);
  } else if (toUnit === 'NM3/HR' || toUnit === 'NM³/HR') {
    return m3HrStd;
  } else if (CONVERSION_FACTORS.mass_flow[toUnit.toLowerCase()]) {
    // Convert volumetric to mass flow
    const kgHr = m3HrStd * gasProps.density_stp;
    const kgS = kgHr / 3600.0;
    return kgS / CONVERSION_FACTORS.mass_flow[toUnit.toLowerCase()];
  } else {
    throw new Error(`Unknown gas flow unit: ${toUnit}`);
  }
}

// Heating value conversion
function convertHeatingValue(value, fromUnit, toUnit, gasDensityStp = null) {
  const fromKey = fromUnit.toLowerCase();
  const toKey = toUnit.toLowerCase();

  if (fromKey === toKey) {
    return value;
  }

  // Convert to MJ/kg (intermediate unit)
  let mjPerKg;

  if (HEATING_VALUE_FACTORS[fromUnit] !== undefined) {
    if (HEATING_VALUE_FACTORS[fromUnit] === null) {
      // Volumetric unit - needs density
      if (gasDensityStp === null) {
        throw new Error(`Gas density required for ${fromUnit} conversion`);
      }

      if (fromKey === 'mj/nm3' || fromKey === 'mj/nm³') {
        mjPerKg = value / gasDensityStp;
      } else if (fromKey === 'btu/scf') {
        const mjNm3 = value * 0.0372589;
        mjPerKg = mjNm3 / gasDensityStp;
      } else if (fromKey === 'kwh/nm3' || fromKey === 'kwh/nm³') {
        const mjNm3 = value * 3.6;
        mjPerKg = mjNm3 / gasDensityStp;
      } else {
        throw new Error(`Conversion from ${fromUnit} not implemented`);
      }
    } else {
      // Mass-based unit
      mjPerKg = value * HEATING_VALUE_FACTORS[fromUnit];
    }
  } else {
    throw new Error(`Unknown heating value unit: ${fromUnit}`);
  }

  // Convert from MJ/kg to target
  if (HEATING_VALUE_FACTORS[toUnit] !== undefined) {
    if (HEATING_VALUE_FACTORS[toUnit] === null) {
      // Volumetric unit - needs density
      if (gasDensityStp === null) {
        throw new Error(`Gas density required for ${toUnit} conversion`);
      }

      if (toKey === 'mj/nm3' || toKey === 'mj/nm³') {
        return mjPerKg * gasDensityStp;
      } else if (toKey === 'btu/scf') {
        const mjNm3 = mjPerKg * gasDensityStp;
        return mjNm3 / 0.0372589;
      } else if (toKey === 'kwh/nm3' || toKey === 'kwh/nm³') {
        const mjNm3 = mjPerKg * gasDensityStp;
        return mjNm3 / 3.6;
      } else {
        throw new Error(`Conversion to ${toUnit} not implemented`);
      }
    } else {
      // Mass-based unit
      return mjPerKg / HEATING_VALUE_FACTORS[toUnit];
    }
  } else {
    throw new Error(`Unknown heating value unit: ${toUnit}`);
  }
}

// Get category for a unit
function getCategory(unit) {
  // Normalize unit
  unit = UNIT_ALIASES[unit.toLowerCase()] || unit;

  // Check temperature first
  if (['K', 'C', 'F', 'R'].includes(unit.toUpperCase())) {
    return 'temperature';
  }

  // Check gas flow
  const gasFlowUnits = ['SCFM', 'ACFM', 'NM3/HR', 'NM³/HR'];
  if (gasFlowUnits.includes(unit.toUpperCase())) {
    return 'gas_flow';
  }

  // Check heating value
  if (HEATING_VALUE_FACTORS[unit] !== undefined) {
    return 'heating_value';
  }

  // Search in conversion factors
  for (const [category, units] of Object.entries(CONVERSION_FACTORS)) {
    if (unit in units) {
      return category;
    }
  }

  return null;
}

// Main conversion function
function convert(value, fromUnit, toUnit, options = {}) {
  // Normalize units
  fromUnit = UNIT_ALIASES[fromUnit.toLowerCase()] || fromUnit;
  toUnit = UNIT_ALIASES[toUnit.toLowerCase()] || toUnit;

  const fromCategory = getCategory(fromUnit);
  const toCategory = getCategory(toUnit);

  if (!fromCategory) {
    throw new Error(`Unknown unit: ${fromUnit}`);
  }

  if (!toCategory) {
    throw new Error(`Unknown unit: ${toUnit}`);
  }

  if (fromCategory !== toCategory) {
    // Check if it's a supported cross-category conversion (mass flow <-> gas flow)
    const isGasFlowConversion =
      (fromCategory === 'gas_flow' && toCategory === 'mass_flow') ||
      (fromCategory === 'mass_flow' && toCategory === 'gas_flow');

    if (!isGasFlowConversion) {
      throw new Error(`Cannot convert ${fromUnit} (${fromCategory}) to ${toUnit} (${toCategory})`);
    }
  }

  // Handle special categories
  if (fromCategory === 'temperature') {
    return convertTemperature(value, fromUnit, toUnit);
  }

  if (fromCategory === 'gas_flow' || toCategory === 'gas_flow') {
    return convertGasFlow(value, fromUnit, toUnit, options);
  }

  if (fromCategory === 'heating_value') {
    return convertHeatingValue(value, fromUnit, toUnit, options.gasDensityStp);
  }

  // Standard linear conversion
  const factors = CONVERSION_FACTORS[fromCategory];
  const fromFactor = factors[fromUnit];
  const toFactor = factors[toUnit];

  if (fromFactor === undefined || toFactor === undefined) {
    throw new Error(`Conversion factors not found for ${fromUnit} to ${toUnit}`);
  }

  // Convert to base unit then to target unit
  const baseValue = value * fromFactor;
  return baseValue / toFactor;
}

// Get all units for a category
function getUnitsForCategory(category) {
  if (category === 'temperature') {
    return ['K', 'C', 'F', 'R'];
  }

  if (category === 'gas_flow') {
    return ['SCFM', 'ACFM', 'Nm³/hr'];
  }

  if (category === 'heating_value') {
    return Object.keys(HEATING_VALUE_FACTORS);
  }

  if (category in CONVERSION_FACTORS) {
    return Object.keys(CONVERSION_FACTORS[category]);
  }

  return [];
}

// Get all categories
function getCategories() {
  return ['temperature', 'gas_flow', 'heating_value', ...Object.keys(CONVERSION_FACTORS)];
}

// Helper to build the search cache
function _buildSearchCache() {
  const cache = {
    flat: [],
    byCategory: {}
  };
  const reverseAliases = getReverseAliases();
  const categories = getCategories();

  categories.forEach(cat => {
    cache.byCategory[cat] = [];
    const units = getUnitsForCategory(cat);
    units.forEach(unit => {
      const lowerUnit = unit.toLowerCase();
      const aliases = reverseAliases[unit] || [];
      // Optimization: Aliases are already lowercase from UNIT_ALIASES keys, so no need to map/lowerCase them

      const item = {
        unit,
        category: cat,
        lowerUnit,
        aliases
      };

      cache.flat.push(item);
      cache.byCategory[cat].push(item);
    });
  });

  return cache;
}

// Search units by query
function searchUnits(query, category = null) {
  query = query.toLowerCase().trim();
  if (!query) {
    return [];
  }

  // Build cache if needed
  if (!_SEARCH_CACHE) {
    _SEARCH_CACHE = _buildSearchCache();
  }

  const results = [];

  // Optimization: Search only within category if specified
  const candidates =
    category && _SEARCH_CACHE.byCategory[category]
      ? _SEARCH_CACHE.byCategory[category]
      : _SEARCH_CACHE.flat;

  // Use cache for searching
  for (const item of candidates) {
    // Filter by category if specified
    if (category && item.category !== category) {
      continue;
    }

    // Check if unit matches
    if (item.lowerUnit.includes(query)) {
      results.push({ unit: item.unit, category: item.category, score: 100 });
    } else {
      // Check aliases
      if (item.aliases) {
        for (const alias of item.aliases) {
          if (alias.includes(query)) {
            const score = alias === query ? 75 : 50;
            results.push({
              unit: item.unit,
              category: item.category,
              score: score,
              matchedAlias: alias
            });
            break;
          }
        }
      }
    }
  }

  // Sort by score (higher first), then alphabetically
  return results.sort((a, b) => {
    if (b.score !== a.score) {
      return b.score - a.score;
    }
    return a.unit.localeCompare(b.unit);
  });
}

/**
 * Debounce a function to limit how often it runs
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
  let timeout;
  return function (...args) {
    const context = this;
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(context, args), wait);
  };
}

// Export functions
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    convert,
    getCategory,
    getCategories,
    getUnitsForCategory,
    searchUnits,
    customUnitManager,
    GAS_DATABASE,
    StandardConditions,
    CONVERSION_FACTORS,
    HEATING_VALUE_FACTORS,
    debounce
  };
}
