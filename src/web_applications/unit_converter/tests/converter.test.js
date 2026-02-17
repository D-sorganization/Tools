// Mock localStorage
const localStorageMock = (function () {
  let store = {};
  return {
    getItem: function (key) {
      return store[key] || null;
    },
    setItem: function (key, value) {
      store[key] = value.toString();
    },
    removeItem: function (key) {
      delete store[key];
    },
    clear: function () {
      store = {};
    }
  };
})();

Object.defineProperty(global, 'localStorage', { value: localStorageMock });

const converter = require('../unit-converter-app/converter.js');

describe('Unit Converter', () => {
  describe('Basic Conversions', () => {
    test('converts length', () => {
      expect(converter.convert(1, 'm', 'cm')).toBeCloseTo(100);
      expect(converter.convert(1, 'km', 'm')).toBeCloseTo(1000);
      expect(converter.convert(1, 'in', 'cm')).toBeCloseTo(2.54);
      expect(converter.convert(1, 'mi', 'km')).toBeCloseTo(1.609344);
    });

    test('converts mass', () => {
      expect(converter.convert(1, 'kg', 'g')).toBeCloseTo(1000);
      expect(converter.convert(1, 'lb', 'kg')).toBeCloseTo(0.45359237);
      expect(converter.convert(1, 'oz', 'g')).toBeCloseTo(28.349523125);
    });

    test('converts volume', () => {
      expect(converter.convert(1, 'L', 'mL')).toBeCloseTo(1000);
      expect(converter.convert(1, 'gal', 'L')).toBeCloseTo(3.785411784);
    });

    test('converts pressure', () => {
      expect(converter.convert(1, 'atm', 'Pa')).toBeCloseTo(101325);
      expect(converter.convert(1, 'bar', 'kPa')).toBeCloseTo(100);
      expect(converter.convert(14.696, 'psi', 'atm')).toBeCloseTo(1, 3);
    });
  });

  describe('Temperature Conversions', () => {
    test('converts Celsius to Kelvin', () => {
      expect(converter.convert(0, 'C', 'K')).toBeCloseTo(273.15);
      expect(converter.convert(100, 'C', 'K')).toBeCloseTo(373.15);
    });

    test('converts Fahrenheit to Celsius', () => {
      expect(converter.convert(32, 'F', 'C')).toBeCloseTo(0);
      expect(converter.convert(212, 'F', 'C')).toBeCloseTo(100);
      expect(converter.convert(-40, 'F', 'C')).toBeCloseTo(-40);
    });

    test('converts Rankine to Kelvin', () => {
      expect(converter.convert(0, 'R', 'K')).toBeCloseTo(0);
      expect(converter.convert(491.67, 'R', 'K')).toBeCloseTo(273.15);
    });
  });

  describe('getCategory', () => {
    test('identifies length units', () => {
      expect(converter.getCategory('m')).toBe('length');
      expect(converter.getCategory('meter')).toBe('length');
    });

    test('identifies gas flow units (case insensitive)', () => {
      expect(converter.getCategory('Nm3/hr')).toBe('gas_flow');
      expect(converter.getCategory('Nm³/hr')).toBe('gas_flow');
      expect(converter.getCategory('SCFM')).toBe('gas_flow');
      expect(converter.getCategory('acfm')).toBe('gas_flow');
    });

    test('identifies heating value units', () => {
      expect(converter.getCategory('MJ/kg')).toBe('heating_value');
      expect(converter.getCategory('BTU/scf')).toBe('heating_value');
    });

    test('returns null for unknown unit', () => {
      expect(converter.getCategory('invalid_unit')).toBeNull();
    });
  });

  describe('Gas Flow Conversions', () => {
    test('converts SCFM to Nm3/hr', () => {
      // 1 SCFM approx 1.699 Nm3/hr at standard conditions mismatch
      const result = converter.convert(1, 'SCFM', 'Nm3/hr');
      expect(result).toBeGreaterThan(1.5);
      expect(result).toBeLessThan(1.8);
    });

    test('converts mass flow to gas flow (Cross Category)', () => {
      // 1 kg/s Air to SCFM
      const result = converter.convert(1, 'kg/s', 'SCFM');
      expect(result).toBeGreaterThan(1700);
      expect(result).toBeLessThan(1800);
    });

    test('converts gas flow to mass flow (Cross Category)', () => {
      const result = converter.convert(1000, 'SCFM', 'kg/hr');
      expect(result).toBeGreaterThan(0);
    });

    test('ACFM conversion requires temperature and pressure', () => {
      expect(() => {
        converter.convert(100, 'ACFM', 'SCFM');
      }).toThrow('Temperature and pressure are required');
    });

    test('ACFM to SCFM with params', () => {
      // At STP, ACFM = SCFM (if standard is STP)
      // Default standard is SCFM_60F (288.706 K, 1 atm)
      // Let's use params equal to SCFM_60F conditions
      const options = {
        temperature: 288.706,
        pressure: 101325,
        standardCondition: 'SCFM_60F'
      };
      const result = converter.convert(100, 'ACFM', 'SCFM', options);
      expect(result).toBeCloseTo(100, 1);
    });
  });

  describe('Heating Value Conversions', () => {
    test('converts mass based heating values', () => {
      // 1 MJ/kg = 1000 kJ/kg
      expect(converter.convert(1, 'MJ/kg', 'kJ/kg')).toBeCloseTo(1000);
    });

    test('converts volumetric heating values requires density', () => {
      expect(() => {
        converter.convert(1, 'MJ/Nm3', 'MJ/kg');
      }).toThrow('Gas density required');
    });

    test('converts volumetric to mass heating value with density', () => {
      // 1 MJ/Nm3 / density (kg/Nm3) = MJ/kg
      // density = 1 kg/Nm3 -> 1 MJ/kg
      const options = { gasDensityStp: 1.0 };
      expect(converter.convert(1, 'MJ/Nm3', 'MJ/kg', options)).toBeCloseTo(1);
    });
  });

  describe('Search Units', () => {
    test('finds units by name', () => {
      const results = converter.searchUnits('meter');
      expect(results.length).toBeGreaterThan(0);
      expect(results[0].unit).toBe('m'); // canonical unit for meter
    });

    test('finds units by alias', () => {
      const results = converter.searchUnits('lbs');
      expect(results.some(r => r.unit === 'lb')).toBe(true);
    });

    test('returns empty array for no match', () => {
      expect(converter.searchUnits('xyz123')).toEqual([]);
    });
  });

  describe('Custom Unit Manager', () => {
    beforeEach(() => {
      converter.customUnitManager.clearAll();
    });

    afterEach(() => {
      converter.customUnitManager.clearAll();
    });

    test('adds a custom unit', () => {
      // 1 my_meter = 2 meters
      const result = converter.customUnitManager.addUnit('length', 'my_meter', 'm', 2.0);
      expect(result.success).toBe(true);

      expect(converter.getCategory('my_meter')).toBe('length');
      expect(converter.convert(1, 'my_meter', 'm')).toBeCloseTo(2.0);
    });

    test('removes a custom unit', () => {
      converter.customUnitManager.addUnit('length', 'my_meter', 'm', 2.0);
      converter.customUnitManager.removeUnit('length', 'my_meter');

      expect(() => {
        converter.convert(1, 'my_meter', 'm');
      }).toThrow('Unknown unit');
    });

    test('cannot add duplicate unit', () => {
      converter.customUnitManager.addUnit('length', 'my_unit', 'm', 2.0);
      expect(() => {
        converter.customUnitManager.addUnit('length', 'my_unit', 'm', 3.0);
      }).toThrow("Unit 'my_unit' already exists");
    });
  });

  describe('Error Handling', () => {
    test('throws for incompatible categories', () => {
      expect(() => {
        converter.convert(1, 'm', 'kg');
      }).toThrow('Cannot convert m (length) to kg (mass)');
    });

    test('throws for unknown units', () => {
      expect(() => {
        converter.convert(1, 'blah', 'm');
      }).toThrow('Unknown unit: blah');
    });
  });
  describe('Utility Functions', () => {
    test('getCategories returns all categories', () => {
      const categories = converter.getCategories();
      expect(categories).toContain('length');
      expect(categories).toContain('mass');
      expect(categories).toContain('temperature');
      expect(categories.length).toBeGreaterThan(10);
    });

    test('getUnitsForCategory returns units', () => {
      const units = converter.getUnitsForCategory('length');
      expect(units).toContain('m');
      expect(units).toContain('ft');
    });

    test('getUnitsForCategory returns empty for invalid category', () => {
      expect(converter.getUnitsForCategory('invalid')).toEqual([]);
    });
  });

  describe('Validation & Edge Cases', () => {
    test('handles non-numeric input gracefully (NaN)', () => {
      // Depending on implementation, might return NaN or throw
      // In JS arithmetic, string * number = NaN usually.
      const result = converter.convert('abc', 'm', 'cm');
      expect(result).toBeNaN();
    });

    test('warns or handles missing options for required conversions', () => {
      expect(() => {
        converter.convert(100, 'ACFM', 'SCFM', {}); // Missing temp/press
      }).toThrow('Temperature and pressure are required');
    });
  });

  describe('Additional Categories', () => {
    test('converts dynamic viscosity', () => {
      expect(converter.convert(1, 'Pa·s', 'cP')).toBeCloseTo(1000);
    });

    test('converts kinematic viscosity', () => {
      expect(converter.convert(1, 'St', 'm2/s')).toBeCloseTo(0.0001);
    });

    test('converts density', () => {
      expect(converter.convert(1000, 'kg/m3', 'g/cm3')).toBeCloseTo(1);
    });
  });

  describe('Heating Value Extended', () => {
    test('converts kWh/Nm3 to MJ/kg with density', () => {
      // 1 kWh = 3.6 MJ
      // 1 kWh/Nm3 = 3.6 MJ/Nm3
      // MJ/kg = (MJ/Nm3) / density
      // If density = 1.0, result should be 3.6
      const options = { gasDensityStp: 1.0 };
      expect(converter.convert(1, 'kWh/Nm3', 'MJ/kg', options)).toBeCloseTo(3.6);
    });
  });

  describe('Debounce Utility', () => {
    test('debounces a function', done => {
      let counter = 0;
      const increment = () => {
        counter++;
      };

      const debouncedIncrement = converter.debounce(increment, 50);

      // Call multiple times
      debouncedIncrement();
      debouncedIncrement();
      debouncedIncrement();

      expect(counter).toBe(0);

      // Wait for debounce
      setTimeout(() => {
        expect(counter).toBe(1);
        done();
      }, 100);
    });

    test('executes immediately if wait is 0 (or close to)', done => {
      // This test depends on how debounce is implemented.
      // Standard debounce with only wait param usually doesn't execute immediately unless trailing.
      // It always waits 'wait' ms.
      // If wait is small, it just runs quickly.
      let counter = 0;
      const increment = () => counter++;
      const debounced = converter.debounce(increment, 10);

      debounced();
      expect(counter).toBe(0);

      setTimeout(() => {
        expect(counter).toBe(1);
        done();
      }, 20);
    });
  });
});
