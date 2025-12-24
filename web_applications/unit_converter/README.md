# Unit Converter

A professional, NIST-compliant unit converter Progressive Web App (PWA) designed for engineers and scientists. Works completely offline and can be installed on iOS devices without the App Store.

## Features

- **16+ Unit Categories**: Length, Mass, Volume, Temperature, Pressure, Energy, Power, Flow, and more
- **100+ Units**: All with NIST-standard conversion factors
- **Works Offline**: Full functionality without internet connection
- **iOS Optimized**: Native-feeling interface with iOS design patterns
- **Dark Mode**: Automatic theme switching based on system preferences
- **Custom Units**: Add your own custom units with conversion factors
- **Conversion History**: Track your recent conversions
- **Bidirectional**: Convert from either direction with swap functionality
- **No Backend Required**: 100% client-side JavaScript
- **Installable**: Add to home screen as a standalone app

## Installation

### iOS Installation

1. Open Safari on your iPhone/iPad
2. Navigate to the app URL (e.g., deployed on GitHub Pages)
3. Tap the Share button (square with arrow pointing up)
4. Scroll down and tap "Add to Home Screen"
5. Tap "Add" in the top right corner
6. The app icon will appear on your home screen

### Web Browser

Simply open `unit-converter-app/index.html` in any modern web browser, or access the deployed version.

## Usage

1. **Select Category**: Choose the type of unit you want to convert (e.g., Length, Temperature)
2. **Enter Value**: Type the value in the "From" field
3. **Select Units**: Choose your source and target units from the dropdowns
4. **See Results**: The conversion appears instantly in the "To" field
5. **Swap Units**: Tap the swap button to reverse the conversion direction

### Advanced Features

#### Gas Flow Conversions
For gas flow conversions (SCFM/ACFM), specify:
- Standard condition (60°F, 70°F, STP, NTP, or SATP)
- Gas type (Air, Nitrogen, Oxygen, etc.)
- Operating temperature and pressure

#### Heating Value Conversions
For heating value volumetric conversions, specify the gas density at STP.

#### Custom Units
Add your own units by:
1. Clicking the settings icon
2. Selecting a category
3. Defining the unit symbol and conversion factor relative to a reference unit
4. Optionally adding aliases for easier searching

## Unit Categories Supported

- **Length**: m, cm, mm, µm, nm, Å, km, in, ft, yd, mi, mil
- **Mass**: kg, g, mg, tonne, lb, oz, ton, long_ton, slug, grain
- **Volume**: m³, L, mL, ft³, gal, qt, pt, fl_oz, bbl
- **Temperature**: K, C, F, R
- **Pressure**: Pa, kPa, MPa, GPa, bar, atm, psi, torr, mmHg, inHg, inH2O, ftH2O
- **Energy**: J, kJ, MJ, GJ, BTU, cal, kcal, Wh, kWh, MWh, therm
- **Power**: W, kW, MW, GW, hp, metric_hp, BTU/hr, cal/s, kcal/hr
- **Gas Flow**: SCFM, ACFM with multiple standard conditions
- **Heating Value**: BTU/lb, MJ/kg, with volumetric conversions
- **Mass Flow**: kg/s, kg/min, kg/hr, lb/hr, lb/min, tonne/hr, ton/hr
- **Volumetric Flow**: m³/s, L/s, ft³/min, gal/min, gpm
- **Area**: m², cm², km², ft², in², acre, hectare
- **Time**: s, min, hr, day
- **Density**: kg/m³, g/L, g/cm³, lb/ft³, lb/gal
- **Dynamic Viscosity**: Pa·s, cP, P, mPa·s
- **Kinematic Viscosity**: m²/s, cSt, St
- **Thermal Conductivity**: W/m·K, BTU/(ft·hr·°F), cal/(cm·s·°C)
- **Heat Transfer Coefficient**: W/m²·K, BTU/(ft²·hr·°F)
- **Specific Heat**: J/kg·K, kJ/kg·K, BTU/lb·°F, cal/g·°C

## Technical Details

### Conversion Accuracy
- **NIST Compliant**: All conversion factors sourced from NIST Special Publication 811 (2008 Edition)
- **Physical Constants**: CODATA 2018 values for gas calculations
- **High Precision**: JavaScript floating-point arithmetic with proper rounding

### Architecture
- **100% Client-Side**: All conversions happen in JavaScript, no server required
- **Offline Support**: Service Worker caching for full offline functionality
- **Local Storage**: Conversion history, custom units, and theme preferences
- **Progressive Enhancement**: Works on all modern browsers with enhanced features on capable devices

### Files

- `unit-converter-app/index.html`: Main app structure and UI
- `unit-converter-app/styles.css`: Mobile-optimized, iOS-friendly styles
- `unit-converter-app/converter.js`: Core conversion logic with NIST-compliant factors
- `unit-converter-app/app.js`: UI logic, event handlers, and state management
- `unit-converter-app/service-worker.js`: Offline functionality and caching
- `unit-converter-app/manifest.json`: PWA configuration for installation

## Local Development

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Unit_Converter
   ```

2. Open the app:
   ```bash
   # Simple local testing
   open unit-converter-app/index.html

   # Or serve with a local server for full PWA testing
   cd unit-converter-app
   python -m http.server 8000
   # Navigate to http://localhost:8000/
   ```

3. Make changes to the files in `unit-converter-app/`

4. For service worker updates, increment the cache version in `service-worker.js`

## Deployment

### GitHub Pages

1. Push to GitHub repository
2. Go to Settings → Pages
3. Select your branch and `/unit-converter-app` folder (or root if you move the files)
4. Access at `https://yourusername.github.io/repository-name/`

### Other Hosting

Simply upload the contents of `unit-converter-app/` to any static web hosting service (Netlify, Vercel, etc.).

## Browser Support

- **iOS Safari**: 11.3+ (full PWA support)
- **Chrome/Edge**: Desktop and mobile (full PWA support)
- **Firefox**: Desktop and mobile
- **Samsung Internet**: Mobile

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- **Conversion Factors**: NIST Special Publication 811 (2008 Edition)
- **Physical Constants**: CODATA 2018
- Originally developed as part of a gasification model project, now standalone

## Version

**v2.0.0** - Enhanced with gas flow conversions, custom units, and improved mobile experience
