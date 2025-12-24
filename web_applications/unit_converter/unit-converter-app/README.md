# Unit Converter PWA

A professional, NIST-compliant unit converter Progressive Web App designed for engineers and scientists. Works completely offline and can be installed on your iOS device without the App Store.

## Features

- **16 Unit Categories**: Length, Mass, Volume, Temperature, Pressure, Energy, Power, and more
- **100+ Units**: All with NIST-standard conversion factors
- **Works Offline**: Full functionality without internet connection
- **iOS Optimized**: Native-feeling interface with iOS design patterns
- **Dark Mode**: Automatic theme switching
- **Conversion History**: Track your recent conversions
- **Bidirectional**: Convert from either direction
- **No App Store Required**: Install directly from Safari

## Installation on iOS

1. Open Safari on your iPhone/iPad
2. Navigate to the app URL (e.g., `https://yourusername.github.io/unit-converter-app/`)
3. Tap the Share button (square with arrow pointing up)
4. Scroll down and tap "Add to Home Screen"
5. Tap "Add" in the top right corner
6. The app icon will appear on your home screen

## Usage

1. **Select Category**: Choose the type of unit you want to convert (e.g., Length, Temperature)
2. **Enter Value**: Type the value in the "From" field
3. **Select Units**: Choose your source and target units from the dropdowns
4. **See Results**: The conversion appears instantly in the "To" field
5. **Swap Units**: Tap the swap button to reverse the conversion direction

### Categories Supported

- **Length**: m, cm, mm, µm, nm, Å, km, in, ft, yd, mi, mil
- **Mass**: kg, g, mg, tonne, lb, oz, ton, long_ton, slug, grain
- **Volume**: m³, L, mL, ft³, gal, qt, pt, fl_oz, bbl
- **Temperature**: K, C, F, R
- **Pressure**: Pa, kPa, MPa, GPa, bar, atm, psi, torr, mmHg, inHg, inH2O, ftH2O
- **Energy**: J, kJ, MJ, GJ, BTU, cal, kcal, Wh, kWh, MWh, therm
- **Power**: W, kW, MW, GW, hp, metric_hp, BTU/hr, cal/s, kcal/hr
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

## Keyboard Shortcuts

- `Cmd/Ctrl + K`: Focus and select the "From" value field
- `Cmd/Ctrl + Shift + S`: Swap units

## Technical Details

- **100% Client-Side**: All conversions happen in JavaScript
- **No Backend Required**: Static files only
- **Offline Support**: Service Worker caching
- **Local Storage**: Conversion history and theme preference
- **NIST Compliant**: All conversion factors sourced from NIST Special Publication 811

## Development

This PWA was ported from the Python-based gasification model unit converter, maintaining identical NIST-compliant conversion factors.

### Files

- `index.html`: Main app structure
- `styles.css`: Mobile-optimized, iOS-friendly styles
- `converter.js`: Core conversion logic (ported from Python)
- `app.js`: UI logic and event handlers
- `service-worker.js`: Offline functionality
- `manifest.json`: PWA configuration

### Local Testing

1. Open `index.html` in a browser
2. For full PWA testing, serve with a local server:
   ```bash
   python -m http.server 8000
   ```
3. Navigate to `http://localhost:8000/unit-converter-app/`

## Deployment

### GitHub Pages

1. Push this folder to your GitHub repository
2. Go to Settings → Pages
3. Select branch and `/unit-converter-app` folder
4. Access at `https://yourusername.github.io/unit-converter-app/`

### Update Service Worker Version

When making changes, update the cache version in `service-worker.js`:

```javascript
const CACHE_NAME = 'unit-converter-v1.0.1'; // Increment version
```

## Browser Support

- iOS Safari 11.3+ (PWA support)
- Chrome/Edge (Desktop & Mobile)
- Firefox
- Samsung Internet

## License

Based on the gasification model unit converter project.

## Credits

- Conversion factors: NIST Special Publication 811 (2008 Edition)
- Physical constants: CODATA 2018
- Ported from Python gasification model unit converter
