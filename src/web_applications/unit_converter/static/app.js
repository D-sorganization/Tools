/**
 * Unit Converter - Frontend (GUI only)
 * All calculations are performed by the Python backend via /api/convert.
 * This file handles UI state, event binding, and API calls.
 */

// DOM elements
const categorySelect = document.getElementById('category');
const fromValueInput = document.getElementById('fromValue');
const fromUnitSelect = document.getElementById('fromUnit');
const toUnitSelect = document.getElementById('toUnit');
const convertBtn = document.getElementById('convertBtn');
const swapBtn = document.getElementById('swapBtn');
const resultDisplay = document.getElementById('resultDisplay');
const resultUnit = document.getElementById('resultUnit');
const errorMessage = document.getElementById('errorMessage');
const recentList = document.getElementById('recentList');
const clearHistoryBtn = document.getElementById('clearHistory');

// Conditional param sections
const gasFlowParams = document.getElementById('gasFlowParams');
const heatingValueParams = document.getElementById('heatingValueParams');
const standardConditionSelect = document.getElementById('standardCondition');
const gasTypeSelect = document.getElementById('gasType');
const temperatureInput = document.getElementById('temperature');
const pressureInput = document.getElementById('pressure');
const gasDensityInput = document.getElementById('gasDensity');

// Theme selector
const themeSelect = document.getElementById('themeSelect');

// State
let categoryData = {};
let conversionHistory = [];
let debounceTimer = null;

// ============================================================================
// INITIALIZATION
// ============================================================================

async function init() {
  loadTheme();
  loadHistory();
  await fetchCategories();
  setupEventListeners();
  populateUnits(categorySelect.value);
  updateConditionalParams();
  performConversion();
}

// ============================================================================
// THEME (inherits from shared/theme-definitions/themes.json)
// ============================================================================

function loadTheme() {
  var saved = localStorage.getItem('unitConverterTheme');
  if (saved && themeSelect) {
    document.documentElement.setAttribute('data-theme', saved);
    themeSelect.value = saved;
  }
}

function changeTheme(themeName) {
  document.documentElement.setAttribute('data-theme', themeName);
  localStorage.setItem('unitConverterTheme', themeName);
}

async function fetchCategories() {
  try {
    const response = await fetch('/api/categories');
    if (response.ok) {
      categoryData = await response.json();
    }
  } catch (err) {
    // Categories are already in the HTML via Jinja, so this is a fallback
    console.error('Failed to fetch categories:', err);
  }
}

// ============================================================================
// UNIT POPULATION
// ============================================================================

function populateUnits(category) {
  const data = categoryData[category];
  if (!data) return;

  const units = data.units;

  fromUnitSelect.innerHTML = '';
  toUnitSelect.innerHTML = '';

  units.forEach(function(unit) {
    fromUnitSelect.add(new Option(unit, unit));
    toUnitSelect.add(new Option(unit, unit));
  });

  if (units.length > 1) {
    fromUnitSelect.selectedIndex = 0;
    toUnitSelect.selectedIndex = 1;
  }
}

function updateConditionalParams() {
  const category = categorySelect.value;

  if (category === 'gas_flow') {
    gasFlowParams.classList.remove('hidden');
  } else {
    gasFlowParams.classList.add('hidden');
  }

  if (category === 'heating_value') {
    heatingValueParams.classList.remove('hidden');
  } else {
    heatingValueParams.classList.add('hidden');
  }
}

// ============================================================================
// CONVERSION (calls Python backend)
// ============================================================================

async function performConversion() {
  hideError();

  const value = parseFloat(fromValueInput.value);
  if (isNaN(value)) {
    resultDisplay.textContent = '--';
    resultUnit.textContent = '';
    return;
  }

  const fromUnit = fromUnitSelect.value;
  const toUnit = toUnitSelect.value;

  if (!fromUnit || !toUnit) return;

  const payload = {
    value: value,
    from_unit: fromUnit,
    to_unit: toUnit,
  };

  // Add optional parameters
  const category = categorySelect.value;

  if (category === 'gas_flow') {
    const temp = parseFloat(temperatureInput.value);
    const press = parseFloat(pressureInput.value);
    if (!isNaN(temp)) payload.temperature = temp;
    if (!isNaN(press)) payload.pressure = press;
    payload.gas_type = gasTypeSelect.value;
    payload.standard_condition = standardConditionSelect.value || 'SCFM_60F';
  }

  if (category === 'heating_value') {
    const density = parseFloat(gasDensityInput.value);
    if (!isNaN(density)) payload.gas_density_stp = density;
  }

  try {
    const response = await fetch('/api/convert', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      showError(data.error || 'Conversion failed');
      return;
    }

    resultDisplay.textContent = data.formatted;
    resultUnit.textContent = toUnit;

    // Flash animation
    resultDisplay.classList.remove('result-flash');
    // Force reflow to restart animation
    void resultDisplay.offsetWidth;
    resultDisplay.classList.add('result-flash');

    addToHistory(value, fromUnit, data.result, toUnit, category);
  } catch (err) {
    showError('Network error: could not reach backend');
  }
}

function swapUnits() {
  const fromUnit = fromUnitSelect.value;
  const toUnit = toUnitSelect.value;

  fromUnitSelect.value = toUnit;
  toUnitSelect.value = fromUnit;

  performConversion();
}

// ============================================================================
// HISTORY
// ============================================================================

function loadHistory() {
  try {
    const saved = localStorage.getItem('unitConverterHistory');
    conversionHistory = saved ? JSON.parse(saved) : [];
    renderHistory();
  } catch (e) {
    conversionHistory = [];
  }
}

function saveHistory() {
  try {
    localStorage.setItem(
      'unitConverterHistory',
      JSON.stringify(conversionHistory.slice(0, 20))
    );
  } catch (e) {
    // silent
  }
}

function addToHistory(fromValue, fromUnit, toValue, toUnit, category) {
  const item = {
    fromValue: fromValue,
    fromUnit: fromUnit,
    toValue: toValue,
    toUnit: toUnit,
    category: category,
    timestamp: Date.now(),
  };

  // Remove duplicates
  conversionHistory = conversionHistory.filter(function(h) {
    return !(
      h.fromValue === fromValue &&
      h.fromUnit === fromUnit &&
      h.toUnit === toUnit
    );
  });

  conversionHistory.unshift(item);
  saveHistory();
  renderHistory();
}

function clearHistory() {
  conversionHistory = [];
  localStorage.removeItem('unitConverterHistory');
  renderHistory();
}

function renderHistory() {
  if (conversionHistory.length === 0) {
    recentList.innerHTML = '<p class="empty-state">No recent conversions</p>';
    return;
  }

  recentList.innerHTML = '';
  conversionHistory.slice(0, 10).forEach(function(item, index) {
    const timeAgo = formatTimeAgo(item.timestamp);
    const btn = document.createElement('button');
    btn.className = 'recent-item';
    btn.type = 'button';
    btn.dataset.index = index.toString();

    const fromVal = formatNumber(item.fromValue);
    const toVal = formatNumber(item.toValue);

    const textDiv = document.createElement('div');
    textDiv.className = 'recent-item-text';
    textDiv.textContent = fromVal + ' ' + item.fromUnit + ' = ' + toVal + ' ' + item.toUnit;
    btn.appendChild(textDiv);

    const timeDiv = document.createElement('div');
    timeDiv.className = 'recent-item-time';
    timeDiv.textContent = item.category + ' \u2022 ' + timeAgo;
    btn.appendChild(timeDiv);

    btn.addEventListener('click', function() {
      loadFromHistory(item);
    });

    recentList.appendChild(btn);
  });
}

function loadFromHistory(item) {
  categorySelect.value = item.category;
  populateUnits(item.category);
  updateConditionalParams();

  setTimeout(function() {
    fromValueInput.value = item.fromValue;
    fromUnitSelect.value = item.fromUnit;
    toUnitSelect.value = item.toUnit;
    performConversion();
  }, 10);
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.classList.remove('hidden');
}

function hideError() {
  errorMessage.classList.add('hidden');
}

// ============================================================================
// UTILITIES
// ============================================================================

function formatNumber(num) {
  if (typeof num !== 'number' || isNaN(num)) return '';
  if (Math.abs(num) >= 1e10 || (Math.abs(num) < 1e-6 && num !== 0)) {
    return num.toExponential(6);
  }
  return parseFloat(num.toPrecision(10)).toString();
}

function formatTimeAgo(timestamp) {
  var seconds = Math.floor((Date.now() - timestamp) / 1000);
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return Math.floor(seconds / 60) + 'm ago';
  if (seconds < 86400) return Math.floor(seconds / 3600) + 'h ago';
  if (seconds < 604800) return Math.floor(seconds / 86400) + 'd ago';
  return new Date(timestamp).toLocaleDateString();
}

function debounce(fn, delay) {
  return function() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(fn, delay);
  };
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
  categorySelect.addEventListener('change', function() {
    populateUnits(categorySelect.value);
    updateConditionalParams();
    performConversion();
  });

  fromValueInput.addEventListener('input', debounce(performConversion, 300));
  fromUnitSelect.addEventListener('change', performConversion);
  toUnitSelect.addEventListener('change', performConversion);

  convertBtn.addEventListener('click', performConversion);
  swapBtn.addEventListener('click', swapUnits);
  clearHistoryBtn.addEventListener('click', clearHistory);

  // Gas flow / heating value param changes trigger re-conversion
  standardConditionSelect.addEventListener('change', performConversion);
  gasTypeSelect.addEventListener('change', performConversion);
  temperatureInput.addEventListener('input', debounce(performConversion, 300));
  pressureInput.addEventListener('input', debounce(performConversion, 300));
  gasDensityInput.addEventListener('input', debounce(performConversion, 300));

  // Theme switching
  if (themeSelect) {
    themeSelect.addEventListener('change', function() {
      changeTheme(themeSelect.value);
    });
  }

  // Keyboard shortcut: Enter to convert
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && document.activeElement !== convertBtn) {
      performConversion();
    }
  });
}

// ============================================================================
// START
// ============================================================================

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
