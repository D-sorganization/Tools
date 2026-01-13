/* global debounce */
/**
 * Unit Converter - Enhanced UI Logic
 * Version 2.0 with search, custom units, gas flow, and heating values
 */

// DOM Elements
const categorySelect = document.getElementById('category');
const fromValueInput = document.getElementById('fromValue');
const toValueInput = document.getElementById('toValue');
const fromUnitSelect = document.getElementById('fromUnit');
const toUnitSelect = document.getElementById('toUnit');
const fromUnitSearch = document.getElementById('fromUnitSearch');
const toUnitSearch = document.getElementById('toUnitSearch');
const fromUnitDropdown = document.getElementById('fromUnitDropdown');
const toUnitDropdown = document.getElementById('toUnitDropdown');
const swapButton = document.getElementById('swapButton');
const errorMessage = document.getElementById('errorMessage');
const warningMessage = document.getElementById('warningMessage');
const recentList = document.getElementById('recentList');
const clearHistoryButton = document.getElementById('clearHistory');
const themeToggle = document.getElementById('themeToggle');
const customUnitsButton = document.getElementById('customUnitsButton');
const fromUnitSearchTrigger = document.getElementById('fromUnitSearchTrigger');
const toUnitSearchTrigger = document.getElementById('toUnitSearchTrigger');
const copyResultButton = document.getElementById('copyResult');
const installPrompt = document.getElementById('installPrompt');
const dismissInstall = document.getElementById('dismissInstall');

// Gas flow & heating value parameters
const gasFlowParams = document.getElementById('gasFlowParams');
const heatingValueParams = document.getElementById('heatingValueParams');
const standardConditionSelect = document.getElementById('standardCondition');
const gasTypeSelect = document.getElementById('gasType');
const temperatureInput = document.getElementById('temperature');
const pressureInput = document.getElementById('pressure');
const gasDensityInput = document.getElementById('gasDensity');

// Custom units modal
const customUnitsModal = document.getElementById('customUnitsModal');
const closeModal = document.getElementById('closeModal');
const customCategorySelect = document.getElementById('customCategory');
const customUnitInput = document.getElementById('customUnit');
const referenceUnitSelect = document.getElementById('referenceUnit');
const conversionFactorInput = document.getElementById('conversionFactor');
const customAliasesInput = document.getElementById('customAliases');
const addCustomUnitButton = document.getElementById('addCustomUnit');
const customUnitsList = document.getElementById('customUnitsList');

// State
let currentCategory = 'length';
let conversionHistory = [];
let searchTimeout = null;
let clearHistoryTimeout = null;

// Initialize
function init() {
  loadTheme();
  loadHistory();
  loadStandardConditionPreference();
  setupEventListeners();
  populateUnits(currentCategory);
  updateConditionalParams();
  renderCustomUnitsList();
  checkInstallPrompt();

  // Set default values for quick demo
  fromValueInput.value = '1';
  performConversion();
}

// Theme Management
function loadTheme() {
  const savedTheme = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
}

// Standard Condition Preference Management
function loadStandardConditionPreference() {
  const saved = localStorage.getItem('standardCondition');
  if (saved && standardConditionSelect) {
    // Check if the saved value exists in the dropdown
    const options = Array.from(standardConditionSelect.options).map(opt => opt.value);
    if (options.includes(saved)) {
      standardConditionSelect.value = saved;
    } else {
      // Default to SCFM_60F if saved value is invalid
      standardConditionSelect.value = 'SCFM_60F';
    }
  } else if (standardConditionSelect) {
    // Default to SCFM_60F
    standardConditionSelect.value = 'SCFM_60F';
  }
}

function saveStandardConditionPreference() {
  if (standardConditionSelect) {
    localStorage.setItem('standardCondition', standardConditionSelect.value);
  }
}

// Unit Search/Autocomplete
function setupUnitSearch(searchInput, dropdown, unitSelect) {
  let selectedIndex = -1;

  searchInput.addEventListener('focus', () => {
    unitSelect.style.display = 'none';
    searchInput.style.display = 'block';
  });

  searchInput.addEventListener('input', e => {
    const query = e.target.value.trim();

    // Reset selection on input change
    selectedIndex = -1;

    if (!query) {
      dropdown.style.display = 'none';
      return;
    }

    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      const results = searchUnits(query, currentCategory);

      if (results.length === 0) {
        dropdown.innerHTML = '<div class="dropdown-item dropdown-empty">No units found</div>';
        dropdown.style.display = 'block';
        return;
      }

      dropdown.innerHTML = results
        .slice(0, 10)
        .map(
          result => `
        <div class="dropdown-item" data-unit="${escapeHtml(result.unit)}">
          <span class="dropdown-unit">${escapeHtml(result.unit)}</span>
          ${result.matchedAlias ? `<span class="dropdown-alias">(${escapeHtml(result.matchedAlias)})</span>` : ''}
        </div>
      `
        )
        .join('');

      dropdown.style.display = 'block';

      // Add click handlers
      dropdown.querySelectorAll('.dropdown-item[data-unit]').forEach(item => {
        item.addEventListener('click', () => {
          const selectedUnit = item.dataset.unit;
          unitSelect.value = selectedUnit;
          searchInput.value = selectedUnit;
          dropdown.style.display = 'none';
          performConversion();
        });
      });
    }, 150);
  });

  searchInput.addEventListener('blur', () => {
    setTimeout(() => {
      dropdown.style.display = 'none';
      // Show select again if search is empty
      if (!searchInput.value) {
        searchInput.style.display = 'none';
        unitSelect.style.display = 'block';
      }
    }, 200);
  });

  searchInput.addEventListener('keydown', e => {
    const items = dropdown.querySelectorAll('.dropdown-item[data-unit]');
    if (items.length === 0 && e.key !== 'Escape') {
      if (e.key === 'Enter') {
        e.preventDefault();
      }
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      selectedIndex = (selectedIndex + 1) % items.length;
      updateSelection(items);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      selectedIndex = (selectedIndex - 1 + items.length) % items.length;
      updateSelection(items);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (selectedIndex >= 0 && items[selectedIndex]) {
        items[selectedIndex].click();
      } else if (items.length > 0) {
        // Default to first item if none selected
        items[0].click();
      }
    } else if (e.key === 'Escape') {
      dropdown.style.display = 'none';
      searchInput.blur();
    }
  });

  function updateSelection(items) {
    items.forEach((item, index) => {
      if (index === selectedIndex) {
        item.classList.add('selected');
        item.scrollIntoView({ block: 'nearest' });
      } else {
        item.classList.remove('selected');
      }
    });
  }
}

// Unit Population
function populateUnits(category) {
  const units = getUnitsForCategory(category);

  // Clear existing options
  fromUnitSelect.innerHTML = '';
  toUnitSelect.innerHTML = '';

  // Populate both selects
  units.forEach(unit => {
    const option1 = new Option(unit, unit);
    const option2 = new Option(unit, unit);
    fromUnitSelect.add(option1);
    toUnitSelect.add(option2);
  });

  // Set default selections
  if (units.length > 1) {
    fromUnitSelect.selectedIndex = 0;
    toUnitSelect.selectedIndex = 1;
  }

  // Clear search inputs
  fromUnitSearch.value = '';
  toUnitSearch.value = '';

  performConversion();
}

// Conditional Parameters
function updateConditionalParams() {
  const category = currentCategory;

  // Gas flow parameters
  if (category === 'gas_flow') {
    gasFlowParams.style.display = 'block';
  } else {
    gasFlowParams.style.display = 'none';
  }

  // Heating value parameters
  if (category === 'heating_value') {
    heatingValueParams.style.display = 'block';
  } else {
    heatingValueParams.style.display = 'none';
  }
}

// Conversion
function performConversion(direction = 'from') {
  try {
    hideError();
    hideWarning();

    const value = parseFloat(direction === 'from' ? fromValueInput.value : toValueInput.value);
    if (isNaN(value)) {
      if (direction === 'from') {
        toValueInput.value = '';
      } else {
        fromValueInput.value = '';
      }
      return;
    }

    const fromUnit = fromUnitSelect.value;
    const toUnit = toUnitSelect.value;

    if (!fromUnit || !toUnit) {
      return;
    }

    // Prepare options for special conversions
    const options = {};

    // Gas flow options
    if (currentCategory === 'gas_flow') {
      const temp = parseFloat(temperatureInput.value);
      const press = parseFloat(pressureInput.value);

      if (temp) {
        options.temperature = temp;
      }
      if (press) {
        options.pressure = press;
      }
      options.gasType = gasTypeSelect.value;
      options.standardCondition = standardConditionSelect.value || 'SCFM_60F';
    }

    // Heating value options
    if (currentCategory === 'heating_value') {
      const density = parseFloat(gasDensityInput.value);
      if (density) {
        options.gasDensityStp = density;
      }
    }

    // Perform conversion
    let result;
    if (direction === 'from') {
      result = convert(value, fromUnit, toUnit, options);
      toValueInput.value = formatNumber(result);
      addToHistory(value, fromUnit, result, toUnit, currentCategory);
    } else {
      result = convert(value, toUnit, fromUnit, options);
      fromValueInput.value = formatNumber(result);
      addToHistory(result, fromUnit, value, toUnit, currentCategory);
    }

    // Check if using custom units
    checkCustomUnitsWarning(fromUnit, toUnit);
  } catch (error) {
    showError(error.message);
  }
}

function swapUnits() {
  // Swap unit selections
  const fromUnit = fromUnitSelect.value;
  const toUnit = toUnitSelect.value;

  fromUnitSelect.value = toUnit;
  toUnitSelect.value = fromUnit;

  // Update search inputs
  fromUnitSearch.value = toUnit;
  toUnitSearch.value = fromUnit;

  // Swap values
  const fromValue = fromValueInput.value;
  const toValue = toValueInput.value;

  fromValueInput.value = toValue;
  toValueInput.value = fromValue;

  // Animate button
  swapButton.style.transform = 'rotate(180deg)';
  setTimeout(() => {
    swapButton.style.transform = '';
  }, 300);
}

// Custom Units Warning
function checkCustomUnitsWarning(fromUnit, toUnit) {
  const isFromCustom = customUnitManager.isCustomUnit(currentCategory, fromUnit);
  const isToCustom = customUnitManager.isCustomUnit(currentCategory, toUnit);

  if (isFromCustom || isToCustom) {
    const units = [];
    if (isFromCustom) {
      units.push(fromUnit);
    }
    if (isToCustom) {
      units.push(toUnit);
    }
    showWarning(
      `Custom unit${units.length > 1 ? 's' : ''} in use: ${units.join(', ')}. Verify conversion factors.`
    );
  }
}

// History Management
function loadHistory() {
  try {
    const saved = localStorage.getItem('conversionHistory');
    conversionHistory = saved ? JSON.parse(saved) : [];
    renderHistory();
  } catch {
    conversionHistory = [];
  }
}

function saveHistory() {
  try {
    const trimmed = conversionHistory.slice(0, 20);
    localStorage.setItem('conversionHistory', JSON.stringify(trimmed));
  } catch {
    // Silent fail for localStorage errors
  }
}

function addToHistory(fromValue, fromUnit, toValue, toUnit, category) {
  const item = {
    fromValue,
    fromUnit,
    toValue,
    toUnit,
    category,
    timestamp: Date.now()
  };

  // Remove duplicates
  conversionHistory = conversionHistory.filter(
    h => !(h.fromValue === fromValue && h.fromUnit === fromUnit && h.toUnit === toUnit)
  );

  conversionHistory.unshift(item);
  saveHistory();
  renderHistory();
}

function clearHistory() {
  if (clearHistoryButton.classList.contains('confirming')) {
    conversionHistory = [];
    localStorage.removeItem('conversionHistory');
    renderHistory();
    resetClearButton();
  } else {
    clearHistoryButton.classList.add('confirming');
    clearHistoryButton.textContent = 'Confirm?';
    clearHistoryTimeout = setTimeout(resetClearButton, 3000);
  }
}

function resetClearButton() {
  clearHistoryButton.classList.remove('confirming');
  clearHistoryButton.textContent = 'Clear';
  if (clearHistoryTimeout) {
    clearTimeout(clearHistoryTimeout);
    clearHistoryTimeout = null;
  }
}

function renderHistory() {
  if (conversionHistory.length === 0) {
    recentList.innerHTML = '<p class="empty-state">No recent conversions</p>';
    return;
  }

  recentList.innerHTML = conversionHistory
    .map(item => {
      const timeAgo = formatTimeAgo(item.timestamp);
      return `
      <div class="recent-item" data-index="${conversionHistory.indexOf(item)}">
        <div class="recent-item-text">
          ${formatNumber(item.fromValue)} ${escapeHtml(item.fromUnit)} = ${formatNumber(item.toValue)} ${escapeHtml(item.toUnit)}
        </div>
        <div class="recent-item-time">${escapeHtml(getCategoryLabel(item.category))} • ${timeAgo}</div>
      </div>
    `;
    })
    .join('');

  // Add click handlers
  document.querySelectorAll('.recent-item').forEach(item => {
    item.addEventListener('click', () => {
      const index = parseInt(item.dataset.index);
      loadFromHistory(conversionHistory[index]);
    });
  });
}

function loadFromHistory(item) {
  categorySelect.value = item.category;
  currentCategory = item.category;
  populateUnits(item.category);
  updateConditionalParams();

  setTimeout(() => {
    fromValueInput.value = item.fromValue;
    fromUnitSelect.value = item.fromUnit;
    toUnitSelect.value = item.toUnit;
    fromUnitSearch.value = item.fromUnit;
    toUnitSearch.value = item.toUnit;
    performConversion();
  }, 10);
}

// Custom Units Management
function openCustomUnitsModal() {
  customUnitsModal.style.display = 'flex';
  populateReferenceUnits();
  renderCustomUnitsList();
}

function closeCustomUnitsModal() {
  customUnitsModal.style.display = 'none';
}

function populateReferenceUnits() {
  const category = customCategorySelect.value;
  const units = getUnitsForCategory(category);

  referenceUnitSelect.innerHTML = '';
  units.forEach(unit => {
    const option = new Option(unit, unit);
    referenceUnitSelect.add(option);
  });
}

function addCustomUnit() {
  try {
    const category = customCategorySelect.value;
    const unit = customUnitInput.value.trim();
    const refUnit = referenceUnitSelect.value;
    const factor = parseFloat(conversionFactorInput.value);
    const aliasesStr = customAliasesInput.value.trim();
    const aliases = aliasesStr
      ? aliasesStr
          .split(',')
          .map(a => a.trim())
          .filter(a => a)
      : [];

    if (!unit) {
      alert('Please enter a unit symbol');
      return;
    }

    if (isNaN(factor) || factor <= 0) {
      alert('Please enter a valid positive conversion factor');
      return;
    }

    const result = customUnitManager.addUnit(category, unit, refUnit, factor, aliases);

    // Clear form
    customUnitInput.value = '';
    conversionFactorInput.value = '';
    customAliasesInput.value = '';

    // Update UI
    renderCustomUnitsList();

    // If current category matches, repopulate units
    if (currentCategory === category) {
      populateUnits(currentCategory);
    }

    alert(result.message);
  } catch (error) {
    alert('Error: ' + error.message);
  }
}

function removeCustomUnit(category, unit) {
  if (confirm(`Remove custom unit '${unit}'?`)) {
    try {
      customUnitManager.removeUnit(category, unit);
      renderCustomUnitsList();

      // If current category matches, repopulate units
      if (currentCategory === category) {
        populateUnits(currentCategory);
      }
    } catch (error) {
      alert('Error: ' + error.message);
    }
  }
}

function renderCustomUnitsList() {
  const customUnits = customUnitManager.getCustomUnits();

  if (Object.keys(customUnits).length === 0) {
    customUnitsList.innerHTML = '<p class="empty-state">No custom units defined</p>';
    return;
  }

  customUnitsList.innerHTML = Object.keys(customUnits)
    .map(category => {
      const units = customUnits[category];
      return `
      <div class="custom-category">
        <h4 class="custom-category-title">${escapeHtml(getCategoryLabel(category))}</h4>
        <div class="custom-units-items">
          ${Object.keys(units)
            .map(
              unit => `
            <div class="custom-unit-item">
              <span class="custom-unit-name">${escapeHtml(unit)}</span>
              <button class="custom-unit-remove" data-category="${category}" data-unit="${escapeHtml(unit)}">×</button>
            </div>
          `
            )
            .join('')}
        </div>
      </div>
    `;
    })
    .join('');

  // Add remove handlers
  customUnitsList.querySelectorAll('.custom-unit-remove').forEach(btn => {
    btn.addEventListener('click', () => {
      removeCustomUnit(btn.dataset.category, btn.dataset.unit);
    });
  });
}

// Copy Result
async function copyResult() {
  const value = toValueInput.value;
  if (!value) {
    return;
  }

  try {
    await navigator.clipboard.writeText(value);

    // Visual feedback
    const originalContent = copyResultButton.innerHTML;
    copyResultButton.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="20 6 9 17 4 12"></polyline>
      </svg>
    `;
    copyResultButton.classList.add('success');
    copyResultButton.setAttribute('aria-label', 'Copied!');
    copyResultButton.setAttribute('title', 'Copied!');

    setTimeout(() => {
      copyResultButton.innerHTML = originalContent;
      copyResultButton.classList.remove('success');
      copyResultButton.setAttribute('aria-label', 'Copy result');
      copyResultButton.setAttribute('title', 'Copy result');
    }, 2000);
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error('Failed to copy:', err);
    showWarning('Failed to copy to clipboard');
    setTimeout(hideWarning, 3000);
  }
}

// Error/Warning Handling
function showError(message) {
  errorMessage.textContent = message;
  errorMessage.style.display = 'block';
}

function hideError() {
  errorMessage.style.display = 'none';
}

function showWarning(message) {
  warningMessage.textContent = '⚠️ ' + message;
  warningMessage.style.display = 'block';
}

function hideWarning() {
  warningMessage.style.display = 'none';
}

// Utilities
function escapeHtml(unsafe) {
  if (unsafe === undefined || unsafe === null) {
    return '';
  }
  return String(unsafe)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatNumber(num) {
  if (typeof num !== 'number' || isNaN(num)) {
    return '';
  }

  // Use scientific notation for very large or very small numbers
  if (Math.abs(num) >= 1e10 || (Math.abs(num) < 1e-6 && num !== 0)) {
    return num.toExponential(6);
  }

  // For normal numbers, use up to 10 significant digits
  const str = num.toPrecision(10);
  const parsed = parseFloat(str);

  return parsed.toString();
}

function formatTimeAgo(timestamp) {
  const seconds = Math.floor((Date.now() - timestamp) / 1000);

  if (seconds < 60) {
    return 'just now';
  }
  if (seconds < 3600) {
    return `${Math.floor(seconds / 60)}m ago`;
  }
  if (seconds < 86400) {
    return `${Math.floor(seconds / 3600)}h ago`;
  }
  if (seconds < 604800) {
    return `${Math.floor(seconds / 86400)}d ago`;
  }

  return new Date(timestamp).toLocaleDateString();
}

function getCategoryLabel(category) {
  const labels = {
    length: 'Length',
    mass: 'Mass',
    volume: 'Volume',
    temperature: 'Temperature',
    pressure: 'Pressure',
    energy: 'Energy',
    power: 'Power',
    gas_flow: 'Gas Flow',
    heating_value: 'Heating Value',
    mass_flow: 'Mass Flow',
    volumetric_flow: 'Volumetric Flow',
    area: 'Area',
    time: 'Time',
    density: 'Density',
    dynamic_viscosity: 'Dynamic Viscosity',
    kinematic_viscosity: 'Kinematic Viscosity',
    thermal_conductivity: 'Thermal Conductivity',
    heat_transfer: 'Heat Transfer',
    specific_heat: 'Specific Heat'
  };

  return labels[category] || category;
}

// Install Prompt
function checkInstallPrompt() {
  if (window.matchMedia('(display-mode: standalone)').matches) {
    return;
  }

  const dismissed = localStorage.getItem('installPromptDismissed');
  if (dismissed) {
    return;
  }

  const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
  if (isIOS) {
    setTimeout(() => {
      installPrompt.style.display = 'block';
    }, 3000);
  }
}

function dismissInstallPrompt() {
  installPrompt.style.display = 'none';
  localStorage.setItem('installPromptDismissed', 'true');
}

// Event Listeners
function setupEventListeners() {
  // Category change
  categorySelect.addEventListener('change', e => {
    currentCategory = e.target.value;
    populateUnits(currentCategory);
    updateConditionalParams();
  });

  // From value input
  fromValueInput.addEventListener(
    'input',
    debounce(() => {
      performConversion('from');
    }, 300)
  );

  // To value input (bidirectional conversion)
  toValueInput.addEventListener(
    'input',
    debounce(() => {
      performConversion('to');
    }, 300)
  );

  // Unit selects
  fromUnitSelect.addEventListener('change', () => {
    fromUnitSearch.value = fromUnitSelect.value;
    performConversion('from');
  });

  toUnitSelect.addEventListener('change', () => {
    toUnitSearch.value = toUnitSelect.value;
    performConversion('from');
  });

  // Search triggers
  fromUnitSearchTrigger.addEventListener('click', () => {
    fromUnitSelect.style.display = 'none';
    fromUnitSearch.style.display = 'block';
    fromUnitSearch.focus();
  });

  toUnitSearchTrigger.addEventListener('click', () => {
    toUnitSelect.style.display = 'none';
    toUnitSearch.style.display = 'block';
    toUnitSearch.focus();
  });

  // Search inputs
  setupUnitSearch(fromUnitSearch, fromUnitDropdown, fromUnitSelect);
  setupUnitSearch(toUnitSearch, toUnitDropdown, toUnitSelect);

  // Gas flow / heating value parameters
  standardConditionSelect.addEventListener('change', () => {
    saveStandardConditionPreference();
    performConversion();
  });
  gasTypeSelect.addEventListener('change', () => performConversion());
  temperatureInput.addEventListener('input', () => performConversion());
  pressureInput.addEventListener('input', () => performConversion());
  gasDensityInput.addEventListener('input', () => performConversion());

  // Swap button
  swapButton.addEventListener('click', swapUnits);

  // Theme toggle
  themeToggle.addEventListener('click', toggleTheme);

  // Clear history
  clearHistoryButton.addEventListener('click', clearHistory);

  // Custom units
  customUnitsButton.addEventListener('click', openCustomUnitsModal);
  closeModal.addEventListener('click', closeCustomUnitsModal);

  // Copy result
  if (copyResultButton) {
    copyResultButton.addEventListener('click', copyResult);
  }
  customCategorySelect.addEventListener('change', populateReferenceUnits);
  addCustomUnitButton.addEventListener('click', addCustomUnit);

  // Modal backdrop click
  customUnitsModal
    .querySelector('.modal-backdrop')
    .addEventListener('click', closeCustomUnitsModal);

  // Install prompt dismiss
  if (dismissInstall) {
    dismissInstall.addEventListener('click', dismissInstallPrompt);
  }

  // Keyboard shortcuts
  document.addEventListener('keydown', e => {
    // Ctrl/Cmd + K to focus from value
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      fromValueInput.focus();
      fromValueInput.select();
    }

    // Ctrl/Cmd + Shift + S to swap
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'S') {
      e.preventDefault();
      swapUnits();
    }

    // Escape to close modal
    if (e.key === 'Escape' && customUnitsModal.style.display === 'flex') {
      closeCustomUnitsModal();
    }
  });
}

// Service Worker Registration
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker
      .register('service-worker.js')
      .then(() => {
        // Service worker registered successfully
      })
      .catch(() => {
        // Service worker registration failed
      });
  });
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
