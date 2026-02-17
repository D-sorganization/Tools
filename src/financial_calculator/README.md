# Financial Calculator

A comprehensive financial modeling tool for plant operations analysis, featuring NPV calculations, IRR computation, payback period analysis, and 10-year pro forma projections. Available as both PyQt6 desktop and React web applications.

## Purpose

The Financial Calculator enables project economists and engineers to:

- Evaluate capital investment viability
- Perform comprehensive financial modeling
- Generate multi-year cash flow projections
- Calculate key financial metrics (NPV, IRR, ROE, Payback)

## Key Features

- **NPV/IRR Analysis**: Net Present Value and Internal Rate of Return calculations
- **Payback Period**: Time to recover initial investment
- **10-Year Pro Forma**: Detailed annual projections with cumulative cash flow
- **EBITDA Calculation**: Earnings Before Interest, Taxes, Depreciation, Amortization
- **Return on Equity**: Profitability relative to shareholder equity
- **Variable/Fixed Cost Modeling**: Comprehensive cost structure analysis
- **Debt Financing**: Interest expense and debt ratio modeling
- **Dual Interface**: PyQt6 desktop and React/TypeScript web versions

## Installation

### Desktop Application (PyQt6)

```bash
pip install PyQt6
cd Tools/src/financial_calculator
python launch_pyqt6.py
```

### Web Application (React)

```bash
cd Tools/src/financial_calculator/web
npm install
npm run dev
```

### Full Dependencies

```bash
# Python backend
pip install PyQt6 numpy

# Web frontend
npm install react react-dom vite tailwindcss
```

## Usage Instructions

### Desktop Application

1. Launch: `python launch_pyqt6.py`
2. Enter plant operations parameters
3. Configure revenue and cost structure
4. Set capital and financing terms
5. Click "Calculate Financial Model"
6. Review results in Summary and Projections tabs

### Web Application

1. Start dev server: `npm run dev`
2. Open browser to `http://localhost:5173`
3. Enter parameters in left panel
4. Click "Calculate Financial Model"
5. View results in right panel

## Input Parameters

### Plant Operations

| Parameter            | Description           | Default | Unit           |
| -------------------- | --------------------- | ------- | -------------- |
| Plant Capacity       | Daily throughput      | 100     | TPD (tons/day) |
| Operating Days       | Annual operating days | 330     | days/year      |
| Capacity Utilization | Operating efficiency  | 85      | %              |

### Revenue Parameters

| Parameter         | Description                | Default | Unit  |
| ----------------- | -------------------------- | ------- | ----- |
| Product Price     | Main product selling price | 500     | $/ton |
| Byproduct Revenue | Secondary product revenue  | 50      | $/ton |
| Byproduct Yield   | Byproduct fraction         | 10      | %     |

### Variable Costs ($/ton)

| Parameter        | Description          | Default |
| ---------------- | -------------------- | ------- |
| Feedstock Cost   | Raw material cost    | 200     |
| Labor Cost       | Variable labor       | 30      |
| Utilities Cost   | Power, water, etc.   | 40      |
| Maintenance Cost | Variable maintenance | 15      |
| Consumables      | Supplies, catalysts  | 10      |

### Fixed Costs ($/year)

| Parameter      | Description        | Default |
| -------------- | ------------------ | ------- |
| Fixed Labor    | Salaries, benefits | 500,000 |
| Insurance      | Annual premium     | 100,000 |
| Property Tax   | Annual tax         | 50,000  |
| Admin Overhead | G&A expenses       | 200,000 |

### Capital & Financing

| Parameter           | Description         | Default    | Range |
| ------------------- | ------------------- | ---------- | ----- |
| Total Capital       | Initial investment  | 10,000,000 | $     |
| Debt Ratio          | Leverage percentage | 60         | %     |
| Interest Rate       | Annual debt cost    | 7          | %     |
| Depreciation Period | Asset useful life   | 10         | years |
| Tax Rate            | Corporate tax rate  | 25         | %     |

## Output Format

### Summary Metrics

| Metric           | Description             | Example     |
| ---------------- | ----------------------- | ----------- |
| Annual Feedstock | Total annual throughput | 28,050 tons |
| Total Revenue    | Gross annual revenue    | $12,543,750 |
| Total Costs      | Variable + Fixed costs  | $9,876,500  |
| Net Income       | After-tax profit        | $1,834,687  |
| EBITDA           | Operating cash flow     | $2,667,250  |
| Return on Equity | ROE percentage          | 45.9%       |
| Payback Period   | Capital recovery time   | 3.8 years   |

### 10-Year Projections Table

| Year | Revenue | Costs | EBITDA | Net Income | Cumulative CF |
| ---- | ------- | ----- | ------ | ---------- | ------------- |
| 1    | $12.5M  | $9.9M | $2.7M  | $1.8M      | -$8.2M        |
| 2    | $12.5M  | $9.9M | $2.7M  | $1.8M      | -$5.4M        |
| ...  | ...     | ...   | ...    | ...        | ...           |
| 10   | $12.5M  | $9.9M | $2.7M  | $1.8M      | $11.3M        |

## Example Usage

### Basic Project Evaluation

```bash
# Launch desktop app
python launch_pyqt6.py

# Configure a 200 TPD plant:
# - Plant Capacity: 200 TPD
# - Operating Days: 350
# - Utilization: 90%
# - Product Price: $600/ton
# - Total Capital: $25,000,000

# Click Calculate to see results
```

### Programmatic Calculations

```python
# NPV Calculation
def calculate_npv(cash_flows, discount_rate):
    npv = 0
    for t, cf in enumerate(cash_flows):
        npv += cf / (1 + discount_rate) ** t
    return npv

# IRR Calculation (using scipy)
from scipy.optimize import brentq

def calculate_irr(cash_flows):
    def npv_at_rate(r):
        return sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows))
    return brentq(npv_at_rate, -0.99, 10.0)

# Example
cash_flows = [-10000000, 2667250, 2667250, 2667250, 2667250, 2667250]
irr = calculate_irr(cash_flows) * 100
print(f"IRR: {irr:.1f}%")
```

### Sensitivity Analysis

```python
# Vary product price from $400 to $600
prices = range(400, 601, 50)
for price in prices:
    # Recalculate with new price
    revenue = annual_tons * price
    # ... compute NPV
    print(f"Price ${price}/ton -> NPV: ${npv:,.0f}")
```

## Financial Formulas

### Annual Production

```
Annual Feedstock = Capacity (TPD) x Operating Days x Utilization
Product Tons = Annual Feedstock x Product Yield Factor
```

### Revenue Calculation

```
Total Revenue = Product Revenue + Byproduct Revenue
Product Revenue = Product Tons x Product Price
Byproduct Revenue = Byproduct Tons x Byproduct Price
```

### Cost Structure

```
Variable Costs = Annual Tons x (Feedstock + Labor + Utilities + Maintenance + Consumables)
Fixed Costs = Fixed Labor + Insurance + Property Tax + Admin
Total Costs = Variable Costs + Fixed Costs
```

### Profitability Metrics

```
Gross Margin = Revenue - Variable Costs
EBITDA = Gross Margin - Fixed Costs
EBIT = EBITDA - Depreciation
Interest Expense = Debt Amount x Interest Rate
EBT = EBIT - Interest Expense
Net Income = EBT - Taxes
```

### Return on Equity

```
Equity = Total Capital x (1 - Debt Ratio)
ROE = Net Income / Equity x 100%
```

### Payback Period

```
Cash Flow = Net Income + Depreciation
Payback = Total Capital / Annual Cash Flow
```

## Troubleshooting

### Negative Net Income

**Causes**:

- Product price too low
- Costs too high
- Low utilization

**Solutions**:

- Increase product price
- Reduce variable costs
- Improve capacity utilization

### Very Long Payback Period

**Causes**:

- High capital cost
- Low operating margin
- Excessive debt service

**Solutions**:

- Review capital cost estimates
- Optimize cost structure
- Consider different financing terms

### ROE Calculation Issues

**Issue**: ROE shows infinity or very high values

**Cause**: Debt ratio near 100% (minimal equity)

**Solution**: Reduce debt ratio to reasonable levels (40-70%)

### Web App Not Loading

```bash
# Clear npm cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm run dev
```

## Related Tools

- **Optimizer GUI**: For optimizing financial parameters
- **Multi-Parameter Analysis**: For sensitivity studies on financial inputs
- **Data Processor**: For importing historical cost/revenue data

## Technical Notes

### Depreciation Method

Straight-line depreciation is used:

```
Annual Depreciation = Total Capital / Depreciation Years
```

### Tax Calculation

Taxes are only applied to positive earnings:

```python
taxes = max(0, ebt * tax_rate)
```

### Inflation Assumptions

The current model assumes constant prices. For inflation modeling:

```python
escalation_rate = 0.03  # 3% annual
year_n_price = base_price * (1 + escalation_rate) ** n
```

## Version History

- **1.0.0**: Initial PyQt6 release
- **1.1.0**: Added 10-year projections
- **1.2.0**: Web application (React/TypeScript)
- **1.3.0**: Catppuccin Mocha theme integration
- **1.4.0**: Added byproduct revenue modeling
