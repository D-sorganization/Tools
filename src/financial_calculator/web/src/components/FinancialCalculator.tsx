import { useState } from "react";

// Catppuccin Mocha colors
const colors = {
  base: "#1e1e2e",
  mantle: "#181825",
  surface0: "#313244",
  surface1: "#45475a",
  text: "#cdd6f4",
  subtext0: "#a6adc8",
  blue: "#89b4fa",
  green: "#a6e3a1",
  red: "#f38ba8",
  yellow: "#f9e2af",
  peach: "#fab387",
  mauve: "#cba6f7",
  teal: "#94e2d5",
  lavender: "#b4befe",
};

interface Results {
  annualFeedstockTons: number;
  totalRevenue: number;
  totalCosts: number;
  netIncome: number;
  ebitda: number;
  roe: number;
  paybackYears: number;
}

function FinancialCalculator() {
  const [plantCapacity, setPlantCapacity] = useState(100);
  const [operatingDays, setOperatingDays] = useState(330);
  const [utilization, setUtilization] = useState(85);
  const [productPrice, setProductPrice] = useState(500);
  const [feedstockCost, setFeedstockCost] = useState(200);
  const [laborCost, setLaborCost] = useState(30);
  const [utilitiesCost, setUtilitiesCost] = useState(40);
  const [maintenanceCost, setMaintenanceCost] = useState(15);
  const [fixedLabor, setFixedLabor] = useState(500000);
  const [insurance, setInsurance] = useState(100000);
  const [capital, setCapital] = useState(10000000);
  const [debtRatio, setDebtRatio] = useState(60);
  const [interestRate, setInterestRate] = useState(7);
  const [depreciationYears, setDepreciationYears] = useState(10);
  const [taxRate, setTaxRate] = useState(25);
  const [results, setResults] = useState<Results | null>(null);

  const calculate = () => {
    const util = utilization / 100;
    const annualFeedstock = plantCapacity * operatingDays * util;
    const productTons = annualFeedstock * 0.85;
    const byproductTons = annualFeedstock * 0.1;

    const productRevenue = productTons * productPrice;
    const byproductRevenue = byproductTons * 50;
    const totalRevenue = productRevenue + byproductRevenue;

    const variableCosts =
      annualFeedstock *
      (feedstockCost + laborCost + utilitiesCost + maintenanceCost + 10);
    const fixedCosts = fixedLabor + insurance + 50000 + 200000;
    const totalCosts = variableCosts + fixedCosts;

    const grossMargin = totalRevenue - variableCosts;
    const ebitda = grossMargin - fixedCosts;
    const depreciation = capital / depreciationYears;
    const ebit = ebitda - depreciation;

    const debtAmount = capital * (debtRatio / 100);
    const interestExpense = debtAmount * (interestRate / 100);
    const ebt = ebit - interestExpense;
    const taxes = Math.max(0, ebt * (taxRate / 100));
    const netIncome = ebt - taxes;

    const equity = capital * (1 - debtRatio / 100);
    const roe = equity > 0 ? (netIncome / equity) * 100 : 0;
    const cashFlow = netIncome + depreciation;
    const paybackYears = cashFlow > 0 ? capital / cashFlow : 0;

    setResults({
      annualFeedstockTons: annualFeedstock,
      totalRevenue,
      totalCosts,
      netIncome,
      ebitda,
      roe,
      paybackYears,
    });
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(
      value,
    );
  };

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-8" style={{ color: colors.blue }}>
        Financial Calculator
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Panel */}
        <div className="space-y-6">
          {/* Plant Operations */}
          <div
            className="rounded-lg p-4"
            style={{
              backgroundColor: colors.mantle,
              border: `1px solid ${colors.surface1}`,
            }}
          >
            <h2
              className="text-lg font-semibold mb-4"
              style={{ color: colors.lavender }}
            >
              Plant Operations
            </h2>
            <div className="space-y-3">
              <InputField
                label="Plant Capacity (TPD)"
                value={plantCapacity}
                onChange={setPlantCapacity}
                min={0}
                max={10000}
              />
              <InputField
                label="Operating Days/Year"
                value={operatingDays}
                onChange={setOperatingDays}
                min={0}
                max={365}
              />
              <InputField
                label="Capacity Utilization (%)"
                value={utilization}
                onChange={setUtilization}
                min={0}
                max={100}
              />
            </div>
          </div>

          {/* Revenue */}
          <div
            className="rounded-lg p-4"
            style={{
              backgroundColor: colors.mantle,
              border: `1px solid ${colors.surface1}`,
            }}
          >
            <h2
              className="text-lg font-semibold mb-4"
              style={{ color: colors.lavender }}
            >
              Revenue Parameters
            </h2>
            <InputField
              label="Product Price ($/ton)"
              value={productPrice}
              onChange={setProductPrice}
              min={0}
              max={10000}
            />
          </div>

          {/* Variable Costs */}
          <div
            className="rounded-lg p-4"
            style={{
              backgroundColor: colors.mantle,
              border: `1px solid ${colors.surface1}`,
            }}
          >
            <h2
              className="text-lg font-semibold mb-4"
              style={{ color: colors.lavender }}
            >
              Variable Costs ($/ton)
            </h2>
            <div className="space-y-3">
              <InputField
                label="Feedstock"
                value={feedstockCost}
                onChange={setFeedstockCost}
                min={0}
                max={5000}
              />
              <InputField
                label="Labor"
                value={laborCost}
                onChange={setLaborCost}
                min={0}
                max={1000}
              />
              <InputField
                label="Utilities"
                value={utilitiesCost}
                onChange={setUtilitiesCost}
                min={0}
                max={1000}
              />
              <InputField
                label="Maintenance"
                value={maintenanceCost}
                onChange={setMaintenanceCost}
                min={0}
                max={500}
              />
            </div>
          </div>

          {/* Fixed Costs */}
          <div
            className="rounded-lg p-4"
            style={{
              backgroundColor: colors.mantle,
              border: `1px solid ${colors.surface1}`,
            }}
          >
            <h2
              className="text-lg font-semibold mb-4"
              style={{ color: colors.lavender }}
            >
              Fixed Costs ($/year)
            </h2>
            <div className="space-y-3">
              <InputField
                label="Fixed Labor"
                value={fixedLabor}
                onChange={setFixedLabor}
                min={0}
                max={10000000}
                step={10000}
              />
              <InputField
                label="Insurance"
                value={insurance}
                onChange={setInsurance}
                min={0}
                max={1000000}
                step={10000}
              />
            </div>
          </div>

          {/* Capital & Financing */}
          <div
            className="rounded-lg p-4"
            style={{
              backgroundColor: colors.mantle,
              border: `1px solid ${colors.surface1}`,
            }}
          >
            <h2
              className="text-lg font-semibold mb-4"
              style={{ color: colors.lavender }}
            >
              Capital & Financing
            </h2>
            <div className="space-y-3">
              <InputField
                label="Total Capital ($)"
                value={capital}
                onChange={setCapital}
                min={0}
                max={1000000000}
                step={100000}
              />
              <InputField
                label="Debt Ratio (%)"
                value={debtRatio}
                onChange={setDebtRatio}
                min={0}
                max={100}
              />
              <InputField
                label="Interest Rate (%)"
                value={interestRate}
                onChange={setInterestRate}
                min={0}
                max={30}
                step={0.5}
              />
              <InputField
                label="Depreciation (years)"
                value={depreciationYears}
                onChange={setDepreciationYears}
                min={1}
                max={40}
              />
              <InputField
                label="Tax Rate (%)"
                value={taxRate}
                onChange={setTaxRate}
                min={0}
                max={50}
              />
            </div>
          </div>

          <button
            onClick={calculate}
            className="w-full py-3 rounded-lg font-bold text-lg transition-colors"
            style={{ backgroundColor: colors.blue, color: colors.base }}
          >
            Calculate Financial Model
          </button>
        </div>

        {/* Results Panel */}
        <div className="space-y-6">
          <h2 className="text-2xl font-bold" style={{ color: colors.green }}>
            Financial Analysis Results
          </h2>

          {results && (
            <div className="grid grid-cols-2 gap-4">
              <MetricCard
                label="Annual Feedstock"
                value={`${formatNumber(results.annualFeedstockTons)} tons`}
                color={colors.blue}
              />
              <MetricCard
                label="Total Revenue"
                value={formatCurrency(results.totalRevenue)}
                color={colors.green}
              />
              <MetricCard
                label="Total Costs"
                value={formatCurrency(results.totalCosts)}
                color={colors.red}
              />
              <MetricCard
                label="Net Income"
                value={formatCurrency(results.netIncome)}
                color={colors.yellow}
              />
              <MetricCard
                label="EBITDA"
                value={formatCurrency(results.ebitda)}
                color={colors.peach}
              />
              <MetricCard
                label="Return on Equity"
                value={`${results.roe.toFixed(1)}%`}
                color={colors.mauve}
              />
              <MetricCard
                label="Payback Period"
                value={`${results.paybackYears.toFixed(1)} years`}
                color={colors.teal}
              />
            </div>
          )}

          {!results && (
            <div
              className="rounded-lg p-8 text-center"
              style={{ backgroundColor: colors.surface0 }}
            >
              <p style={{ color: colors.subtext0 }}>
                Enter parameters and click Calculate to see results
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

interface InputFieldProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}

function InputField({
  label,
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
}: InputFieldProps) {
  return (
    <div className="flex items-center justify-between gap-4">
      <label className="text-sm" style={{ color: colors.text }}>
        {label}
      </label>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        min={min}
        max={max}
        step={step}
        className="w-32 px-3 py-2 rounded text-right"
        style={{
          backgroundColor: colors.surface0,
          color: colors.text,
          border: `1px solid ${colors.surface1}`,
        }}
      />
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: string;
  color: string;
}

function MetricCard({ label, value, color }: MetricCardProps) {
  return (
    <div
      className="rounded-lg p-4"
      style={{ backgroundColor: colors.surface0 }}
    >
      <p className="text-sm mb-1" style={{ color: colors.subtext0 }}>
        {label}
      </p>
      <p className="text-xl font-bold" style={{ color }}>
        {value}
      </p>
    </div>
  );
}

export default FinancialCalculator;
