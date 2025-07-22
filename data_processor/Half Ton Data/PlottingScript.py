import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.drawing.image import Image

# Load CSV file
file_path = "C:\Users\diete\Dropbox\Programming\Plotting Python Script\2024-04-17 Data.csv" 
cd C:\Users\diete\Dropbox\Programming\DataPlotterPython\
df = pd.read_csv(file_path)

# Convert 'time' column to datetime
df["time"] = pd.to_datetime(df["time"])

# Sort by time to ensure correct ordering
df = df.sort_values(by="time")

# Calculate rolling one-minute average for h2_pct
df["h2_avg"] = df["h2_pct"].rolling("1T", on="time").mean()

# Replace outliers with the rolling one-minute average
df["h2_filtered"] = np.where(
    (df["h2_pct"] > 1.1 * df["h2_avg"]) | (df["h2_pct"] < 0.9 * df["h2_avg"]),
    df["h2_avg"],  # Replace outliers with rolling average
    df["h2_pct"]
)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df["time"], df["co_pct"], marker="o", linestyle="-", label="CO %")
plt.plot(df["time"], df["co2_pct"], marker="o", linestyle="-", label="CO2 %")
plt.plot(df["time"], df["ch4_pct"], marker="o", linestyle="-", label="CH4 %")
plt.plot(df["time"], df["h2_filtered"], marker="o", linestyle="-", label="Filtered H2 %", color="red")
plt.xlabel("Time")
plt.ylabel("Percentage (%)")
plt.legend()
plt.title("Gas Concentrations Over Time")
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot to an image file
plot_path = "plot.png"
plt.savefig(plot_path)
plt.close()

# Save to an Excel file
wb = Workbook()
ws = wb.active
ws.title = "Gas Data"
ws.append(["Time", "CO %", "CO2 %", "CH4 %", "Filtered H2 %"])

# Write data to Excel
for _, row in df.iterrows():
    ws.append([row["time"], row["co_pct"], row["co2_pct"], row["ch4_pct"], row["h2_filtered"]])

# Insert the plot into the Excel file
img = Image(plot_path)
ws.add_image(img, "F2")

# Save the Excel file
wb.save("gas_data.xlsx")

print("Processing complete! The filtered data and plot have been saved to 'gas_data.xlsx'.")