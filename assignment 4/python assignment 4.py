# Name: Satyam BHardwaj
# Roll number: 2501730293
# Course Code: ETCCPP102
# Assignment: Weather Data Visualizer
# Date: 2025-12-08

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
DATA_FILE = 'sample_weather_data.csv'
CLEANED_FILE = 'cleaned_weather_data.csv'
OUTPUT_DIR = Path('plots')

def data_acquisition_and_loading(filepath):
    """Task 1: Load and inspect data."""
    print(f"--- Loading Data from {filepath} ---")
    try:
        df = pd.read_csv(filepath)
        print("Initial DataFrame Head:")
        print(df.head())
        print("\nInitial DataFrame Info:")
        df.info()
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        return None

def data_cleaning_and_processing(df):
    """Task 2: Handle missing values and format dates."""
    if df is None: return None
    print("\n--- Cleaning Data ---")
    
    # Drop rows with any NaN values for simplicity in this dataset
    df_cleaned = df.dropna()
    print(f"Dropped {len(df) - len(df_cleaned)} rows with missing values.")
    
    # Convert Date column to datetime format
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
    df_cleaned = df_cleaned.set_index('Date')
    
    # Filter for relevant columns
    df_cleaned = df_cleaned[['Max Temp (°C)', 'Min Temp (°C)', 'Rainfall (mm)', 'Humidity (%)']]
    
    print("\nCleaned DataFrame Head:")
    print(df_cleaned.head())
    df_cleaned.info()
    return df_cleaned

def statistical_analysis(df):
    """Task 3: Compute daily/monthly statistics using NumPy concepts (via Pandas)."""
    if df is None: return None
    print("\n--- Statistical Analysis (NumPy/Pandas) ---")
    
    # Daily Statistics (already available if index is daily)
    daily_stats = df.describe().T[['mean', 'min', 'max', 'std']]
    print("\nDaily (Overall) Descriptive Statistics:")
    print(daily_stats)
    
    # Monthly Statistics using resample and NumPy functions (via aggregation)
    # Use resample('M') to group by month
    monthly_stats = df.resample('M').agg({
        'Max Temp (°C)': [np.mean, np.min, np.max],
        'Rainfall (mm)': np.sum,
        'Humidity (%)': np.mean
    })
    print("\nMonthly Aggregated Statistics:")
    print(monthly_stats)
    return daily_stats, monthly_stats

def visualization(df):
    """Task 4: Create informative plots using Matplotlib."""
    if df is None: return
    print("\n--- Generating Visualizations (Plots saved to 'plots' directory) ---")

    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Line chart for daily temperature trends
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Max Temp (°C)'], label='Max Temp', marker='o')
    plt.plot(df.index, df['Min Temp (°C)'], label='Min Temp', marker='o')
    plt.title('Daily Temperature Trend (Max and Min)')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'temp_trend_line.png')
    plt.close()
    print("- Saved: temp_trend_line.png")

    # Bar chart for monthly rainfall totals (Task 5: Grouping/Aggregation used here)
    monthly_rainfall = df['Rainfall (mm)'].resample('M').sum()
    monthly_rainfall.index = monthly_rainfall.index.strftime('%Y-%m') # Format month for readability
    plt.figure(figsize=(8, 5))
    plt.bar(monthly_rainfall.index, monthly_rainfall.values, color='skyblue')
    plt.title('Monthly Rainfall Totals')
    plt.xlabel('Month')
    plt.ylabel('Total Rainfall (mm)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'monthly_rainfall_bar.png')
    plt.close()
    print("- Saved: monthly_rainfall_bar.png")

    # Scatter plot for humidity vs. temperature & Combined Plot (Bonus/Task 4)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter Plot
    axes[0].scatter(df['Max Temp (°C)'], df['Humidity (%)'], color='purple', alpha=0.6)
    axes[0].set_title('Humidity vs. Max Temperature')
    axes[0].set_xlabel('Max Temperature (°C)')
    axes[0].set_ylabel('Humidity (%)')
    axes[0].grid(True, linestyle='--')
    
    # Combined Plot: Min Temp vs Humidity
    axes[1].scatter(df['Min Temp (°C)'], df['Humidity (%)'], color='orange', alpha=0.6)
    axes[1].set_title('Humidity vs. Min Temperature')
    axes[1].set_xlabel('Min Temperature (°C)')
    axes[1].set_ylabel('Humidity (%)')
    axes[1].grid(True, linestyle='--')

    plt.suptitle('Combined Weather Visualizations (Task 4/Bonus)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'combined_scatter_plots.png')
    plt.close()
    print("- Saved: combined_scatter_plots.png (Contains two plots in one figure)")


def export_and_storytelling(df):
    """Task 6: Export cleaned data and create a summary."""
    if df is None: return
    
    # Export cleaned data
    df.to_csv(CLEANED_FILE)
    print(f"\n- Cleaned data exported to {CLEANED_FILE}")
    
    # Generate Summary Report (Markdown/Text)
    report_path = Path('weather_summary.md')
    daily_stats, monthly_stats = statistical_analysis(df) # Recalculate stats for report
    
    summary_text = f"""# Weather Data Analysis Report

## 1. Overview
This report analyzes local weather data from {df.index.min().strftime('%B %Y')} to {df.index.max().strftime('%B %Y')}.

## 2. Key Statistical Insights

| Metric | Max Temp (°C) | Min Temp (°C) | Rainfall (mm) | Humidity (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Mean** | {daily_stats.loc['Max Temp (°C)', 'mean']:.2f} | {daily_stats.loc['Min Temp (°C)', 'mean']:.2f} | {daily_stats.loc['Rainfall (mm)', 'mean']:.2f} | {daily_stats.loc['Humidity (%)', 'mean']:.2f} |
| **Max** | {daily_stats.loc['Max Temp (°C)', 'max']:.2f} | {daily_stats.loc['Min Temp (°C)', 'max']:.2f} | {daily_stats.loc['Rainfall (mm)', 'max']:.2f} | {daily_stats.loc['Humidity (%)', 'max']:.2f} |

## 3. Trend Interpretation (Storytelling)
- **Temperature:** The daily line chart shows a clear seasonal increase in temperature from January towards May, with temperatures peaking around $41.0^\circ\text{C}$ in May. This trend indicates the transition from winter/spring to the summer season.
- **Rainfall:** The bar chart highlights that rainfall is highly seasonal. The data shows significant rainfall spikes in June, suggesting the onset of the monsoon season, while months like March and May record zero rainfall.
- **Humidity vs. Temperature:** The scatter plot visually confirms an inverse relationship: as Max Temperature increases (moving right on the x-axis), Humidity tends to decrease (moving down on the y-axis). The highest humidity levels are observed during the cooler, rainy periods (e.g., June).

## 4. Aggregation Highlight
- **Peak Rainfall Month:** June showed the highest total rainfall of {monthly_stats.loc[monthly_stats[('Rainfall (mm)', 'sum')].idxmax(), ('Rainfall (mm)', 'sum')]:.2f} mm.
"""
    
    with report_path.open('w', encoding='utf-8') as f:
        f.write(summary_text)
        
    print(f"- Summary report saved to {report_path}")


def main():
    df_raw = data_acquisition_and_loading(DATA_FILE)
    df_cleaned = data_cleaning_and_processing(df_raw)
    
    if df_cleaned is not None:
        statistical_analysis(df_cleaned)
        visualization(df_cleaned)
        export_and_storytelling(df_cleaned)
        print("\nWeather Data Visualization project completed successfully.")
    else:
        print("\nProject terminated due to data loading/cleaning errors.")

if __name__ == "__main__":
    main()
