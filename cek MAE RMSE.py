import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the datasets
master_data = pd.read_csv("Data Master.csv")
bp_results = pd.read_csv("bp_results.csv")

# Print information about the datasets to understand their structure
print("Master Data columns:", master_data.columns.tolist())
print("BP Results columns:", bp_results.columns.tolist())

# Assuming there's a common identifier column (like 'subject_id')
# Modify this based on your actual column names
common_id_column = 'Subject'  # Replace with actual column name

# Merge datasets based on subject identifier
merged_data = pd.merge(master_data, bp_results, on=common_id_column, how='inner')

# Check if merge was successful
print(f"\nMerged data shape: {merged_data.shape}")
print(f"Number of common subjects: {merged_data.shape[0]}")

# Assuming systolic columns are named 'Sistolik' in master_data and 'predicted_Sistolik' in bp_results
# Modify these based on your actual column names
master_systolic_column = 'SYSTOLIC'  # Replace with actual column name
bp_systolic_column = 'SBP'  # Replace with actual column name

# Calculate MAE and RMSE for systolic blood pressure
mae = mean_absolute_error(merged_data[master_systolic_column], merged_data[bp_systolic_column])
rmse = np.sqrt(mean_squared_error(merged_data[master_systolic_column], merged_data[bp_systolic_column]))

print(f"\nMetrics for Systolic Blood Pressure comparison:")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"RMSE (Root Mean Square Error): {rmse:.4f}")

# Calculate individual errors for each subject
merged_data['absolute_error'] = np.abs(merged_data[master_systolic_column] - merged_data[bp_systolic_column])
merged_data['squared_error'] = (merged_data[master_systolic_column] - merged_data[bp_systolic_column])**2

# Display top 5 subjects with highest error
print("\nTop 5 subjects with highest absolute error:")
print(merged_data.sort_values(by='absolute_error', ascending=False).head(5)[[common_id_column, master_systolic_column, bp_systolic_column, 'absolute_error']])

# Visualize the comparison
plt.figure(figsize=(10, 6))
plt.scatter(merged_data[master_systolic_column], merged_data[bp_systolic_column], alpha=0.6)
plt.plot([min(merged_data[master_systolic_column]), max(merged_data[master_systolic_column])], 
         [min(merged_data[master_systolic_column]), max(merged_data[master_systolic_column])], 
         'r--')
plt.xlabel('Actual Systolic BP (Data Master)')
plt.ylabel('Predicted Systolic BP (bp_results)')
plt.title('Comparison of Systolic Blood Pressure Values')
plt.grid(True, alpha=0.3)

# Add text with metrics
plt.annotate(f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}',
             xy=(0.05, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle='round', fc='white', alpha=0.8),
             verticalalignment='top')

plt.savefig('systolic_comparison.png')
plt.show()

# Create a histogram of errors
plt.figure(figsize=(10, 6))
plt.hist(merged_data['absolute_error'], bins=20, alpha=0.7, color='skyblue')
plt.xlabel('Absolute Error in Systolic BP')
plt.ylabel('Frequency')
plt.title('Distribution of Systolic BP Prediction Errors')
plt.grid(True, alpha=0.3)
plt.savefig('systolic_error_distribution.png')
plt.show()

# Export results to csv
merged_data.to_csv('systolic_comparison_results.csv', index=False)

# Calculate percentage of predictions within different error thresholds
within_5 = (merged_data['absolute_error'] <= 5).mean() * 100
within_10 = (merged_data['absolute_error'] <= 10).mean() * 100
within_15 = (merged_data['absolute_error'] <= 15).mean() * 100

print(f"\nPercentage of predictions within error thresholds:")
print(f"Within 5 mmHg: {within_5:.2f}%")
print(f"Within 10 mmHg: {within_10:.2f}%")
print(f"Within 15 mmHg: {within_15:.2f}%")
