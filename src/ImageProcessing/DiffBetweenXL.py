import pandas as pd

# Read the first Excel file
df1 = pd.read_excel('../../labels/Copy of 0_0.xlsx')

# Read the second Excel file
df2 = pd.read_excel('output_filename.xlsx')

# Extract column 'a' from both DataFrames
col_a_df1 = df1['a']
col_a_df2 = df2['a']

# Compute the differences between corresponding elements
diff_col_a = col_a_df2 - col_a_df1
average_diff = diff_col_a.mean()
print("Average difference in column A:", average_diff)
# Print just the difference column
print("Difference Column:")
print(diff_col_a)
