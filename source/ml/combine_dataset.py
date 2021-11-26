from dataset_output_generation import generate_output
import os
import pandas as pd

folder_result = 'results/current'

os.makedirs('output', exist_ok=True)
output_filepath = 'output/id_to_mean_total_time.csv'
input_folder = 'input'

# 1. calculate average total time response for all the experiments
generate_output(folder_result, output_filepath)

# 2. combine all input files (files with cost functions)
df = pd.DataFrame(columns=['algorithm', 'id', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'fog', 'cloud'])
df_in = pd.DataFrame()
input_files = os.listdir(input_folder)

for input_file in input_files:
    temp_df = pd.read_csv(input_folder + '/' + input_file, header=None, names=['algorithm', 'id', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'fog', 'cloud'])
    df_in = df_in.append(temp_df)
    print(input_file)

pd.set_option('display.max_columns', None)
print(df_in)

# 3. read output results
df_out = pd.read_csv(output_filepath)
print(df_out)

# 4. merge input and output by id
df = pd.merge(df_in, df_out, on=['id', 'id'])
print(df)

df.to_csv('dataset_small_1.csv')