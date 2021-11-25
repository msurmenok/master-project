from dataset_output_generation import generate_output
import os
import pandas as pd

folder_result = 'results/current'

os.makedirs('output', exist_ok=True)
output_filepath = 'output/id_to_mean_total_time.csv'
input_folder = 'input'

# 1. calculate average total time response for all the experiments
# generate_output(folder_result, output_filepath)

# 2. combine all input files (files with cost functions)
input_files = os.listdir(input_folder)

for input_file in input_files:
    print(input_file)

# add header

# merge with output_filepath by id
