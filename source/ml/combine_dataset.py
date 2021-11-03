from generate_output import generate_output

folder_result = 'results/current'
output_filepath = 'output/id_to_mean_total_time.csv'
generate_output(folder_result, output_filepath)

# combine all input files (files with cost functions)
# add header

# merge with output_filepath by id