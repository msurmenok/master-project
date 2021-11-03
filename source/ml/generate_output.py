import os
import re
import pandas as pd


def compute_times_df(ldf):
    ldf["time_latency"] = ldf["time_reception"] - ldf["time_emit"]
    ldf["time_wait"] = ldf["time_in"] - ldf["time_reception"]
    ldf["time_service"] = ldf["time_out"] - ldf["time_in"]
    ldf["time_response"] = ldf["time_out"] - ldf["time_reception"]
    ldf["time_total_response"] = ldf["time_response"] + ldf["time_latency"]


def generate_output(folder_result, output_filepath):
    folder_result = folder_result
    output_file = output_filepath

    files_result = os.listdir(folder_result)

    # generate y
    file_averages = open(output_file, 'a+')
    file_averages.write('id,average total time\n')

    for file in files_result:
        pattern = '^Results.*\d.csv$'
        if re.search(pattern, file):
            id = file[8:-4]
            print(file, id)
            df = pd.read_csv(folder_result + "/" + file)
            compute_times_df(df)
            average_total_time = df["time_total_response"].mean()
            file_averages.write("%s,%f\n" % (id, average_total_time))

    file_averages.close()
