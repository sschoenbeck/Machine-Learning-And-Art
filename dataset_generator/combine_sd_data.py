import pandas as pd
import os


def main():
    working_dir = os.getcwd()

    print(f'Work directory: {working_dir}')
    timeline_folder = working_dir + "\\timeline_data\\"
    timeline_files = [f for f in os.listdir(timeline_folder) if os.path.isfile(os.path.join(timeline_folder, f))]
    print(f'Found video paths: {timeline_files}')
    print(f'Found video paths: {len(timeline_files)}')

    timeline_files = timeline_files
    sample_df = pd.read_csv(timeline_folder + timeline_files[0])
    column_names = sample_df.columns
    full_timeline_df = pd.DataFrame(columns=column_names)
    i = 0
    for timeline_file in timeline_files:
        i += 1
        if i % 20 == 0:
            print(i)
        timeline_df = pd.read_csv(timeline_folder + timeline_file)
        if len(timeline_df.index) == 1:
            full_timeline_df = full_timeline_df.append(timeline_df)
        else:
            row_index = int(len(timeline_df.index)/2)
            row_df = timeline_df.iloc[[row_index]]
            full_timeline_df = full_timeline_df.append(row_df)
    print(full_timeline_df)
    full_timeline_df.to_csv(f'{working_dir}\\movie_timelines\\SD.csv', index=False)


if __name__ == "__main__":
    main()
