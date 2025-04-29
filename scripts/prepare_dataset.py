import pandas as pd
import numpy as np
import os

date_unit = 'datetime64[us]'

mean_cols = [
    'mean_CPU_usage_rate', 'canonical_memory_usage', 'assigned_memory_usage',
    'unmapped_page_cache_memory_usage', 'total_page_cache_memory_usage',
    'mean_disk_I/O_time', 'mean_local_disk_space_used',
    'cycles_per_instruction', 'memory_accesses_per_instruction',
    'sampled_CPU_usage'
]

max_cols = ['maximum_memory_usage',
            'maximum_CPU_usage', 'maximum_disk_IO_time']


def binning(df: pd.DataFrame, time_interval_in_minutes: int) -> pd.DataFrame:
    df[['start_time', 'end_time']] = df[[
        'start_time', 'end_time']].astype(date_unit)

    # Ensure all files use the same binning range
    global_bin_start = df['start_time'].min().floor(
        f'{time_interval_in_minutes}min')
    global_bin_end = df['end_time'].max().ceil(
        f'{time_interval_in_minutes}min')
    time_bins = pd.date_range(
        start=global_bin_start, end=global_bin_end, freq=f'{time_interval_in_minutes}min')

    bin_starts = time_bins[:-1].astype(date_unit).values.astype(np.int64)
    bin_ends = time_bins[1:].astype(date_unit).values.astype(np.int64)

    start_times = df['start_time'].values.astype(np.int64)
    end_times = df['end_time'].values.astype(np.int64)

    overlap_starts = np.maximum(
        start_times[:, np.newaxis], bin_starts[np.newaxis, :])
    overlap_ends = np.minimum(
        end_times[:, np.newaxis], bin_ends[np.newaxis, :])
    overlaps = np.maximum(overlap_ends - overlap_starts, 0)

    bin_duration = time_interval_in_minutes * 60 * 1e6  # microseconds
    overlap_weights = overlaps / bin_duration

    row_sums = overlap_weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Prevent division by zero
    normalized_weights = overlap_weights / row_sums

    df_bins = pd.DataFrame(index=time_bins[:-1])

    for col in mean_cols:
        values = df[col].values[:, np.newaxis]
        df_bins[col] = (normalized_weights * values).sum(axis=0)

    for col in max_cols:
        values = df[col].values[:, np.newaxis]
        mask = normalized_weights > 0
        df_bins[col] = np.nanmax(np.where(mask, values, np.nan), axis=0)

    return df_bins.reset_index(names='time_bins')


if __name__ == '__main__':
    dataset_path = '../datasets/raw'
    output_file_2 = '../datasets/processed/test_binned_sum_2_task_usage.csv'
    output_file_5 = '../datasets/processed/test_binned_sum_5_task_usage.csv'

    final_result_2 = None
    final_result_5 = None

    files = sorted(os.listdir(dataset_path))

    for file in files:
        if not file.endswith('.csv'):
            continue
        print(f'Processing file: {file}')

        df = pd.read_csv(os.path.join(dataset_path, file), header=None, names=[
            "start_time", "end_time", "job_ID", "task_index", "machine_ID",
            "mean_CPU_usage_rate", "canonical_memory_usage", "assigned_memory_usage",
            "unmapped_page_cache_memory_usage", "total_page_cache_memory_usage",
            "maximum_memory_usage", "mean_disk_I/O_time", "mean_local_disk_space_used",
            "maximum_CPU_usage", "maximum_disk_IO_time", "cycles_per_instruction",
            "memory_accesses_per_instruction", "sample_portion", "aggregation_type",
            "sampled_CPU_usage"
        ])
        df.fillna(0, inplace=True)

        binned_2min = binning(df, time_interval_in_minutes=2)
        binned_5min = binning(df, time_interval_in_minutes=5)

        print(f'2-min binned shape: {binned_2min.shape}')
        print(f'5-min binned shape: {binned_5min.shape}')

        if final_result_2 is None:
            final_result_2 = binned_2min
            final_result_5 = binned_5min
        else:
            final_result_2 = pd.concat([final_result_2, binned_2min]).groupby(
                'time_bins').aggregate({
                    **{col: 'sum' for col in mean_cols},
                    **{col: 'max' for col in max_cols}
                }).reset_index()
            final_result_5 = pd.concat([final_result_5, binned_5min]).groupby(
                'time_bins').aggregate({
                    **{col: 'sum' for col in mean_cols},
                    **{col: 'max' for col in max_cols}
                }).reset_index()

        print(f'Final result shape (2-min bins): {final_result_2.shape}')
        print(f'Final result shape (5-min bins): {final_result_5.shape}')

    final_result_2.to_csv(output_file_2, index=False, mode='w')
    final_result_5.to_csv(output_file_5, index=False, mode='w')
