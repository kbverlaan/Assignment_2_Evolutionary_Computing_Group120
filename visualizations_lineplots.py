import pandas as pd
import os
import matplotlib.pyplot as plt

#Takes the txt file with logged mean and max fitness and creates a dataframe
def log_to_dataframe(log_file):
    data = []
    
    # Open the file
    with open(log_file, 'r') as file:
        for line in file:
            parts = line.strip().split(",")
            if len(parts) == 3:
                generation, mean_fitness, max_fitness = parts
                data.append({'Generation': int(generation), 'Mean Fitness': float(mean_fitness), 'Max Fitness': float(max_fitness)})
    
    return pd.DataFrame(data)

# General function to combine fitness runs into a dataframe
def combine_runs_to_dataframe(log_folder, num_runs, fitness_type):
    combined_df = None

    for run_num in range(num_runs):
        log_file = os.path.join(log_folder, f'Island_evolution_run{run_num}.txt')
        df = log_to_dataframe(log_file)
        
        if combined_df is None:
            combined_df = df[['Generation', fitness_type]].rename(columns={fitness_type: f'Run {run_num}'})
        else:
            join_column = df[fitness_type].rename(f'Run {run_num}')
            combined_df = pd.concat([combined_df, join_column], axis=1)

    return combined_df

# Makes a plot for both mean and max fitness
def plot_fitness(maxEG1, meanEG1, maxEG2, meanEG2, filename, experiment):
    df1 = pd.read_csv(maxEG1)
    df2 = pd.read_csv(meanEG1)
    df3 = pd.read_csv(maxEG2)
    df4 = pd.read_csv(meanEG2)
    
    # Calculate the mean fitness across 10 runs for each generation
    for df, col_name in zip([df1, df2, df3, df4], ['Mean Max Fitness', 'Mean Mean Fitness', 'Mean Max Fitness', 'Mean Mean Fitness']):
        df[col_name] = df.loc[:, 'Run 0':'Run 9'].mean(axis=1)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df1['Generation'], df1['Mean Max Fitness'], label='Max fitness EG1', color='blue', marker='o')
    plt.plot(df2['Generation'], df2['Mean Mean Fitness'], label='Mean fitness EG1', color='green', marker='x')
    plt.plot(df3['Generation'], df3['Mean Max Fitness'], label='Max fitness EG2', color='red', marker='o')
    plt.plot(df4['Generation'], df4['Mean Mean Fitness'], label='Mean fitness EG2', color='yellow', marker='x')

    plt.title(f'Fitness Across Generations in 10 runs for {experiment}')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness from 10 Runs')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, format='png', dpi=300)
    plt.close()

# General function to save combined dataframes to CSV
def save_combined_dataframes(folders, num_runs, fitness_type, output_prefix):
    for folder in folders:
        df = combine_runs_to_dataframe(folder, num_runs, fitness_type)
        output_file = os.path.join(folder, f'combined_{output_prefix}_{os.path.basename(folder)}.csv')
        df.to_csv(output_file, index=False)

# Parameters
num_runs = 10
subfolder = 'plots'
if not os.path.exists(subfolder):
    os.makedirs(subfolder)

folders = ['EA1_EG1_logs', 'EA1_EG2_logs', 'EA2_EG1_logs', 'EA2_EG2_logs']

# Combine and save all mean and max fitness data
save_combined_dataframes(folders, num_runs, 'Mean Fitness', 'mean_runs')
save_combined_dataframes(folders, num_runs, 'Max Fitness', 'max_runs')

# Get CSV file paths for plotting
csv_files = {
    'EA1_EG1_mean': 'EA1_EG1_logs/combined_mean_runs_EA1_EG1_logs.csv',
    'EA1_EG2_mean': 'EA1_EG2_logs/combined_mean_runs_EA1_EG2_logs.csv',
    'EA2_EG1_mean': 'EA2_EG1_logs/combined_mean_runs_EA2_EG1_logs.csv',
    'EA2_EG2_mean': 'EA2_EG2_logs/combined_mean_runs_EA2_EG2_logs.csv',
    'EA1_EG1_max': 'EA1_EG1_logs/combined_max_runs_EA1_EG1_logs.csv',
    'EA1_EG2_max': 'EA1_EG2_logs/combined_max_runs_EA1_EG2_logs.csv',
    'EA2_EG1_max': 'EA2_EG1_logs/combined_max_runs_EA2_EG1_logs.csv',
    'EA2_EG2_max': 'EA2_EG2_logs/combined_max_runs_EA2_EG2_logs.csv'
}

# Plotting
plot_fitness(csv_files['EA1_EG1_max'], csv_files['EA1_EG1_mean'], csv_files['EA1_EG2_max'], csv_files['EA1_EG2_mean'], f'{subfolder}/EA1.png', 'EA1')
plot_fitness(csv_files['EA2_EG1_max'], csv_files['EA2_EG1_mean'], csv_files['EA2_EG2_max'], csv_files['EA2_EG2_mean'], f'{subfolder}/EA2.png', 'EA2')