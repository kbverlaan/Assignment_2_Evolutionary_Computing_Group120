import pandas as pd
import os
import matplotlib.pyplot as plt

# Takes the txt file with logged mean and max fitness and creates a dataframe
def log_to_dataframe(log_file):
    # Define the column names explicitly
    columns = ['Generation', 'Mean Fitness', 'Max Fitness', 'Diversity']
    
    data = []
    
    # Open the file
    with open(log_file, 'r') as file:
        for line in file:
            parts = line.strip().split(",")
            if len(parts) == 4:
                generation, mean_fitness, max_fitness, diversity = parts
                data.append({'Generation': int(generation), 'Mean Fitness': float(mean_fitness), 'Max Fitness': float(max_fitness), 'Diversity': float(diversity)})
    
    return pd.DataFrame(data, columns=columns)

def combine_runs_to_dataframe(log_folder, num_runs, fitness_type):
    all_dfs = []

    for run_num in range(num_runs):
        log_file = os.path.join(log_folder, f'Island_evolution_run{run_num}.txt')
        df = log_to_dataframe(log_file)
        
        # Append only the relevant fitness type and Generation
        all_dfs.append(df[['Generation', fitness_type]])

    # Concatenate all the data along rows, ensuring there's only one Generation column
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Ensure that 'Generation' is not duplicated across columns
    combined_df = combined_df.drop_duplicates(subset=['Generation', fitness_type])
    
    # Calculate mean and standard deviation grouped by Generation]
    max_df = combined_df.groupby('Generation').max()
    mean_df = combined_df.groupby('Generation').mean()
    std_df = combined_df.groupby('Generation').std()
    
    # Create a new DataFrame with Generation, mean, and std columns
    result_df = pd.DataFrame({
        'Generation': mean_df.index,
        'Max': max_df[fitness_type],
        'Mean': mean_df[fitness_type],
        'Std': std_df[fitness_type]
    }).reset_index(drop=True)
    
    return result_df

# Makes a plot for both mean and max fitness including standard deviation
# Makes a plot for both mean and max fitness including standard deviation
def plot_fitness(maxEG1, meanEG1, maxEG2, meanEG2, filename, experiment):
    df1 = pd.read_csv(maxEG1)
    df2 = pd.read_csv(meanEG1)
    df3 = pd.read_csv(maxEG2)
    df4 = pd.read_csv(meanEG2)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot Max fitness EG1 with standard deviation
    plt.plot(df1['Generation'], df1['Mean'], label='Max fitness EA1', color='blue', linestyle='-')
    plt.fill_between(df1['Generation'], df1['Mean'] - df1['Std'], df1['Mean'] + df1['Std'], color='blue', alpha=0.2)
    
    # Plot Mean fitness EG1 with standard deviation
    plt.plot(df2['Generation'], df2['Mean'], label='Mean fitness EA1', color='blue', linestyle='--')
    plt.fill_between(df2['Generation'], df2['Mean'] - df2['Std'], df2['Mean'] + df2['Std'], color='blue', alpha=0.2)
    
    # Plot Max fitness EG2 with standard deviation
    plt.plot(df3['Generation'], df3['Mean'], label='Max fitness EA2', color='red', linestyle='-')
    plt.fill_between(df3['Generation'], df3['Mean'] - df3['Std'], df3['Mean'] + df3['Std'], color='red', alpha=0.2)
    
    # Plot Mean fitness EG2 with standard deviation
    plt.plot(df4['Generation'], df4['Mean'], label='Mean fitness EA2', color='red', linestyle='--')
    plt.fill_between(df4['Generation'], df4['Mean'] - df4['Std'], df4['Mean'] + df4['Std'], color='red', alpha=0.2)
    
    plt.title(f'Fitness Across Generations in 10 runs for {experiment}', fontsize=14)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Average Fitness', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, format='png', dpi=300)
    plt.legend(fontsize=12, loc='upper left')  # Adjust size and location
    plt.close()

# Makes a plot for diversity with standard deviation
def plot_diversity(EA1EG1, EA1EG2, EA2EG1, EA2EG2, filename):
    df1 = pd.read_csv(EA1EG1)
    df2 = pd.read_csv(EA1EG2)
    df3 = pd.read_csv(EA2EG1)
    df4 = pd.read_csv(EA2EG2)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot Diversity EA1 EG1 with standard deviation
    plt.plot(df1['Generation'], df1['Mean'], label='Diversity EA1 EG1', color='blue', linestyle='-')
    #plt.fill_between(df1['Generation'], df1['Mean'] - df1['Std'], df1['Mean'] + df1['Std'], color='blue', alpha=0.2)
    
    # Plot Diversity EA1 EG2 with standard deviation
    plt.plot(df2['Generation'], df2['Mean'], label='Diversity EA1 EG2', color='blue', linestyle='--')
    #plt.fill_between(df2['Generation'], df2['Mean'] - df2['Std'], df2['Mean'] + df2['Std'], color='blue', alpha=0.2)
    
    # Plot Diversity EA2 EG1 with standard deviation
    plt.plot(df3['Generation'], df3['Mean'], label='Diversity EA2 EG1', color='red', linestyle='-')
    #plt.fill_between(df3['Generation'], df3['Mean'] - df3['Std'], df3['Mean'] + df3['Std'], color='red', alpha=0.2)
    
    # Plot Diversity EA2 EG2 with standard deviation
    plt.plot(df4['Generation'], df4['Mean'], label='Diversity EA2 EG2', color='red', linestyle='--')
    #plt.fill_between(df4['Generation'], df4['Mean'] - df4['Std'], df4['Mean'] + df4['Std'], color='red', alpha=0.2)

    plt.title(f'Diversity Across Generations in 10 runs', fontsize=14)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Average Diversity', fontsize=14)
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
save_combined_dataframes(folders, num_runs, 'Diversity', 'diversity')

# Get CSV file paths for plotting
csv_files = {
    'EA1_EG1_mean': 'EA1_EG1_logs/combined_mean_runs_EA1_EG1_logs.csv',
    'EA1_EG2_mean': 'EA1_EG2_logs/combined_mean_runs_EA1_EG2_logs.csv',
    'EA2_EG1_mean': 'EA2_EG1_logs/combined_mean_runs_EA2_EG1_logs.csv',
    'EA2_EG2_mean': 'EA2_EG2_logs/combined_mean_runs_EA2_EG2_logs.csv',
    'EA1_EG1_max': 'EA1_EG1_logs/combined_max_runs_EA1_EG1_logs.csv',
    'EA1_EG2_max': 'EA1_EG2_logs/combined_max_runs_EA1_EG2_logs.csv',
    'EA2_EG1_max': 'EA2_EG1_logs/combined_max_runs_EA2_EG1_logs.csv',
    'EA2_EG2_max': 'EA2_EG2_logs/combined_max_runs_EA2_EG2_logs.csv',
    'EA1_EG1_diversity': 'EA1_EG1_logs/combined_diversity_EA1_EG1_logs.csv',
    'EA1_EG2_diversity': 'EA1_EG2_logs/combined_diversity_EA1_EG2_logs.csv',
    'EA2_EG1_diversity': 'EA2_EG1_logs/combined_diversity_EA2_EG1_logs.csv',
    'EA2_EG2_diversity': 'EA2_EG2_logs/combined_diversity_EA2_EG2_logs.csv'
}

# Plotting
plot_fitness(csv_files['EA1_EG1_max'], csv_files['EA1_EG1_mean'], csv_files['EA2_EG1_max'], csv_files['EA2_EG1_mean'], f'{subfolder}/EG1.png', 'EG1')
plot_fitness(csv_files['EA1_EG2_max'], csv_files['EA1_EG2_mean'], csv_files['EA2_EG2_max'], csv_files['EA2_EG2_mean'], f'{subfolder}/EG2.png', 'EG2')
plot_diversity(csv_files['EA1_EG1_diversity'], csv_files['EA1_EG2_diversity'], csv_files['EA2_EG1_diversity'], csv_files['EA2_EG2_diversity'], f'{subfolder}/diversity.png')