import numpy as np
from controller import player_controller
from evoman.environment import Environment
import os
import matplotlib.pyplot as plt

#Loads an individual, plays against all 8 enemies and calculates gains
def get_gains_from_file(env, subfolder, filename):
    
    #Open the file and read the content
    filepath = os.path.join(subfolder, filename)
    with open(filepath, 'r') as file:
        data = file.read()

    #Use eval() to convert the string to an array
    solution = eval(data)
    solution = np.array(solution)

    #Play the game with individual
    scores = env.play(pcont=solution)

    #Calculate gain
    #gain = player_energy - enemy_energy
    gain = scores[1] - scores[2]

    return gain

#Create a boxplot of the gains for each run
def create_boxplot(data1, data2, filename, enemygroup):
    # Create a boxplot
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.boxplot([data1, data2])
    plt.title(f'Gains for enemygroup {enemygroup}')
    plt.ylabel('Gains')
    plt.xlabel('EA1 compared to EA2')
    plt.grid(False)
    plt.savefig(filename, format='png', dpi=300)
    plt.close()

#Get the gains for all 10 runs of one experiment
def get_gains_for_experiment(env, num_runs, experiment):
    gains = []

    #loop through best individual for each run
    for i in range(num_runs):
        #make the correct filename
        subfolder = str(experiment) + "_winners"
        filename = str(experiment) + "_winner_run" + str(i) + ".txt"

        #calculate gains for individual
        gains.append(get_gains_from_file(env, subfolder, filename))
    
    return gains
      
experiment_name = 'run_winners'
enemygroup = [1, 2, 3, 4, 5, 6, 7, 8]
num_runs=10

#Initialize the Evoman environment
env = Environment(
    experiment_name=experiment_name,
    enemies=enemygroup,
    multiplemode='yes',
    playermode="ai",
    player_controller=player_controller(_n_hidden=10),
    enemymode="static",
    level=2,
    logs="off", 
    savelogs="no", 
    speed="fastest",
    visuals=False
    )

#Calculate the gains for each experiment
gains_EA1_EG1 = get_gains_for_experiment(env, num_runs, "EA1_EG1")
gains_EA1_EG2 = get_gains_for_experiment(env, num_runs, "EA1_EG2")
gains_EA2_EG1 = get_gains_for_experiment(env, num_runs, "EA2_EG1")
gains_EA2_EG2 = get_gains_for_experiment(env, num_runs, "EA2_EG2")

#Create a subfolder to store plots
subfolder = 'plots'
if not os.path.exists(subfolder):
            os.makedirs(subfolder)

#Create the boxplots
create_boxplot(gains_EA1_EG1, gains_EA2_EG1, subfolder + '/EG1.png', 1)
create_boxplot(gains_EA1_EG2, gains_EA2_EG2, subfolder + '/EG2.png', 2)