import numpy as np
import time
from evolution_framework import create_env, evaluate, evolve, initialize_pop, calculate_genotypic_diversity
import os
    
n_runs = 10 #number of runs (should be 10 for report)
generations = 50 #number of generations
total_pop_size = 150 #population size

n_hidden = 10 #number of hidden nodes in NN
inputs = 265 #amount of weights
min_weight = -1 #minimum weight for the NN
max_weight = 1 #maximum weight for the NN

experiment_name = 'test'
group = [1, 5, 6] #which enemies to train on (if you want quick, do less)
#group = [2, 5, 8]
g = 0
#egroups = [enemygroup1, enemygroup2]

mutate_rate = 0.3 #amount of mutations
nr_children = 3 #amount off offspring to generate
tournament_size = 3

generation_times = [] #record the generation times
max_stagnation = 0 #record the number of generations without max_fitness improvement
mean_stagnation = 0 #record the number of generation without mean_fitness improvement


subfolder = f"EA2_EG{g+1}_logs"
if not os.path.exists(subfolder):
    os.makedirs(subfolder)

#Running the experiment for the amount of runs with one group
for i in range(n_runs):
    # Initialize logging for mean fitness
    log_file = os.path.join(subfolder, "Island_evolution_run" + str(i) + ".txt")
    with open(log_file, "w") as log:
        pass

    print(f"\n------RUN {i}------")

    ### INITIALIZATION
    #create the environment to play the game
    env = create_env(experiment_name, group, n_hidden)
    
    #Initialize a random population
    pop = initialize_pop(total_pop_size, inputs, min_weight, max_weight)

    #Evaluate each individual
    scores = evaluate(env, pop)

    #calculate the genotypic diversity
    diversity = calculate_genotypic_diversity(pop)

    max_fitness = np.array(scores).max()
    mean_fitness = np.array(scores).mean()

    # Output formatting
    print("\n{:<20} {:<20} {:<20} {:<20} {:<20} {:<20}".format("Generation", "Max Fitness", "Mean Fitness", "Stagnation of max", "Stagnation of mean", "Diversity"))

#----EVOLUTION-----
    
    for j in range(generations):
        #track stats
        start = time.time()
        current_gen_max = None
        prev_pop_max = max_fitness

        #Outputting the generation info
        print("{:<20} {:<20} {:<20} {:<20} {:<20} {:20}".format(
                    f"{j}",
                    f"{max_fitness:.2f}", 
                    f"{mean_fitness:.2f}",
                    f"{max_stagnation}",
                    f"{mean_stagnation}",
                    f"{diversity}"
                ))

        with open(log_file, "a") as log:
            log.write(f"{j},{mean_fitness},{max_fitness},{diversity}\n")

        # Evolve the population
        pop, scores = evolve(env, pop, nr_children, scores, total_pop_size, tournament_size, mutate_rate)

        old_max = max_fitness
        old_mean = mean_fitness

        # Calculate the new max and mean fitness
        max_fitness = np.array(scores).max()
        mean_fitness = np.array(scores).mean()
        #calucalte the new genotypic diversity
        diversity = calculate_genotypic_diversity(pop)

        if old_max == max_fitness:
            max_stagnation += 1
        
        if old_mean == max_fitness:
            mean_stagnation += 1
        
        #time calculation
        end = time.time()
        gen_time = end - start
        generation_times.append(gen_time)
        mean_generation_time = np.mean(generation_times)

    #Select the best individual
    if len(pop) == len(scores):

        winner = pop[np.argmax(np.array(scores))]
        winner_score = (np.array(scores)).max()
        print(f'Best individual after evolution scores {winner_score}')

        # Save the best individual to a file after all generations
        subfolderw = f"EA2_EG{g+1}_winners"
        if not os.path.exists(subfolderw):
            os.makedirs(subfolderw)
        best_individual_file = os.path.join(subfolderw, f"EA2_EG{g+1}_winner_run" + str(i) + ".txt")
        with open(best_individual_file, "w") as best_file:
                best_file.write(f"{winner.tolist()}")
    else:
        print('Error: pop and scores not same size')

#Here we have to add all kinds of graph stuff for the report