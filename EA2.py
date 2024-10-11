import numpy as np
import time
from evolution_framework import create_env, evaluate, evolve, initialize_pop
    
n_runs = 1 #number of runs (should be 10 for report)
generations = 250 #number of generations
total_pop_size = 100 #population size

n_hidden = 10 #number of hidden nodes in NN
inputs = 265 #amount of weights
min_weight = -1 #minimum weight for the NN
max_weight = 1 #maximum weight for the NN

experiment_name = 'test'
enemygroup = [5, 7] #which enemies to train on (if you want quick, do less)

mutate_rate = 0.4 #amount of mutations
nr_children = 2 #amount off offspring to generate
tournament_size = 3

generation_times = [] #record the generation times
max_stagnation = 0 #record the number of generations without max_fitness improvement
mean_stagnation = 0 #record the number of generation without mean_fitness improvement

#Running the experiment for the amount of runs with one group
for i in range(n_runs):
    ### INITIALIZATION
    #create the environment to play the game
    env = create_env(experiment_name, enemygroup, n_hidden)
    
    #Initialize a random population
    pop = initialize_pop(total_pop_size, inputs, min_weight, max_weight)

    #Evaluate each individual
    scores = evaluate(env, pop)

    max_fitness = np.array(scores).max()
    mean_fitness = np.array(scores).mean()

    # Output formatting
    print("\n{:<20} {:<20} {:<20} {:<20} {:<20}".format("Generation", "Max Fitness", "Mean Fitness", "Stagnation of max", "Stagnation of mean"))

#----EVOLUTION-----
    
    for j in range(generations):
        #track stats
        start = time.time()
        current_gen_max = None
        prev_pop_max = max_fitness

        #Outputting the generation info
        print("{:<20} {:<20} {:<20} {:<20} {:<20}".format(
                    f"{j}",
                    f"{max_fitness:.2f}", 
                    f"{mean_fitness:.2f}",
                    f"{max_stagnation}",
                    f"{mean_stagnation}"
                ))

        # Evolve the population
        pop, scores = evolve(env, pop, nr_children, scores, total_pop_size, tournament_size, mutate_rate)

        old_max = max_fitness
        old_mean = mean_fitness

        # Calculate the new max and mean fitness
        max_fitness = np.array(scores).max()
        mean_fitness = np.array(scores).mean()

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
else:
    print('Error: pop and scores not same size')

#Here we have to add all kinds of graph stuff for the report