import random
import numpy as np
import time
import string
import sys, os
from controller import player_controller
from evoman.environment import Environment
from scipy.spatial.distance import pdist, squareform

#Initiliazes a random population
def initialize_pop(pop_size, nr_weights, min_weight, max_weight):
    #Make a random population
    pop = np.random.uniform(min_weight, max_weight, (pop_size, nr_weights))
    
    return pop

def sort_population_by_fitness(pop, scores):
    """ Sort the population and scores in descending order of fitness. """
    scores = np.array(scores)
    pop = np.array(pop)
    
    # Sort scores in descending order and reorder population accordingly
    sorted_indices = np.argsort(scores)[::-1]
    sorted_pop = pop[sorted_indices]
    sorted_scores = scores[sorted_indices]

    return sorted_pop, sorted_scores


#Initiliazes the environment in which to play the game
def create_env(experiment_name, enemygroup, n_hidden):
    if len(enemygroup) > 1:
        multiplemode = 'yes'
    else:
        multiplemode = 'no'

    env = Environment(
        experiment_name=experiment_name,
        enemies=enemygroup,
        multiplemode=multiplemode,
        playermode="ai",
        player_controller=player_controller(_n_hidden=n_hidden),
        enemymode="static",
        level=2,
        logs="off", 
        savelogs="no", 
        speed="fastest",
        visuals=False
    )

    return env

def calculate_genotypic_diversity(island):
    # Calculate pairwise distances between all individuals
    if len(island) < 2:
        return 0  # Not enough individuals to calculate diversity

    # Flatten the island's genomes and calculate pairwise distances
    pairwise_distances = pdist(island, metric='euclidean')
    
    # Calculate the mean pairwise distance
    mean_distance = np.mean(pairwise_distances)

    return mean_distance

def calculate_genotypic_diversity_combined(islands):
    # Filter out empty or None islands and ensure correct dimensions
    numeric_populations = [
        np.array(island, dtype=np.float64) for island in islands.values() 
        if island is not None and len(island) > 0
    ]

    # Check if the numeric populations are non-empty and have consistent dimensions
    if len(numeric_populations) == 0:
        return 0  # No individuals to calculate diversity
    
    # Ensure all islands have consistent dimensions
    # Here we check if all populations have the same shape along axis 1
    individual_length = numeric_populations[0].shape[1]  # Expected genome size (e.g., 265)
    for population in numeric_populations:
        if population.shape[1] != individual_length:
            raise ValueError(f"Population has inconsistent genome length. Expected {individual_length}, found {population.shape[1]}.")

    # Flatten the list of populations into one combined population
    combined_population = np.vstack(numeric_populations)
    
    # Check if the combined population has at least 2 individuals
    if len(combined_population) < 2:
        return 0  # Not enough individuals to calculate diversity
    
    # Calculate pairwise distances between all individuals in the combined population
    pairwise_distances = pdist(combined_population, metric='euclidean')
    
    # Calculate the mean pairwise distance
    mean_distance = np.mean(pairwise_distances)
    
    # Return the genotypic diversity of the combined population
    return mean_distance

#evaluates each individual in pop by playing the game
def evaluate(env, pop):

    #empty array for the results
    results = np.array([])

    #evaluate each individuals
    for individual in pop:
        #play game for each enemy, results in array of fitness
        fitness, _, _, _ = env.play(pcont=individual)

        #append result for each individual
        results = np.append(results, fitness)
    
    return results

#Selects a parent for reproduction
def parent_selection(pop, scores, tournament_size=3):
    
    #Pick a random parent if we have population of 1 or 2
    if len(pop) < tournament_size:
        tournament_size=1

    # Randomly select individuals for the tournament
    tournament_indices = np.random.choice(len(pop), tournament_size, replace=False)
    tournament_individuals = [pop[i] for i in tournament_indices]
    tournament_scores = [scores[i] for i in tournament_indices]
    
    # Find the individual with the highest fitness in the tournament
    winner_index = np.argmax(tournament_scores)
    winner = tournament_individuals[winner_index]
    
    return winner

def reproduce(parent1, parent2):
    # Generate a random mask with True/False values for crossover
    mask = np.random.rand(len(parent1)) < 0.5

    # Create a child by picking genes from each parent based on the mask
    child = np.where(mask, parent1, parent2)

    return child

def create_offspring(nr_children, pop, scores, tournament_size):
    # Ensure scores are in numpy array format for calculation
    scores = np.array(scores)

    # Create an empty list to store offspring
    offspring = []
    total_children = nr_children * len(pop)
    
    # Generate offspring based on fitness probabilities
    for _ in range(total_children):
        # Select two parents based on the fitness probabilities
        parent1 = parent_selection(pop, scores, tournament_size)
        parent2 = parent_selection(pop, scores, tournament_size)
        
        # Recombine parents to create a child
        child = reproduce(parent1, parent2)
        offspring.append(child)
    
    # Return the population of offspring
    return offspring

#Takes a population and mutates it with a mutation rate
def mutate(pop, mutate_rate):
    for individual in pop:
        if np.random.rand() < mutate_rate:
            # Apply small mutation to each weight in the individual
            mutation = np.random.normal(0, 0.1, individual.shape)
            individual += mutation
    return pop

#Takes a larger population a selects the best of pop_size from it
def select_individuals(pop, scores, pop_size):
    pop = np.array(pop)
    scores = np.array(scores)

    # sort the individuals
    sorted = np.argsort(scores)[::-1]

    # select the best indices
    best = sorted[:pop_size]
    new_pop = pop[best]
    new_scores = scores[best]

    #Return the new population with corresponding new scores
    return new_pop, new_scores

#Selects pop_size individuals from larger population with elitism
def select_individuals_tournament(pop, scores, pop_size, tournament_size=3):
    pop = np.array(pop)
    scores = np.array(scores)

    # Initialize the new population arrays
    new_pop = np.zeros((pop_size, pop.shape[1]))
    new_scores = np.zeros(pop_size)

    # Elitism: Add the best individual directly to the new population
    best_idx = np.argmax(scores)
    new_pop[0] = pop[best_idx]
    new_scores[0] = scores[best_idx]

    # Start from index 1 since the best individual is already added
    for i in range(1, pop_size):
        # Randomly choose tournament_size individuals
        selected_indices = np.random.choice(np.arange(len(pop)), size=tournament_size, replace=False)
        
        # Find the index of the best individual in the tournament
        best_idx = selected_indices[np.argmax(scores[selected_indices])]
        
        # Add the best individual from the tournament to the new population
        new_pop[i] = pop[best_idx]
        new_scores[i] = scores[best_idx]
    
    return new_pop, new_scores

def migration_event(islands, scores, migration_pressures):
    for name, island in islands.items():
        if name not in migration_pressures:
            continue

        migration_pressure = migration_pressures[name]

        # Ensure migration_pressure is a float and proceed with migration logic
        if island is not None and len(island) > 0 and migration_pressure is not None and float(np.random.rand()) < migration_pressure:
            island = np.array(island)
            pop_size = len(island)

            small_uninhabited_prob = 10  
            possible_targets = {}

            for target_name, target_island in islands.items():
                if target_name == name:
                    continue  # Skip the current island
                
                if target_island is None or len(target_island) == 0:
                    possible_targets[target_name] = small_uninhabited_prob
                else:
                    fitness = np.mean(scores[target_name]) if scores[target_name] is not None else small_uninhabited_prob
                    possible_targets[target_name] = fitness

            target_names = list(possible_targets.keys())
            weights = np.array(list(possible_targets.values()), dtype=float)
            total_weight = weights.sum()
            if total_weight > 0:
                weights /= total_weight
                target_name = np.random.choice(target_names, p=weights)
                undiscovered = islands[target_name] is None or len(islands[target_name]) == 0

                min_migration = max(5, int(pop_size * 0.25))

                mean = pop_size * migration_pressure * 3
                sigma = pop_size * 0.3

                num_to_exchange = min(pop_size, max(min_migration, int(np.round(np.random.normal(mean, sigma)))))
                num_to_exchange = min(num_to_exchange, pop_size)

                individuals_to_exchange = np.random.choice(pop_size, num_to_exchange, replace=False)
                individuals = island[individuals_to_exchange]

                # Ensure scores[name] is properly initialized as an array
                if scores[name] is None or len(scores[name]) == 0:
                    scores[name] = np.zeros(pop_size)
                elif np.isscalar(scores[name]) or len(scores[name]) < pop_size:
                    scores[name] = np.full(pop_size, scores[name])

                scores_to_exchange = np.array(scores[name])[individuals_to_exchange]

                if islands[target_name] is None:
                    islands[target_name] = individuals
                    scores[target_name] = scores_to_exchange
                else:
                    if islands[target_name].ndim == 1:
                        islands[target_name] = islands[target_name].reshape(1, -1)
                    if individuals.ndim == 1:
                        individuals = individuals.reshape(1, -1)

                    islands[target_name] = np.concatenate((islands[target_name], individuals), axis=0)
                    scores[target_name] = np.concatenate((scores[target_name], scores_to_exchange), axis=0)

                islands[name] = np.delete(island, individuals_to_exchange, axis=0)
                scores[name] = np.delete(scores[name], individuals_to_exchange, axis=0)

                if len(islands[name]) == 0:
                    islands[name] = None
                    scores[name] = None

                if undiscovered:
                    print(f'- Island Discovered: {num_to_exchange} individuals from {name} discovered {target_name}.')
                else:
                    print(f'- Migration: {num_to_exchange} individuals migrated from {name} to {target_name}.')
            
             # Sort the source island after migration
            if islands[name] is not None and scores[name] is not None:
                islands[name], scores[name] = sort_population_by_fitness(islands[name], scores[name])

#evolves a population for a number of generations 
def evolve(env, pop, nr_children, scores, pop_size, tournament_size, mutate_rate):
    
    #create children
    offspring = create_offspring(nr_children, pop, scores, tournament_size)
    
    #Mutate resulting offspring
    mutated_offspring = mutate(offspring, mutate_rate)
    
    #Evaluate new candidates
    offspring_scores = evaluate(env, mutated_offspring)

    #Simply combine the old pop and offspring
    combined_pop = np.concatenate((pop, offspring))
    combined_scores = np.concatenate((scores, offspring_scores))

    # Sort combined population and scores by fitness
    combined_pop, combined_scores = sort_population_by_fitness(combined_pop, combined_scores)

    #Select individuals for next generation
    pop, scores = select_individuals_tournament(combined_pop, combined_scores, pop_size)
    
    #returns the evolved population and new scores
    return pop, scores
    
def logislands(scores, log_file, j, islands):
    all_scores = []
    for score_list in scores.values():
        if score_list is not None:  # Ensure the island has some scores
            all_scores.extend(score_list)

    mean_fitness = np.mean(all_scores)
    max_fitness = np.max(all_scores)
    diversity = calculate_genotypic_diversity_combined(islands)

    with open(log_file, "a") as log:
        log.write(f"{j},{mean_fitness},{max_fitness},{diversity}\n")

def findwinner(islands, scores):
    best_score = -np.inf  # Start with a very low score
    winner = None

    for island_name, score_list in scores.items():
        if score_list is None or len(score_list) == 0:
            continue  # Skip islands with no population or scores

        island_population = islands[island_name]

        if island_population is None or len(island_population) == 0:
            continue  # Skip empty islands

        # Find the best individual in the current island
        island_best_idx = np.argmax(score_list)
        island_best_score = score_list[island_best_idx]

        if island_best_score > best_score:
            best_score = island_best_score
            winner = island_population[island_best_idx]  # Update the winner with the best individual
        
        f, _, _, _ = env.play(pcont=winner)

    return winner, f


if __name__ == "__main__":
    n_runs = 1 #number of runs (should be 10 for report)
    generations = 150 #number of generations
    total_pop_size = 100 #population size

    n_hidden = 10 #number of hidden nodes in NN
    inputs = 265 #amount of weights
    min_weight = -1 #minimum weight for the NN
    max_weight = 1 #maximum weight for the NN

    experiment_name = 'final'
    enemygroup1 = [1, 5, 6] #which enemies to train on (if you want quick, do less)
    enemygroup2 = [2, 5, 8]
    egroups = [enemygroup1, enemygroup2]

    mutate_rate = 0.3 #amount of mutations
    nr_children = 3 #amount off offspring to generate
    tournament_size = 3

    #island params
    nr_islands = 8 #the number of islands
    inhabited_islands = 5

    #migration parameters
    base_migration_prob = 0.02 #probability of exchanging between islands
    pop_weight = 2
    stag_weight = 5
    diversity_weight = 3
    end_weight = 5

    generation_times = [] #record the generation times

    for g, group in enumerate(egroups):
        subfolder = f"EA1_EG{g+1}_logs"
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        #Running the experiment for the amount of runs with one group
        for k in range(n_runs):
            k = k+10
            print(f"-----------RUN {k}-----------")
            
            log_file = os.path.join(subfolder, "Island_evolution_run" + str(k) + ".txt")

            ### INITIALIZATION
            #create the environment to play the game
            env = create_env(experiment_name, group, n_hidden)

            #Initialize a random population
            pop = initialize_pop(total_pop_size, inputs, min_weight, max_weight)
            # split into the inhabited islands
            split_populations = np.array_split(pop, inhabited_islands)

            scores = evaluate(env, pop)

            mean_fitness = scores.mean()
            max_fitness = scores.max()
            diversity = calculate_genotypic_diversity(pop)

            with open(log_file, "w") as log:
                log.write(f"{0},{mean_fitness},{max_fitness},{diversity}\n")

            #ISLAND METHOD EVOLUTION
            #Divide population into nr_islands equal parts
            island_names = list(string.ascii_uppercase[:nr_islands]) # name islands alphabetically

            # Create an empty dictionary for all islands
            islands = {f'Island {name}': None for name in island_names}

            # Randomly select 4 islands to be inhabited
            inhabited_island_names = random.sample(island_names, inhabited_islands)

            for i, island_name in enumerate(inhabited_island_names):
                islands[f'Island {island_name}'] = split_populations[i]

            scores = {name: evaluate(env, pop) if pop is not None else None for name, pop in islands.items()}

            # track max and mean fitnesses
            max_fitnesses = {}
            mean_fitnesses = {}
            stagnation = {}
            population_max = 0
            population_stagnation = 0
            migration_pressure = base_migration_prob
            migration_pressures = {}

            ### EVOLUTION
            generation = 1

            for j in range(generations):
                #track stats
                start = time.time()
                current_gen_max = None
                prev_pop_max = population_max

                #save_output(generation, group, total_pop_size, num_to_exchange, mp, gd,fitness, genotype)


                print(f'--------------- GENERATION {generation} ---------------\n')

                # Possibly migrate individuals between islands
                migration_event(islands, scores, migration_pressures)

                # Output formatting
                print("\n{:<10} {:<15} {:<15} {:<10} {:<10} {:<10}".format("", "Max Fitness", "Mean Fitness", "Pop Size", "MP", "GD"))

                # Evolve each island and check for extinction in reverse order
                for name, island in islands.items():
                    if island is not None and len(island) > 0:
                        pop_size = len(islands[name])

                        islands[name], scores[name] = evolve(env, island, nr_children, scores[name], pop_size, tournament_size, mutate_rate)

                        # calculate max and mean fitnesses
                        max_fitness = np.array(scores[name]).max()
                        mean_fitness = np.array(scores[name]).mean()

                        # save fitnesses and calculate stagnation
                        if name not in max_fitnesses:
                            max_fitnesses[name] = [max_fitness]
                            mean_fitnesses[name] = [mean_fitness]
                            stagnation[name] = (0, 0)
                        else:
                            max_fitnesses[name].append(max_fitness)
                            mean_fitnesses[name].append(mean_fitness)

                            # Check for stagnation in max fitness
                            max_stag, mean_stag = stagnation[name]
                            if len(max_fitnesses[name]) > 1 and max_fitnesses[name][-2] >= max_fitnesses[name][-1]:
                                max_stag += 1
                            else:
                                max_stag = 0  # Reset if there is no stagnation

                            # Check for stagnation in mean fitness
                            if len(mean_fitnesses[name]) > 1 and mean_fitnesses[name][-2] >= mean_fitnesses[name][-1]:
                                mean_stag += 1
                            else:
                                mean_stag = 0  # Reset if there is no stagnation

                            # Update the stagnation dictionary with the new stagnation values
                            stagnation[name] = (max_stag, mean_stag)


                        diversity = calculate_genotypic_diversity(island)

                        max_stag, mean_stag = stagnation[name]

                        # Calculate the migration pressure
                        deviation = abs(pop_size - (total_pop_size/3))
                        pop_factor = deviation / total_pop_size
                        stag_factor = stag_factor = (max_stag + mean_stag * 2) / generations
                        diversity_factor = 1 - (diversity / 15)

                        end_migration_factor = generation / generations


                        # Include end migration factor with its own weight
                        migration_pressures[name] = base_migration_prob * (pop_weight * pop_factor 
                                                                        + stag_weight * stag_factor 
                                                                        + diversity_weight * diversity_factor
                                                                        + end_weight * end_migration_factor)

                        # Print statement with formatted max fitness and mean fitness including stagnation
                        print("{:<10} {:<15} {:<15} {:<10} {:<10.3f} {:<10.2f}".format(
                            name, 
                            f"{max_fitness:.2f} ({max_stag})", 
                            f"{mean_fitness:.2f} ({mean_stag})", 
                            pop_size,
                            migration_pressures[name],
                            diversity
                        ))

                        # track the max of the population
                        if current_gen_max is None or max_fitness > current_gen_max:
                            current_gen_max = max_fitness

                    else:
                        # Skip extinct islands in stats
                        continue

                print()

                # Logging
                logislands(scores, log_file, generation, islands)

                # Generation counter and time calculation
                generation += 1
                end = time.time()
                gen_time = end - start
                generation_times.append(gen_time)
                mean_generation_time = np.mean(generation_times)

                # Max fitness and stagnation
                # Check for stagnation in population-wide max fitness
                if prev_pop_max is not None and current_gen_max == prev_pop_max:
                    population_stagnation += 1
                else:
                    population_stagnation = 0
                    population_max = current_gen_max

                print(f'Best Overall Fitness: {population_max:.2f} ({population_stagnation})\n')
                print(f'Current generation time: {gen_time:.2f} seconds (Mean: {mean_generation_time:.2f} seconds)\n')

            winner, winner_fitness = findwinner(islands, scores)  # Get both the winner and their fitness

            # Now write only the winner's genome to the log file
            subfolderw = f"EA1_EG{g+1}_winners"
            if not os.path.exists(subfolderw):
                os.makedirs(subfolderw)
            best_individual_file = os.path.join(subfolderw, f"EA1_EG{g+1}_winner_run" + str(k) + ".txt")
            with open(best_individual_file, "w") as best_file:
                best_file.write(f"{winner.tolist()}\n")