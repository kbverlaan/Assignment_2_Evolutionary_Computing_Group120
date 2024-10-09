import random
import numpy as np
import time
import sys
from controller import player_controller
from evoman.environment import Environment

#Initiliazes a random population
def initialize_pop(pop_size, nr_weights, min_weight, max_weight):
    #Make a random population
    pop = np.random.uniform(min_weight, max_weight, (pop_size, nr_weights))
    
    return pop

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

#evaluates each individual in pop by playing the game
def evaluate(env, pop):

    #empty array for the results
    results = np.array([])

    #evaluate each individuals
    for individual in pop:
        #play game for each enemy, results in array of fitness
        scores = env.play(pcont=individual)

        fitness = 0
        #add the fitness for each enemy
        for score in scores:
            fitness = fitness + score
        
        #append result for each individual
        results = np.append(results, fitness)
    
    return results

#TODO
#Selects a parent for reproduction
def parent_selection(pop, scores):
    
    #select a parent via a certain mechanism
    
    #random winner for testing framework
    index = random.randint(0, len(pop)-1) 
    winner = pop[index] 
    
    #return one parent
    return winner

#TODO
#Takes two parents and creates one child
def reproduce(parent1, parent2):
    
    #Create a child via a certain reproduction mechanism

    #random child for testing framework
    child = np.random.uniform(min_weight, max_weight, inputs) 

    #Return the child
    return child

#Creates nr_children from a population
def create_offspring(nr_children, pop, scores):
    #create empty list of offspring
    offspring = []
    total_children = nr_children * len(pop)

    #Produce nr_children children
    for i in range(total_children):
        parent1 = parent_selection(pop, scores) #select parent 1
        parent2 = parent_selection(pop, scores) #select parent 2

        #Recombine pairs of parents, generating offspring
        child = reproduce(parent1, parent2)
        offspring.append(child)
    
    #return the population of offspring
    return offspring

#TODO
#Takes a population and mutates it with a mutation rate
def mutate(pop, mutate_rate):
    
    #mutate the population via a certain mechanism
    new_pop = pop #for testing framework

    #Return the mutated population
    return new_pop

#TODO
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

#Exchange individuals in between islands
def exchange_individuals(islands, prob_exchange):

    for name, island in islands.items():
        if island is not None and len(island) > 0 and np.random.rand() < prob_exchange:

            island = np.array(island)

            # Select a random island to exchange with
            # make a list of possible targets
            possible_targets = [target_name for target_name, target_island in islands.items() 
                                if target_name != name and target_island is not None and len(target_island) > 0]

            if not possible_targets:
                continue  # Skip if no other active islands are available

            #choose the target
            target_name = np.random.choice(possible_targets)

            # Exchange a random number of individuals with a right skewed dist
            pop_size = len(island)
            min_migration = max(1, int(pop_size * 0.1))

            mean = pop_size * 0.1
            sigma = 0.5  # Adjust this for skewness

            num_to_exchange = int(np.round(np.random.lognormal(mean, sigma)))
            num_to_exchange = max(min_migration, min(num_to_exchange, pop_size))

            # Randomly select individuals to migrate
            individuals_to_exchange = np.random.choice(pop_size, num_to_exchange, replace=False)
            individuals = island[individuals_to_exchange]

            # Append the selected individuals to the target island
            islands[target_name] = np.append(islands[target_name], individuals, axis=0)

            # Update the current island by removing migrated individuals
            islands[name] = np.delete(island, individuals_to_exchange, axis=0)
            
            # Print output
            if num_to_exchange > 1:
                print(f'Migration event occured: {num_to_exchange} random individuals migrated from {name} to {target_name}.')
            else:
                print(f'Migration event occured: {num_to_exchange} random individual migrated from {name} to {target_name}.')


#evolves a population for a number of generations 
def evolve(pop, nr_children, scores, pop_size):
    
    #create children
    offspring = create_offspring(nr_children, pop, scores)
    
    #Mutate resulting offspring
    mutated_offspring = mutate(offspring, mutate_rate)
    
    #Evaluate new candidates
    offspring_scores = evaluate(env, mutated_offspring)

    #Simply combine the old pop and offspring
    combined_pop = np.concatenate((pop, offspring))
    combined_scores = np.concatenate((scores, offspring_scores))

    #Select individuals for next generation
    pop, scores = select_individuals(combined_pop, combined_scores, pop_size)
    
    #returns the evolved population and new scores
    return pop, scores
    
n_runs = 1 #number of runs (should be 10 for report)
generations = 50 #number of generations
pop_size = 50 #population size

n_hidden = 10 #number of hidden nodes in NN
inputs = 265 #amount of weights
min_weight = -1 #minimum weight for the NN
max_weight = 1 #maximum weight for the NN

experiment_name = 'test'
enemygroup = [5, 7] #which enemies to train on (if you want quick, do less)

mutate_rate = 0.1 #amount of mutations
nr_children = 3 #amount off offspring to generate

nr_islands = 4 #the number of islands
prob_exchange = 0.05 #probability of exchanging between islands

generation_times = [] #record the generation times

#Running the experiment for the amount of runs with one group
for i in range(n_runs):
    ### INITIALIZATION
    #create the environment to play the game
    env = create_env(experiment_name, enemygroup, n_hidden)
    
    #Initialize a random population
    pop = initialize_pop(pop_size, inputs, min_weight, max_weight)

    #Evaluate each individual
    scores = evaluate(env, pop)

    #ISLAND METHOD EVOLUTION
    #Divide population into nr_islands equal parts
    islands = {f"Island {i}": np.array_split(pop, nr_islands)[i] for i in range(nr_islands)}
    scores = {name: evaluate(env, pop) for name, pop in islands.items()}

    ### EVOLUTION
    generation = 1
    
    for j in range(generations):
        start = time.time()
        print(f'---------- GENERATION {generation} ----------')

        # Possibly exchange individuals between islands
        exchange_individuals(islands, prob_exchange)

        # Check for extinction events
        for name, island in islands.items():
            if island is not None and len(island) == 0:
                # Remove the island and corresponding score
                print(f'Extinction event occurred: the population on {name} has gone extinct.')
                islands[name] = None  

        # Output formatting
        print("\n{:<10} {:<15} {:<15} {:<10}".format("", "Max Fitness", "Mean Fitness", "Pop Size"))

        # Evolve each island and check for extinction in reverse order
        for name, island in islands.items():
            if island is not None and len(island) > 0:
                pop_size = len(islands[name])

                scores[name] = evaluate(env, island)
                islands[name], scores[name] = evolve(island, nr_children, scores[name], pop_size)

                max_fitness = np.array(scores[name]).max()
                mean_fitness = np.array(scores[name]).mean()

                print("{:<10} {:<15.2f} {:<15.2f} {:<10}".format(name, max_fitness, mean_fitness, pop_size))
            else:
                # Skip extinct islands in stats
                continue

        print()

        generation += 1
        end = time.time()
        gen_time = end - start
        generation_times.append(gen_time)
        mean_generation_time = np.mean(generation_times)


        print(f'Current generation time: {gen_time:f.1} seconds (Mean: {mean_generation_time:f.1})\n')



#Select the best individual
if len(pop) == len(scores):
    winner = pop[np.argmax(np.array(scores))]
    winner_score = (np.array(scores)).max()
    print(f'Best individual after evolution scores {winner_score}')
else:
    print('Error: pop and scores not same size')

#Here we have to add all kinds of graph stuff for the report