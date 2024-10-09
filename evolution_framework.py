import random
import numpy as np
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
#Takes a larger population a selects pop_size from it
def select_individuals(pop, scores):
    
    #Select pop_size individuals from pop through certain mechanism
    
    #random for testing framework
    new_pop = []
    for i in range(len(pop)):
        index = random.randint(0, len(pop)-1) 
        individual = pop[index]
        new_pop.append(individual)

    #random for testing framework
    new_scores = []
    for i in range(len(pop)):
        index = random.randint(0, len(scores)-1) 
        score = scores[index]
        new_scores.append(score)

    #Return the new population with corresponding new scores
    return new_pop, new_scores

#Exchange individuals in between islands
def exchange_individuals(islands, prob_exchange):

    for i in range(len(islands)):
        if np.random.rand() < prob_exchange:

            # Select a random island to exchange with
            # make a list of possible targets
            possible_islands = list(range(len(islands)))
            possible_islands.remove(i)

            #choose the target
            j = np.random.choice(possible_islands)

            # Exchange a random number of individuals with a gaussian
            pop_size = len(islands[i])
            mean = pop_size * 0.1
            std_dev = pop_size * 0.1

            num_to_exchange = int(np.clip(np.round(np.random.normal(mean, std_dev)), 1, pop_size))

            # Randomly select the individuals from population
            if len(islands[i]) > 5:
                individuals_to_exchange = np.random.choice(len(islands[i]), num_to_exchange, replace=False)

                # Extract the individuals from island i to transfer to island j
                individuals = islands[i][individuals_to_exchange]

                # Append the individuals to island j
                islands[j] = np.append(islands[j], individuals, axis=0)

                # Remove the individuals from island i
                islands[i] = np.delete(islands[i], individuals_to_exchange, axis=0)

                # Print output
                if num_to_exchange > 1:
                    print(f'Migration event occured: {num_to_exchange} individuals migrated from island {i} to island {j}.')
                else:
                    print(f'Migration event occured: {num_to_exchange} individual migrated from island {i} to island {j}.')


#evolves a population for a number of generations 
def evolve(pop, nr_children, scores):
    
    #create children
    offspring = create_offspring(nr_children, pop, scores)
    
    #Mutate resulting offspring
    mutate(offspring, mutate_rate)
    
    #Evaluate new candidates
    scores = evaluate(env, offspring)

    #Select individuals for next generation
    pop, scores = select_individuals(offspring, scores)
    
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
prob_exchange = 0.3 #probability of exchanging between islands

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
    islands = np.array_split(pop, nr_islands)
    scores = [[] for _ in range(nr_islands)]

    ### EVOLUTION
    generation = 1
    
    for j in range(generations):
        print(f'---------- GENERATION {generation} ----------')

        # Possibly exchange individuals between islands
        exchange_individuals(islands, prob_exchange)

        # Output formatting
        print("\n{:<10} {:<15} {:<15} {:<10}".format("", "Max Fitness", "Mean Fitness", "Pop Size"))

        #Evolve each island for some generations
        for i, island in enumerate(islands):
            scores[i] = evaluate(env, island)
            
            # Evolve for 1 generation
            island, scores[i] = evolve(island, nr_children, scores[i])
            
            max_fitness = np.array(scores[i]).max()
            mean_fitness = np.array(scores[i]).mean()
            pop_size = len(np.array(scores[i]))

            print("{:<10} {:<15.2f} {:<15.2f} {:<10}".format(f"Island {i}", max_fitness, mean_fitness, pop_size))

        print()
        generation += 1
       


    pop = np.concatenate(islands)



#Select the best individual
if len(pop) == len(scores):
    winner = pop[np.argmax(np.array(scores))]
    winner_score = (np.array(scores)).max()
    print(f'Best individual after evolution scores {winner_score}')
else:
    print('Error: pop and scores not same size')

#Here we have to add all kinds of graph stuff for the report