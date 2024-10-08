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
    env = Environment(
        experiment_name=experiment_name,
        enemies=enemygroup,
        multiplemode="yes",
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

    #Produce nr_children children
    for i in range(nr_children):
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
def select_individuals(pop, scores, pop_size):
    
    #Select pop_size individuals from pop through certain mechanism
    
    #random for testing framework
    new_pop = []
    for i in range(pop_size):
        index = random.randint(0, len(pop)-1) 
        individual = pop[index]
        new_pop.append(individual)

    #random for testing framework
    new_scores = []
    for i in range(pop_size):
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
            j = (i + 1) % len(islands)  # Get the next island in a circular manner
            # Exchange a number of individuals
            num_to_exchange = min(len(islands[i]), len(islands[j])) // 4  # Exchange 25% of the population
            individuals_to_exchange = np.random.choice(len(islands[i]), num_to_exchange, replace=False)
            for idx in individuals_to_exchange:
                islands[i][idx], islands[j][idx] = islands[j][idx], islands[i][idx]  # Swap individuals

#evolves a population for a number of generations 
def evolve(pop, generations, nr_children, scores):
    
    #for amount of generations evolve the population
    for i in range(generations):
        #create children
        offspring = create_offspring(nr_children, pop, scores)
        
        #Mutate resulting offspring
        mutate(offspring, mutate_rate)
        
        #Evaluate new candidates
        scores = evaluate(env, offspring)

        #Select individuals for next generation
        pop, scores = select_individuals(offspring, scores, pop_size)

        print(f"Best individual in generation {i} scores {(np.array(scores)).max()}")
    
    #returns the evolved population and new scores
    return pop, scores
    
n_runs = 1 #number of runs (should be 10 for report)
generations = 7 #number of generations
pop_size = 6 #population size

n_hidden = 10 #number of hidden nodes in NN
inputs = 265 #amount of weights
min_weight = -1 #minimum weight for the NN
max_weight = 1 #maximum weight for the NN

experiment_name = 'test'
enemygroup = [1, 2] #which enemies to train on (if you want quick, do less)

mutate_rate = 0.1 #amount of mutations
nr_children = 3*pop_size #amount off offspring to generate

nr_islands = 3 #the number of islands
nr_island_gen = 2 #the number of generations we train an island seperately
prob_exchange = 0.7 #probability of exchanging between islands

#Running the experiment for the amount of runs with one group
for i in range(n_runs):

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
    
    for j in range(round(generations/nr_island_gen)):
        
        #Exchange individuals between islands
        exchange_individuals(islands, prob_exchange)

        #Evolve each island for some generations
        for i, island in enumerate(islands):
            scores[i] = evaluate(env, island)
            island, scores[i] = evolve(island, nr_island_gen, nr_children, scores[i])
            print(f'Island {i} has best individual with score {(np.array(scores[i])).max()}')
       
    pop = np.concatenate(islands)

    #Select the best individual
if len(pop) == len(scores):
    winner = pop[np.argmax(np.array(scores))]
    winner_score = (np.array(scores)).max()
    print(f'Best individual after evolution scores {winner_score}')
else:
    print('Error: pop and scores not same size')

#Here we have to add all kinds of graph stuff for the report