"""
Traveling Salesman Problem - Genetic Algorithm Solution
Eamonn Kearney
May 2020
"""


import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt


class City:
    # Constructor
    def __init__(self, id, name, x, y, connected):
        self.id = id
        self.name = name
        self.x = x
        self.y = y
        self.connected = connected # List of connected cities

    # Calculates the distance between two cities using the Pythagorean theorem.
    def distance(self, city):
        xDis = abs(self.x - city.x)  # Abs returns absolute value
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance


class Fitness:
    # Constructor
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    # Determining the accumulated distance of a route
    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range (0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                # This ensures it ends where it began.
                if i + 1 < len(self.route):
                    toCity = self.route[i+1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    # Determining the fitness of a route
    def routeFitness(self):

        # Check for invalid connections
        invalidConnections = 0
        for i in range(len(self.route)-1):
            currentCity = self.route[i]
            nextCity = self.route[i+1]

            if nextCity.id in currentCity.connected:
                invalidConnections=+1


        if self.fitness == 0:
            if invalidConnections > 0:
                self.fitness = 1 / ( float(self.routeDistance()) * invalidConnections )
            else:
                self.fitness = 1 / (float(self.routeDistance()))
        return self.fitness

# Creating an individual route (random order of cities)
# Route through known connected cities
# Seeding
def createRoute(cityList):
    return random.sample(cityList, len(cityList))

# Creating the first generation (Looping through prev.)
def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

# Determining fitness
# Sort the population in order of fitness using dictionary
def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()

    # Sort routes in order of fitness
    sortedRoutes = sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

    return sortedRoutes

# Elitism ensures the best solutions are passed to the next gen.
# Selection function returns a list of route IDs, which are used
# to create the mating pool in the matingPool function.

# Selecting mating pool.
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns= ["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break;
    return selectionResults

# Making the mating pool - Extracting selected individuals from population
def matingPool(population, selectionResults):
    matingPool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingPool.append(population[index])
    return matingPool

# Breeding
# Creating the next generation, crossover.
# Possibly remove and replace:
# 'Ordered crossover' ensures each city only appears in sequence once.
# Randomly select a subset of first parent and fill remainder of route with
# with genes from second parent in order they appear.
# REPLACE THIS WITH REQUIRED METHOD(s)

def breed(parent1, parent2):

    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

# Apply to generation.
# Uses elitism to retain best routes from current population
# Uses breed function to fill rest of the generation

def breedPopulation(matingPool, eliteSize):
    children = []
    length = len(matingPool) - eliteSize
    pool = random.sample(matingPool, len(matingPool))

    for i in range(0, eliteSize):
        children.append(matingPool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingPool)-i-1])
        children.append(child)
    return children

# Mutate
def mutate(individual):

    mutationSelection = random.randint(0,3)

    if mutationSelection == 0:
        print("Mutation method: None")
    elif mutationSelection == 1:
        print("Mutation method: Swap two genes")
        swapped = int(random.random() * len(individual))
        swapWith = int(random.random() * len(individual))

        city1 = individual[swapped]
        city2 = individual[swapWith]

        individual[swapped] = city2
        individual[swapWith] = city1

    elif mutationSelection == 2:
        print("Mutation method: Swap two strings")

        geneSelection = random.randint(0, len(individual)) # Pivot gene - centre of swap
        # Split list into two lists
        rightGenes = individual[geneSelection:]
        leftGenes = individual[:geneSelection]

        # Rebuild list with switched sub-lists
        individual = []
        for i in range(len(rightGenes)):
            individual.append(rightGenes[i])
        for i in range(len(leftGenes)):
            individual.append(leftGenes[i])

    elif mutationSelection == 3:
        print("Mutation method: Moved to ends")

        # Pick a gene and length
        geneSelection = random.randint(0, len(individual))
        lengthSelection = random.randint(0, int(len(individual)/2))
        directionSelection = random.randint(0,1)

        # Slice the list
        # Move through list, find selected gene.
        # move along line by random number
        # Return the contents of end gene
        endGene = 0
        for i in range(len(individual)):
            if individual[i] == geneSelection:
                endGene = i + lengthSelection

        movedGenes = individual[geneSelection:endGene]
        rightGenes = individual[geneSelection:]
        leftGenes = individual[:geneSelection]

        if directionSelection == 0:
            # Move to front
            individual = []
            for i in range(len(movedGenes)):
                individual.append(movedGenes[i])
            for i in range(len(leftGenes)):
                individual.append(leftGenes[i])
            for i in range(len(rightGenes)):
                individual.append(rightGenes[i])
        elif directionSelection == 1:
            # Move to back
            individual = []
            for i in range(len(leftGenes)):
                individual.append(leftGenes[i])
            for i in range(len(rightGenes)):
                individual.append(rightGenes[i])
                for i in range(len(movedGenes)):
                    individual.append(movedGenes[i])

    return individual

# Apply to entire population

def mutatePopulation(population):
    mutatedPop = []

    for i in range(0, len(population)):
        mutatedInd = mutate(population[i])
        mutatedPop.append(mutatedInd)
    return mutatedPop

# Creating a new generation
def nextGeneration(currentGen, eliteSize):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    mating = matingPool(currentGen, selectionResults)
    children = breedPopulation(mating, eliteSize)
    nextGeneration = mutatePopulation(children)
    return nextGeneration

# Running everything together
def geneticAlgorithm(population, popSize, eliteSize, generations):
    pop = initialPopulation(popSize, population)

    best = [] # Tracking progress for plot.
    worst = []
    average = []

    # Plotting first generation distance
    best.append(rankRoutes(pop)[0][1])
    worst.append(rankRoutes(pop)[popSize - 1][1])
    routes = rankRoutes(pop)
    fitnessList = []
    sum = 0
    for i in range(len(routes)):
        fitness = (routes)[i][1]
        sum += fitness
        fitnessList.append(fitness)
    # Calculate mean average of the fitness
    average.append(sum / len(fitnessList))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize)

        # Plot one distance (second -> last generations)
        # Best
        best.append(rankRoutes(pop)[0][1])
        print(rankRoutes(pop)[0][1])

        # Worst
        worst.append(rankRoutes(pop)[popSize-1][1])

        # Average
        routes = rankRoutes(pop)
        fitnessList = []
        sum = 0
        for i in range(len(routes)):
            fitness = (routes)[i][1]
            sum+=fitness
            fitnessList.append(fitness)
        # Calculate mean average of the fitness
        average.append(sum/len(fitnessList))

    plt.ylabel('Fitness')
    plt.xlabel('Generation')

    # Print final best route
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]

    bestRouteCities = []
    for i in range(len(bestRoute)):
        bestRouteCities.append(bestRoute[i].name)

    print(bestRouteCities)

    plt.plot(best, label = "Best")
    plt.plot(average, label="Average")
    plt.plot(worst, label = "Worst")

    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.legend(loc="upper left")
    plt.show()


# Run algorithm, pass in data

cityList = []

f = open("C:\\Users\\...", "r")
print(f.readline())

for line in f:

    fields = line.rstrip("]").split("[")
    for i in range(len(fields)):
        if (i == 0):
            fields2 = line.rstrip("\n").split(",")
            print("Field ", i, fields[i])
        else:
            fields2 = line.rstrip("\n").rstrip("]").split(",")
            fields2[4] = fields2[4].strip("[")
            print("Sub Fields ", i, fields[i])

    print(fields2)
    # As the number of connected cities varies, extract them into a separate list to pass into City constructor.
    connectedCities = []
    for i in range(len(fields2)):
        if i > 3:
            connectedCities.append(fields2[i])

    print(connectedCities)

    cityList.append(City(int(fields2[0]), fields2[1], float(fields2[2]), float(fields2[3]), connectedCities))

# User needs to enter number of generations.
generationNumber = int(input("Generations: "))
geneticAlgorithm(population=cityList, popSize=20, eliteSize=10, generations=generationNumber)
