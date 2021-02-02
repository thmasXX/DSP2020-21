import random
import numpy as np
#Using reals, such as 1.0 and 0.0

def calculating_size(sizeArray, particles, n, m):
    particleSizeArray = []
    for j in range(0,n):
        totalParticleSize = 0
        for k in range(0,m):
            if particles[j][k] == 1:
                totalParticleSize = totalParticleSize + sizeArray[j][k]
        particleSizeArray.append(totalParticleSize)
    return particleSizeArray

def meeting_capacity(particlesize, particleArray, n, m, capacity, sizeArray):
    while particlesize != capacity:
        #Check if capacity is over or below per particle.
        for i in range(0,n):
            #If above, replace a 1 with 0.
            if particlesize[i] > capacity[i]:
                rand1 = random.randint(0,m-1)
                if particleArray[i][rand1] == 1:
                    particleArray[i][rand1] = 0    
            #If below, replace a 0 with 1.
            if particlesize[i] < capacity[i]:
                rand2 = random.randint(0,m-1)
                if particleArray[i][rand2] == 0:
                    particleArray[i][rand2] = 1
        #Calculate new size.
        particlesize = calculating_size(sizeArray, particleArray, n, m)
    return particlesize
    return particlearray

def calculating_fitness(particle, n, m, weights):
    totalFitness = 0
    for i in range(0, n):
        fitness = 0
        for j in range(0, m):
            if particle[i][j] == 1:
                fitness = fitness + weights[j]
        totalFitness = totalFitness + fitness
    return totalFitness

def calculating_gBest(particle, optimumvalue, swarmSize):
    differenceArray = []
    for i in range(0,swarmSize):
        differenceArray.append(particle[i][1])

    #print(differenceArray)
    differenceArray = np.asarray(differenceArray)
    gBestIndex = (np.abs(differenceArray - optimumvalue)).argmin()
    gBest = particle[gBestIndex]
    return gBest

def comparing_two_particles(particlefit, optimumValue):
    #print(particlefit)
    #print(optimumValue)
    particlefitnesses = np.asarray(particlefit)
    pBestIndex = (np.abs(particlefitnesses - optimumValue)).argmin()
    #print(pBestIndex)
    return pBestIndex

def pso(population, pBest, gBest, Vmax, Vmin, optimumValue, swarmSize, n, m, sizeArray, capacity, weights):
    #Initialize generations and newGeneration array which replaces population after every generation.
    GENS = 50
    newpopulation = population
    #if data is changed within new population during the generation where the solution is not feasible
    #data will replaced with the original "population" data, this population data will be overwritten with the
    #newPopulation data every generation.

    condition = 0
    c0 = 1
    c1 = 2
    c2 = 4

    #print(optimumValue)

    #particle data = 0, fitness = 1, velocity = 2, pos = 3

    #Until condition has been met, in every generation, we will loop through particles to get
    #solution to be closer to gBest.
    for i in range(0, GENS):
        for j in range(0,swarmSize):
            solution  = 0
            while solution == 0:
                #Do n times for each particle.
                randvalue = random.randint(1,2)
                
                for k in range(0,randvalue):
                    rand1 = random.randint(0, n-1)
                    rand2 = random.randint(0, m-1)
                    
                    #Change values in particle based on the gBest data.
                    if gBest[0][rand1][rand2] == 0:
                        if newpopulation[j][0][rand1][rand2] == 1:
                            newpopulation[j][0][rand1][rand2] = 0
                            
                    if gBest[0][rand1][rand2] == 1:
                        if newpopulation[j][0][rand1][rand2] == 0:
                            newpopulation[j][0][rand1][rand2] = 1

                #print(newpopulation[j][0])
                            
                #Check whether solution meets capacity.
                particlesize = calculating_size(sizeArray, newpopulation[j][0], n, m)
                
                if particlesize == capacity:
                    #Calculate fitness of new particle.
                    newFitness = calculating_fitness(newpopulation[j][0], n, m, weights)
                    newpopulation[j][1] =  newFitness

                    #Calculate velocity.
                    random1 = random.uniform(0,1)
                    random2 = random.uniform(0,1)
                    newVelocity = (population[j][2] + c1 * random1 * (pBest[j][2] - population[j][2]) +
                                   c2 * random2 * (gBest[2] - population[j][2]))

                    #Make sure velocity is within range.
                    if newVelocity > Vmax:
                        newVelocity = Vmax
                    if newVelocity < Vmin:
                        newVelocity = Vmin
                    
                    #Update velocity.
                    newpopulation[j][2] = newVelocity
                    
                    #Add velocity to current pos to get new position. - NOT COMPLETE.
                    newPos = population[j][3] + newVelocity
                    newpopulation[j][3] = newPos
                    
                    #Get index of value closer to optimum value from new particle and pBest
                    comparingArray = [newFitness, newpopulation[j][1]]
                    indexpBest = comparing_two_particles(comparingArray, optimumValue)
                    #print(indexpBest)
                    
                    #If index is the new particle, replace pBest with new particle.
                    if indexpBest == 0:
                        pBest[j] = newpopulation[j]

                    #Particle has a valid solution.
                    solution = 1
                    
                else:
                    #Solution does not meet capacity therefore data is restored to original.
                    newpopulation[j] = population[j]

        #Copy newpopulation = population.
        population = newpopulation
    
        #Calculate gBest from newpopulation.
        gBest = calculating_gBest(population, optimumValue, swarmSize)
    
        #Output data from each generation.
        print("Generation", i+1)
        print("Best Particle:", gBest[1])
        print("Optimum Value:", optimumValue)
        print("")
    
        #If newpop contains optimum fitness meet condition.
        if optimumValue in population:
            finish(gBest)
    
        #If maxgenerations met, print closest fitness. - NOT COMPLETE
         
def initialization():
    #Read information from text files into variables.
    #Currently only doing for two knapsacks.
    with open('data.txt','r') as reader:
        n = reader.readline()
        m = reader.readline()
        weights = reader.readline()
        capacities = reader.readline()
        sizes = reader.readline()
        optimumValue = reader.readline()
    reader.close()

    n = int(n)
    m = int(m)
    weightsArray = [int(number) for number in weights.split(' ')]
    capacitiesArray = [int(number) for number in capacities.split(' ')]
    sizesArray = [int(item) for item in sizes.split(' ')]
    finalsizesArray = []

    #Splitting up sizes for each knapsack.
    if len(sizesArray) > m:
        z = 0
        y = m
        while y <= len(sizesArray):
            finalsizesArray.append(sizesArray[z:][:y])
            z = z + m
            y = y + m
            
    optimumValue = int(optimumValue)
    #print(finalsizesArray)

    #Create particle tuple containing random 0,1 data dependant on how many knapsacks.
    #Creation of 10 different particles containing random data.
    #First 10, is first set of data, next 20 is second set of data, needs to be split and put together.
    #e.g. particleData[0] and particleData[10] go together.

    particleData = []
    finalParticleData = []

    swarmSize = 20

    for i in range(0,n):
        for j in range(0,swarmSize):
            particle = []
            finalParticle = []
                
            for k in range(0, m):
                if finalsizesArray[i][k] == 0:
                    particle.append(0)
                else:
                    randomAddition = random.randint(0.0,1.0)
                    particle.append(randomAddition)

            particleData.append(particle)

    #print(particleData[0], particleData[10])

    halfParticle = len(particleData) / n
    halfParticle = int(halfParticle)

    t = 0

    while halfParticle < (n*swarmSize):
        dataAdded = [particleData[t], particleData[halfParticle]]
        finalParticleData.append(dataAdded)
        halfParticle = halfParticle + 1
        t = t + 1

    #Output of 1 particle. (Testing)
    #print(finalParticleData[0])
    #print(finalParticleData[0][1][3])

    #Now, I have a particle array, contiaining 10 elements, within those elements are two knapsacks.

    #Calculate original sizes of each particle.
    particleSizeArray = []
    for i in range(0,swarmSize):
        particleSizeArray.append(calculating_size(finalsizesArray, finalParticleData[i], n, m))
    
    #Now we have calculated the size of each variable, we need a function to get the sizes to
    #the capacity, if they are above the capacity.

    for i in range(0,swarmSize):
        particleSizeArray[i] = meeting_capacity(particleSizeArray[i], finalParticleData[i], n, m, capacitiesArray, finalsizesArray)

    #I now have 10 particles that contain knapsacks filled to there capacities.
    #I will now calculate the fitness of each knapsack and the fitnesses of each particle.
    particleFitnesses = []

    for i in range(0,swarmSize):
        particleFitnesses.append(calculating_fitness(finalParticleData[i], n, m, weightsArray))

    #print(particleFitnesses)
    
    #Creating velocity/position/pBest and gBest.
    velocity = []
    position = []
    pBest = []
    Vmax = 2.0
    Vmin = -2.0

    #Adding all fitnesses as pBest currently due to initialization.
    for i in range(0,swarmSize):
        pBestParticle = [finalParticleData[i], particleFitnesses[i]]
        pBest.append(pBestParticle)

    #print(pBest)
    #Randomly generate velocities and positions.
    for i in range(0,swarmSize):
        velocity.append(random.uniform(Vmin, Vmax))
        position.append(random.uniform(0.0, 4.0))

    #print(velocities)
    #print(positions)

    #Appending all data to one final array for simple use. (Might not use this).
    particleArray = []
    for i in range(0,swarmSize):
        appendingData = [finalParticleData[i], particleFitnesses[i], velocity[i], position[i]]
        particleArray.append(appendingData)

    #Adding all fitnesses as pBest currently due to initialization.
    pBest = particleArray

    gBest = calculating_gBest(pBest, optimumValue, swarmSize)
    #print(gBest[2])

    pso(particleArray, pBest, gBest, Vmax, Vmin, optimumValue, swarmSize, n, m, finalsizesArray, capacitiesArray, weightsArray)
    

def finish(gBest):
    print("Optimum value found!")
    print(gBest)

initialization()
