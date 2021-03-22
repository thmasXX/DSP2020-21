import random
import numpy as np
import matplotlib.pyplot as plt
import csv
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

def calculating_fitness(particle, n, m, weights):
    totalFitness = 0
    for i in range(0, n):
        fitness = 0
        for j in range(0, m):
            if particle[i][j] == 1:
                fitness = fitness + weights[j]
        totalFitness = totalFitness + fitness
    return totalFitness

def calculating_gBest(particle, optimumvalue, swarmSize, sizeArray, n, m, capacity):
    differenceArray = []
    for i in range(0,swarmSize):
        differenceArray.append(particle[i][1])
    gBest1 = min(differenceArray, key=lambda x:abs(x-optimumvalue))
    for i in range(0,len(particle)):
        if gBest1 == differenceArray[i]:
            gBestIndex = i
    gBest1 = particle[gBestIndex]
    if len(gBest1) > 4:
        gBest1[4] = calculating_size(sizeArray, gBest1[0], n, m)
    else:
        gBest1.append(calculating_size(sizeArray, gBest1[0], n, m))
    return gBest1

def comparing_particles(particlefit, optimumValue):
    #print(particlefit)
    #print(optimumValue)
    #particlefitnesses = np.asarray(particlefit)
    #pBest = (np.abs(particlefitnesses - optimumValue)).argmin()
    pBest = min(particlefit, key=lambda x:abs(x-optimumValue))
    for i in range(0,len(particlefit)):
        if pBest == particlefit[i]:
            pBestIndex = i
    #print(pBestIndex)
    return pBestIndex

def comparingbestSolutions(particles, optimal):
    gBest = min(particles, key=lambda x:abs(x-optimal))
    return gBest

def repair(particle, particlesize, n, m, capacity, sizeArray):
    repaired = False
    while repaired == False:
        randomRepair = random.randint(0,m-1)
        if particlesize[0] > capacity[0]:
            particle[0][0][randomRepair] = 0
        if particlesize[1] > capacity[1]:
            particle[0][1][randomRepair] = 0
        checkIfRepaired = calculating_size(sizeArray, particle[0], n, m)
        if checkIfRepaired <= capacity:
            repaired = True
            return particle
    

def pso(population, pBest, gBest, Vmax, Vmin, optimumValue, swarmSize, n, m, sizeArray, capacity, weights, gBestArray):
    #Initialize generations and newGeneration array which replaces population after every generation.
    GENS = 75
    newpopulation = population

    filename = 'best solution from each generation.csv'
    f = open(filename, mode='w+')
    f.close()

    gBestFits = []

    #For creating graphs.
    generationlist = []
    list1 = []
    list2 = []
    list3 = []
    
    #if data is changed within new population during the generation where the solution is not feasible
    #data will replaced with the original "population" data, this population data will be overwritten with the
    #newPopulation data every generation.

    condition = 0
    c0 = 1
    c1 = 2
    c2 = 4

    totalReward = 0
    mutationRewards = [0,0,0]

    print("Optimum Value:", optimumValue)
    print("")

    #print(optimumValue)
    #for t in range(0,swarmSize):
        #print(population[t][1])
    #particle data = 0, fitness = 1, velocity = 2, pos = 3, size = 4

    #Until condition has been met, in every generation, we will loop through particles to get
    #solution to be closer to gBest.
    for i in range(0, GENS):
        for j in range(0,swarmSize):
            solution  = 0
            while solution == 0:
                randvalue = random.randint(10,15)
                
                for k in range(0,randvalue):
                    rand1 = random.randint(0, n-1)
                    rand2 = random.randint(0, m-1)
                    
                    #Change values in particle based on the gBest data.
                    if gBest[0][rand1][rand2] == 0:
                        newpopulation[j][0][rand1][rand2] = 0
                            
                    if gBest[0][rand1][rand2] == 1:
                        newpopulation[j][0][rand1][rand2] = 1

                #----ADDING THIS IN FOR TESTING-----
                #"MUTATING/ALTERING" DATA SO THEREFORE THE GBEST CAN LEARN ALSO.
                rate = 0.2
                prob = random.uniform(0.0,100.0)
                choice = random.randint(0,n-1)
                randChoice = random.randint(0,m-1)
                
                choice2 = random.randint(0,n-1)
                randChoice2 = random.randint(0,m-1)

                mutate1 = newpopulation[j]
                mutate2 = newpopulation[j]
                mutate3 = newpopulation[j]

                if prob < (100*rate):
                    #MUTATION 1 - INTERCHANGING MUTATION.
                    interchange = random.randint(0,1)
                    interchange2 = random.randint(0,1)
                    mutate1[0][choice][randChoice] = interchange
                    mutate1[0][choice2][randChoice2] = interchange2

                    #MUTATION 2 - SWAP MUTATION.
                    mutate2[0][choice][randChoice] = mutate2[0][choice][randChoice2]

                    #MUTATION 3 - BIT FLIP MUTATION.
                    if mutate3[0][choice][randChoice] == 1:
                        mutate3[0][choice][randChoice] = 0
                    else:
                        mutate3[0][choice][randChoice] = 1   

                #ADAPTIVE SELECTION.
                if totalReward < 15:
                    randomMutate = random.randint(0,2)
                    
                    if randomMutate == 0:
                        newpopulation[j] = mutate1
                        mutationRewards[0] = mutationRewards[0] + 1
                        totalReward = totalReward + 1
                        selected = 0
                        
                    if randomMutate == 1:
                        newpopulation[j] = mutate2
                        mutationRewards[1] = mutationRewards[1] + 1
                        totalReward = totalReward + 1
                        selected = 1
                        
                    if randomMutate == 2:
                        newpopulation[j] = mutate3
                        mutationRewards[2] = mutationRewards[2] + 1
                        totalReward = totalReward + 1
                        selected = 2                   

                else:
                    mutationPercentage1 = mutationRewards[0]
                    mutationPercentage2 = mutationRewards[0] + mutationRewards[1]
                    mutationPercentage3 = mutationRewards[0] + mutationRewards[1] + mutationRewards[2]

                    randPercentage = random.randint(1,totalReward)

                    if randPercentage <= mutationPercentage1:
                        newpopulation[j] = mutate1
                        mutationRewards[0] = mutationRewards[0] + 1
                        totalReward = totalReward + 1
                        selected = 0

                    elif (randPercentage <= mutationPercentage2) & (randPercentage > mutationPercentage1):
                        newpopulation[j] = mutate2
                        mutationRewards[1] = mutationRewards[1] + 1
                        totalReward = totalReward + 1
                        selected = 1

                    elif (randPercentage <= mutationPercentage3) & (randPercentage > mutationPercentage2):
                        newpopulation[j] = mutate3
                        mutationRewards[2] = mutationRewards[2] + 1
                        totalReward = totalReward + 1
                        selected = 2

                #Check whether solution meets capacity.
                particlesize = calculating_size(sizeArray, newpopulation[j][0], n, m)
                if particlesize > capacity:
                    newpopulation[j] = repair(newpopulation[j], particlesize, n, m, capacity, sizeArray)
                particlesize = calculating_size(sizeArray, newpopulation[j][0], n, m)
                newFitness = calculating_fitness(newpopulation[j][0], n, m, weights)

                if (newFitness >= optimumValue) & (particlesize <= capacity):
                    #Calculate fitness of new particle.

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
                    
                    #Add velocity to current pos to get new position.
                    newPos = population[j][3] + newVelocity
                    newpopulation[j][3] = newPos
                    
                    #Get index of value closer to optimum value from new particle and pBest
                    comparingArray = [newFitness, pBest[j][1]]
                    indexpBest = comparing_particles(comparingArray, optimumValue)
                    #print(indexpBest)

                    newpopulation[j][1] =  newFitness
                    
                    #If index is the new particle, replace pBest with new particle.
                    if indexpBest == 0:
                        pBest[j] = newpopulation[j]

                    #Particle has a valid solution.
                    solution = 1

                else:
                    newpopulation[j] = population[j]
                    if selected == 0:
                        mutationRewards[0] = mutationRewards[0] - 1
                        totalReward = totalReward - 1
                    elif selected == 1:
                        mutationRewards[1] = mutationRewards[1] - 1
                        totalReward = totalReward - 1
                    elif selected == 2:
                        mutationRewards[2] = mutationRewards[2] - 1
                        totalReward = totalReward - 1

        #Copy newpopulation = population.
        population = newpopulation
    
        #Calculate gBest from particle bests.
        gBest = calculating_gBest(pBest, optimumValue, swarmSize, sizeArray, n, m, capacity)
        data = [gBest[0], gBest[1], gBest[4]]
        with open('best solution from each generation.csv', mode='a') as csvFile:
            csvWriter = csv.writer(csvFile, delimiter=',')
            csvWriter.writerow(data)
            csvFile.close()

        generationlist.append(i+1)
        if mutationRewards[0] == 0:
            list1.append(0)
        elif mutationRewards[1] == 0:
            list2.append(0)
        elif mutationRewards[2] == 0:
            list3.append(0)
        else:
            list1.append((mutationRewards[0] / totalReward) * 100)
            list2.append((mutationRewards[1] / totalReward) * 100)
            list3.append((mutationRewards[2] / totalReward) * 100)

        #maxIndex = len(gBestArray) - 1

        #Output data from each generation.
        print("Generation", i+1)
        print("Best Particle:", gBest[1], gBest[4])
        print("")
    
        #If newpop contains optimum fitness meet condition.
        if optimumValue in population:
            finish(gBest)
    
        #If maxgenerations met, print closest fitness.
        if i == GENS - 1:
            with open('best solution from each generation.csv', mode='r') as csvFile:
                csv_reader = csv.reader(csvFile)
                csvList = list(csv_reader)
                csvFile.close()
            for t in range(0, len(csvList)):
                csvList[t][1] = int(csvList[t][1])
                gBestFits.append(csvList[t][1])
            bestSolutionIndex = np.argmin(np.abs(np.array(gBestFits)-optimumValue))
            bestSolution = csvList[bestSolutionIndex]
            finishGenerations(bestSolution)
            print(totalReward, mutationRewards)

        #Function for graph creating for adaptive selection.
        #if i == GENS-1:
            #graphAdaptiveSelection(generationlist, list1, list2, list3)
         
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

    #n = n.strip('\n')
    #m = m.strip('\n')
    #weights = weights.strip('\n')
    #capacities = capacities.strip('\n')
    #sizes = sizes.strip('\n')
    #optimumValue = optimumValue.strip('\n')

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

    gBest = calculating_gBest(pBest, optimumValue, swarmSize, finalsizesArray, n, m, capacitiesArray)

    gBestArray = []
    gBestArray.append(gBest)

    pso(particleArray, pBest, gBest, Vmax, Vmin, optimumValue, swarmSize, n, m, finalsizesArray, capacitiesArray, weightsArray, gBestArray)

def graphAdaptiveSelection(generationlist, list1, list2, list3):
    plt.plot(generationlist, list1)
    plt.plot(generationlist, list2)
    plt.plot(generationlist, list3)

    plt.xlabel('Generations')
    plt.ylabel('Percentage Mutation Chosen')

    plt.title('Adaptive Selection over Generations')
    plt.show()
    

def finish(gBest):
    print("Optimum value found!")
    print(gBest)
    exit()

def finishGenerations(bestSolution):
    print("Maximum generations reached!")
    print("Printing best particle...")
    print(bestSolution)

initialization()
