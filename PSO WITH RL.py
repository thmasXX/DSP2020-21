import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import copy

def retry():
    print("")
    print("Restart program?")
    retry1 = input("Enter Y/N (case-sensitive): ")
    if retry1 == "Y":
        print("")
        mainmenu()
    if retry1 == "N":
        exit()
    else:
        print("Enter appropriate value. (Y/N)")
        retry() 

def calculating_size(sizeArray, particles, n, m):
    #Calculating size of knapsacks.
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
    #Calculating total fitness of a particle.
    totalFitness = 0
    for i in range(0, n):
        fitness = 0
        for j in range(0, m):
            if particle[i][j] == 1:
                fitness = fitness + weights[j]
        totalFitness = totalFitness + fitness
    return totalFitness

def calculating_gBest(particle, optimumvalue, swarmSize, sizeArray, n, m, capacity):
    #Comparing everything in array to find best solution in pBest.
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
    #Compares particles to find best solution.
    pBest = min(particlefit, key=lambda x:abs(x-optimumValue))
    for i in range(0,len(particlefit)):
        if pBest == particlefit[i]:
            pBestIndex = i
    return pBestIndex

def repair(particle, particlesize, n, m, capacity, sizeArray):
    #Repairs particles to put them back within capacity.
    repaired = False
    while repaired == False:
        for i in range(0, n):
            randomRepair = random.randint(0,m-1)
            if particlesize[i] > capacity[i]:
                if sizeArray[i][randomRepair] > 0:
                    particle[0][i][randomRepair] = 0
        checkIfRepaired = calculating_size(sizeArray, particle[0], n, m)
        if checkIfRepaired <= capacity:
            repaired = True
            return particle
    
def pso(population, pBest, gBest, Vmax, Vmin, optimumValue, swarmSize, n, m, sizeArray, capacity, weights, gBestArray):
    #Initialize generations and newGeneration array which replaces population after every generation.
    GENS = 250
    newpopulation = copy.deepcopy(population)

    QLearning = []

    gBestFits = []
    bestFits = []

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

    print("")
    print("Optimum Value:", optimumValue)
    print("")

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
                rate = 0.4
                prob = random.uniform(0.0,100.0)
                
                Interchoice = random.randint(0,n-1)
                InterrandChoice = random.randint(0,m-1) 
                Interchoice2 = random.randint(0,n-1)
                InterrandChoice2 = random.randint(0,m-1)

                Swapchoice = random.randint(0,n-1)
                SwaprandChoice = random.randint(0,m-1)
                Swapchoice2 = random.randint(0,n-1)
                SwaprandChoice2 = random.randint(0,m-1)

                Bitchoice = random.randint(0,n-1)
                BitrandChoice = random.randint(0,m-1)
                
                mutate1 = copy.deepcopy(newpopulation[j])
                mutate2 = copy.deepcopy(newpopulation[j])
                mutate3 = copy.deepcopy(newpopulation[j])

                if prob < (100*rate):
                    #MUTATION 1 - INTERCHANGING MUTATION.
                    interchange = random.randint(0,1)
                    interchange2 = random.randint(0,1)
                    mutate1[0][Interchoice][InterrandChoice] = interchange
                    mutate1[0][Interchoice2][InterrandChoice2] = interchange2

                    #MUTATION 2 - SWAP MUTATION.
                    temp1 = mutate2[0][Swapchoice][SwaprandChoice]
                    mutate2[0][Swapchoice][SwaprandChoice] = mutate2[0][Swapchoice2][SwaprandChoice2]
                    mutate2[0][Swapchoice2][SwaprandChoice2] = temp1

                    #MUTATION 3 - BIT FLIP MUTATION.
                    if mutate3[0][Bitchoice][BitrandChoice] == 1:
                        mutate3[0][Bitchoice][BitrandChoice] = 0
                    else:
                        mutate3[0][Bitchoice][BitrandChoice] = 1   

                #Q-LEARNING ALGORITHM.
                #Update mutations.
                mutate1[1] = calculating_fitness(mutate1[0], n, m, weights)
                mutate2[1] = calculating_fitness(mutate2[0], n, m, weights)
                mutate3[1] = calculating_fitness(mutate3[0], n, m, weights)

                #print(mutate1[1], mutate2[1], mutate3[1], newpopulation[j][1])
                
                #Find best solution out of mutations
                findingBestFitnessMutate = [mutate1[1], mutate2[1], mutate3[1]]
                findingBestFitnessIndex = comparing_particles(findingBestFitnessMutate, optimumValue)
                bestQLearningSol = findingBestFitnessMutate[findingBestFitnessIndex]

                valueMut1 = [mutate1[0], mutate1[1]]
                valueMut2 = [mutate2[0], mutate2[1]]
                valueMut3 = [mutate3[0], mutate3[1]]
                valueNP = [newpopulation[j][0], newpopulation[j][1]]

                #print(bestQLearningSol, newpopulation[j][1])

                #Give it a positive/negative reward based on fitness compared to current global best.
                if bestQLearningSol <= newpopulation[j][1]:
                    reward = 1.0
                else:
                    reward = -1.0

                #Assigning Q-learning values dependant on each mutation.
                if findingBestFitnessIndex == 0:
                    QLvalue = [reward, valueNP, valueMut1]
                elif findingBestFitnessIndex == 1:
                    QLvalue = [reward, valueNP, valueMut2]
                elif findingBestFitnessIndex == 2:
                    QLvalue = [reward, valueNP, valueMut3]

                #Check if any Q-Learning values are within table.
                #Only use value in qLearning table if positive reward.
                if (QLvalue in QLearning) & (reward == 1.0):
                    QLvalueIndex = QLearning.index(QLvalue)
                    #Assign particle Q-Learning already.
                    newpopulation[j][0] = QLearning[QLvalueIndex][2][0]

                #If not within table, add new q-learning to table and assign mutation.
                elif (QLvalue in QLearning) & (reward == -1.0):
                    continue

                else:
                    QLearning.append(QLvalue)
                    newpopulation[j][0] = QLvalue[2][0]
                    
                #Check whether solution meets capacity.
                particlesize = calculating_size(sizeArray, newpopulation[j][0], n, m)
                if particlesize > capacity:
                    newpopulation[j] = repair(newpopulation[j], particlesize, n, m, capacity, sizeArray)
                particlesize = calculating_size(sizeArray, newpopulation[j][0], n, m)
                newFitness = calculating_fitness(newpopulation[j][0], n, m, weights)

                if newFitness >= optimumValue:
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

                    newpopulation[j][1] =  newFitness
                    
                    #If index is the new particle, replace pBest with new particle.
                    if indexpBest == 0:
                        pBest[j] = newpopulation[j]

                    #Particle has a valid solution.
                    solution = 1

                else:
                    newpopulation[j] = copy.deepcopy(population[j])

        #Copy newpopulation = population.
        population = copy.deepcopy(newpopulation)
    
        #Calculate gBest from particle bests.
        gBest = calculating_gBest(pBest, optimumValue, swarmSize, sizeArray, n, m, capacity)
        data = [gBest[0], gBest[1], gBest[4]]
        gBestFits.append(data)

        data2 = copy.deepcopy(gBest[1])
        bestFits.append(data2)

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

        #Output data from each generation.
        print("Generation", i+1)
        print("Best Particle:", gBest[1], gBest[4])
        print("")
    
        #If newpop contains optimum fitness meet condition.
        if gBest[1] == optimumValue:
##            with open('data from single run.csv', mode='a', newline='') as csvFile:
##                csvWriter = csv.writer(csvFile, delimiter=',')
##                for k in range(0, len(bestFits)):
##                    csvWriter.writerow([bestFits[k]])
##                csvWriter.writerow([""])
##                csvFile.close()
            finish(gBest)
    
        #If maxgenerations met, print closest fitness.
        if i == GENS-1:
            #Writes data into file for data set testing.
##            with open('data from single run.csv', mode='a', newline='') as csvFile:
##                csvWriter = csv.writer(csvFile, delimiter=',')
##                for k in range(0, len(bestFits)):
##                    csvWriter.writerow([bestFits[k]])
##                csvWriter.writerow([""])
##                csvFile.close()
            bestSolutionIndex = np.argmin(np.abs(np.array(bestFits)-optimumValue))
            bestSolution = gBestFits[bestSolutionIndex]
            finishGenerations(bestSolution)
            #print(QLearning)
         
def initialization(filename):
    #Read information from text files into variables.
    #Currently only doing for two knapsacks.
    with open(filename,'r') as reader:
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

    #Create particle tuple containing random 0,1 data dependant on how many knapsacks.
    #Creation of 10 different particles containing random data.
    #First 10, is first set of data, next 20 is second set of data, needs to be split and put together.
    #e.g. particleData[0] and particleData[10] go together.

    particleData = []
    finalParticleData = []

    swarmSize = 20

    for j in range(0,swarmSize):
        finalParticle = []
        for i in range(0,n):
            particle = []
                
            for k in range(0,m):     
                if finalsizesArray[i][k] == 0:
                    particle.append(0)
                else:
                    randomAddition = random.randint(0.0,1.0)
                    particle.append(randomAddition)

            finalParticle.append(particle)
            
        finalParticleData.append(finalParticle)

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

    #Randomly generate velocities and positions.
    for i in range(0,swarmSize):
        velocity.append(random.uniform(Vmin, Vmax))
        position.append(random.uniform(0.0, 4.0))

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

def finish(gBest):
    print("Optimum value found!")
    print(gBest)
    retry()

def finishGenerations(bestSolution):
    print("Maximum generations reached!")
    print("Printing best particle...")
    print(bestSolution)
    retry()

def mainmenu():
    print("PSO with Q-Learning algorithm.")
    print("")
    print("Data files are numbered 1-10.")
    dataFileChoice = input("Enter data file number: ")

    if dataFileChoice == '1':
        initialization("data1.txt")
    elif dataFileChoice == '2':
        initialization("data2.txt")
    elif dataFileChoice == '3':
        initialization("data3.txt")
    elif dataFileChoice == '4':
        initialization("data4.txt")
    elif dataFileChoice == '5':
        initialization("data5.txt")
    elif dataFileChoice == '6':
        initialization("data6.txt")
    elif dataFileChoice == '7':
        initialization("data7.txt")
    elif dataFileChoice == '8':
        initialization("data8.txt")
    elif dataFileChoice == '9':
        initialization("data9.txt")
    elif dataFileChoice == '10':
        initialization("data10.txt")
    else:
        print("Enter correct value (1-10).")
        print("")
        mainmenu()

mainmenu()
