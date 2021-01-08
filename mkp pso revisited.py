import random
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
        
def main():
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
            finalsizesArray.append(sizesArray[z:][:m])
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

    for i in range(0,n):
        for j in range(0,10):
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

    while halfParticle < (n*10):
        dataAdded = [particleData[t], particleData[halfParticle]]
        finalParticleData.append(dataAdded)
        halfParticle = halfParticle + 1
        t = t + 1

    #Output of 1 particle. (Testing)
    #print(finalParticleData[0])
    #print(finalParticleData[0][1][3])

    #Now, I have a particle array, contiaining 10 elements, within those elements are two knapsacks.

    #print(finalParticleData)

    #Calculate original sizes of each particle.
    particleSizeArray = []
    for i in range(0,10):
        particleSizeArray.append(calculating_size(finalsizesArray, finalParticleData[i], n, m))

    #print(particleSizeArray[0])
    
    #Now we have calculated the size of each variable, we need a function to get the sizes to
    #the capacity, if they are above the capacity.

    for i in range(0,10):
        particleSizeArray[i] = meeting_capacity(particleSizeArray[i], finalParticleData[i], n, m, capacitiesArray, finalsizesArray)
        
    #print(particleSizeArray)
    #print(finalParticleData)

    #I now have 10 particles that contain knapsacks filled to there capacities.
    #I will now calculate the fitness of each knapsack and the fitnesses of each particle.
    particleFitnesses = []

    for i in range(0,10):
        particleFitnesses.append(calculating_fitness(finalParticleData[i], n, m, weightsArray))

    #print(particleFitnesses)
    
    #I now have the fitnesses of 10 particles.

main()
