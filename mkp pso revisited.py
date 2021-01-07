import random
#Using reals, such as 1.0 and 0.0

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
                    particle.append(0.0)
                else:
                    randomAddition = random.randint(0.0,1.0)
                    particle.append(randomAddition)

            particleData.append(particle)

    print(particleData[0], particleData[10])

    halfParticle = len(particleData) / n
    halfParticle = int(halfParticle)

    t = 0

    while halfParticle < (n*10):
        print(t, halfParticle)
        dataAdded = [particleData[t], particleData[halfParticle]]
        finalParticleData.append(dataAdded)
        halfParticle = halfParticle + 1
        t = t + 1

    #Output of 1 particle.
    print(finalParticleData[0])

    #Now, I have a particle array, contiaining 10 elements, within those elements are two knapsacks.
    
    
    
    #Constraints to 1 add up dependant on 0,1 in particle.
    #Constraints to 2 add up dependant on 0,1 in particle.
    #Function to remove random values to get the value to the capacities of each knapsack.
    #Calculating fitness based on 1,0's in knapsacks and item weights
    #Calculate total fitness.

main()
