import random
        
class Particle:

    def __init__(self, pos, velocity, fit):
        self.pos = pos
        self.velocity = velocity
        self.fit = fit
        self.pBest = [0] * len(pos)

    def sigmoid(self, value):
        return 1.0/(1.0 + math.exp((-1.0 * value))

    def updatePos(self):
        for i in range(0, len(self.position)):
            random1 = random.random()
            if (random1 < self.sigmoid(self.velocity[i])):
                self.pos[i] = 1
            else:
                self.pos[i] = 0

    def updateVelocity(self, gBest):
        r1 = random.random()
        r2 = random.random()
        soc = c1 * r1 * (gBest - self.pos[i])
        cog = c2 * r2 * (self.pBest - self.pos[i])
        self.velocity[i] = (w * self_velocity[i]) + soc + cog

    def updateBestParticle(self, p):
        self.pBest = p

    def updateFitness(self, f):
        self.fit = f

    def getPosition(self):
        reutn self.pos[:]

    def getPBest(self):
        return self.pBest[:]

    def getVelocity(self):
        return self.velocity[:]

    def getParticle(self):
        strPos = (', '.join(str(x) for x in self.pos)
        strVelocity = (', '.join(str(x) for x in self.velocity)
        return ("position: " + strPos + " velocity: " + strVelocity + " fitness: " + str(self.fit))
    
def pso():
    knapsacks = []
    weights = []
    objectWeight = []

    #Taking data from text files and turning into variables.
    with open('data.txt','r') as reader:
        n = reader.readline()
        n = int(n)

        i = 0
        while i < n:
            knapsacks.append(0)
            i = i + 1

        m = reader.readline()
        m = int(m)

        weights = [int(number) for number in reader.readline().split(' ')]
        #print(weights)

        for k in range(0,m):
            data = [k, weights[k]]
            objectWeight.append(data)
        #print(objectWeight)

        capacities = [int(number) for number in reader.readline().split(' ')]
        #print(capacities)

        optimumvalue = reader.readline()
        #print(optimumvalue)
        
    reader.close()

pso()
