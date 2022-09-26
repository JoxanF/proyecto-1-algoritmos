import random
from csv import reader, writer

# Load a CSV file
def load_csv(filename):
	file = open(f'{filename}.csv', "rt")
	lines = reader(file)
    # dataset: list of lists of all the values
	dataset = list(lines)
	return dataset

# In: dataset, string
# Out: void
# creates a new csv with a given dataset and name
def write_csv(dataset, filename):
    newFile = open(f'{filename}.csv', 'w', newline='')
    csvWriter = writer(newFile)

    # writes each line
    csvWriter.writerows(dataset)
 
# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# In: dataset
# Output: dataset
# converts all the values on the dataset from strings to floats
def convertToFloat(dataset):
    # convert string attributes to integers
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)

# In: dataset
# Out: matrix
# returns a matrix that contains 2 matrixes, 1 corresponds to the 0 result and the other to 1 result
# each matrix contains 4 rows and 1372 columns corresponding to the amount of values

def getValuesByResult(dataset):
    matrix = [
        [[],[],[],[]],
        [[],[],[],[]]
        ]
    for line in dataset:
        if (line[4] == 0):
            matrix[0][0].append( line[0])
            matrix[0][1].append( line[1])
            matrix[0][2].append( line[2])
            matrix[0][3].append( line[3])

        elif (line[4] == 1):
            matrix[1][0].append( line[0])
            matrix[1][1].append( line[1])
            matrix[1][2].append( line[2])
            matrix[1][3].append( line[3])

    return matrix

# In: matrix 
# Out: matrix
# returns a matrix that contains 2 matrixes, 1 corresponds to the 0 result and the other to 1 result
# each matrix contains 4 rows and 2 columns (min and max range of values)

def getRangeByResult(valuesMatrix):
    rangeMatrix = [
                [[],[],[],[]],
                [[],[],[],[]]
            ]
    for i in range(4):
        minimum = min(valuesMatrix[0][i])
        maximum = max(valuesMatrix[0][i])
        rangeMatrix[0][i].append(minimum)
        rangeMatrix[0][i].append(maximum)

    for j in range(4):
        minimum = min(valuesMatrix[1][j])
        maximum = max(valuesMatrix[1][j])
        rangeMatrix[1][j].append(minimum)
        rangeMatrix[1][j].append(maximum)

    return rangeMatrix

# In: float, float
# Out: float
# returns a float between the range of 2 floats
# rounds to 5 decimals
def getRandomFloat(float1, float2):
    return round(random.uniform(float1, float2), 5)

# In: int, int
# Out: int
# returns an int between the range of 2 other ints
def getRandomInt(int1, int2):
    return random.randint(int1, int2)

# In: matrix, int
# Out: dataset
# creates a dataset csv with n new registers, each register has column values
# according to the result 0 or 1
def getNewDataset(rangeMatrix, n):
    dataset = []
    for i in range(n):
        register = []
        result = getRandomInt(0, 1)
        for j in range(4):
            min = rangeMatrix[result][j][0]
            max = rangeMatrix[result][j][1]
            columnValue = getRandomFloat(min, max)
            register.append(columnValue)

        register.append(result)
        dataset.append(register)
    return dataset


        
# In: matrix, name, int
# Out: void
# creates a new csv with n new registers, each register has column values
# according to the result 0 or 1, also it ask for the name of the csv
def getNewCsv(rangeMatrix, name, n):
    newDataset = getNewDataset(rangeMatrix, n)
    write_csv(newDataset, name)
