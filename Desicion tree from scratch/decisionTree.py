# CART on the Bank Note dataset
from random import seed
from random import randrange

from createDataset import convertToFloat, getNewCsv, getRangeByResult, getValuesByResult, load_csv
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, measurementData, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, measurementData, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores
 
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset, measurementData):
	left, right = list(), list()
	# measure
	measurementData[0] += 2
	measurementData[2] += 1
	# measure
	for row in dataset:
		if row[index] < value:
			left.append(row)
			# measure
			measurementData[0] += 1
			measurementData[2] += 1
			# measure
		else:
			right.append(row)
			# measure
			measurementData[0] += 1
			measurementData[2] += 1
			# measure
		# measure
		measurementData[0] += 1
		measurementData[1] += 1
		measurementData[2] += 2
		# measure
	# measure
	measurementData[0] += 1
	measurementData[2] += 1
		# measure
	return left, right
 
# Calculate the Gini index for a split dataset
def gini_index(groups, classes, measurementData):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	# measure
	measurementData[0] += 2
	measurementData[2] += 2
	# measure
	for group in groups:
		size = float(len(group))
		# measure
		measurementData[0] += 2
		measurementData[2] += 2
		# measure
		# avoid divide by zero
		if size == 0:
			# measure
			measurementData[0] += 1
			measurementData[1] += 1
			measurementData[2] += 2
			# measure
			continue
		score = 0.0
		# measure
		measurementData[0] += 1
		measurementData[2] += 1
		# measure

		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
			# measure
			measurementData[0] += 3
			measurementData[2] += 3
			# measure
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
		# measure
		measurementData[0] += 1
		measurementData[2] += 1
		# measure
	# measure
	measurementData[0] += 1
	measurementData[2] += 1
	# measure
	return gini
 
# Select the best split point for a dataset
def get_split(dataset, measurementData):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	# mesurement
	measurementData[0] += 5
	measurementData[2] += 2
	# mesurement
	for index in range(len(dataset[0])-1):
		# mesurement
		measurementData[0] += 1
		measurementData[2] += 1
		# mesurement
		for row in dataset:
			groups = test_split(index, row[index], dataset, measurementData)
			gini = gini_index(groups, class_values, measurementData)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
				# mesurement
				measurementData[0] += 4
				measurementData[2] += 1
				# mesurement
			# mesurement
			measurementData[0] += 3
			measurementData[1] += 1
			measurementData[2] += 4
			# mesurement
	# mesurement
	measurementData[0] += 1
	measurementData[2] += 1
	# mesurement
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
# Create a terminal node value
def to_terminal(group, measurementData):
	outcomes = [row[-1] for row in group]
	# measure
	measurementData[0] += 2
	measurementData[2] += 2
	# measure
	return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth, measurementData):
	left, right = node['groups']
	del(node['groups'])
	# mesurement
	measurementData[0] += 2
	measurementData[2] += 2
	# mesurement
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right, measurementData)
		# mesurement
		measurementData[0] += 3
		measurementData[1] += 2
		measurementData[2] += 3
		# mesurement
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left, measurementData), to_terminal(right, measurementData)
		# mesurement
		measurementData[0] += 3
		measurementData[1] += 1
		measurementData[2] += 3
		# mesurement
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left, measurementData)
		# mesurement
		measurementData[0] += 1
		measurementData[1] += 1
		measurementData[2] += 2
		# mesurement
	else:
		node['left'] = get_split(left, measurementData)
		split(node['left'], max_depth, min_size, depth+1, measurementData)
		# mesurement
		measurementData[0] += 1
		measurementData[2] += 2
		# mesurement
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right, measurementData)
		# mesurement
		measurementData[0] += 1
		measurementData[1] += 1
		measurementData[2] += 2
		# mesurement
	else:
		node['right'] = get_split(right, measurementData)
		split(node['right'], max_depth, min_size, depth+1, measurementData)
		# mesurement
		measurementData[0] += 1
		measurementData[2] += 2
		# mesurement
 
# Build a decision tree
def build_tree(train, max_depth, min_size, measurementData):
	root = get_split(train, measurementData)
	split(root, max_depth, min_size, 1, measurementData)
	# mesurement
	measurementData[0] += 2
	measurementData[2] += 3
	# mesurement
	return root
 
# Make a prediction with a decision tree
def predict(node, row, measurementData):
	if row[node['index']] < node['value']:
		# measure
		measurementData[1] += 1
		measurementData[2] += 1
		# measure
		if isinstance(node['left'], dict):
			# measure
			measurementData[0] += 1
			measurementData[1] += 1
			measurementData[2] += 2
			# measure
			return predict(node['left'], row, measurementData)
		else:
			# measure
			measurementData[0] += 1
			measurementData[2] += 1
			# measure
			return node['left']
	else:
		if isinstance(node['right'], dict):
			# measure
			measurementData[0] += 1
			measurementData[1] += 1
			measurementData[2] += 2
			# measure
			return predict(node['right'], row, measurementData)
		else:
			# measure
			measurementData[0] += 1
			measurementData[2] += 1
			# measure
			return node['right']

# Classification and Regression Tree Algorithm that returns the assigns and comparisons
# assigns, comparisons, executed lines
def decision_tree(train, test, measurementData, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size, measurementData)
	predictions = list()
	# mesurement
	measurementData[0] += 2
	measurementData[2] += 2
	# mesurement
	for row in test:
		prediction = predict(tree, row, measurementData)
		predictions.append(prediction)
		# mesurement
		measurementData[0] += 3
		measurementData[2] += 3
		# mesurement
	# mesurement
	measurementData[0] += 1
	measurementData[2] += 1
	# mesurement
	return (predictions)

# Test CART on Bank Note dataset
seed(1)

########################################################################################
# this is an annex that creates a newDataset from the ranges of the original
########################################################################################

# load original dataset
originalFilename = 'data_banknote_authentication'
originalDataset = load_csv(originalFilename)
# convert string attributes to integers
convertToFloat(originalDataset)

# get new csv using ranges of values from original
valuesMatrix = getValuesByResult(originalDataset)
rangeMatrix = getRangeByResult(valuesMatrix)
newFilename = 'new_data_banknote_authentication'
getNewCsv(rangeMatrix, newFilename, 500)

# load new data set
newDataset = load_csv(newFilename)

# convert string attributes to integers
convertToFloat(newDataset)

########################################################################################

# evaluate algorithm
n_folds = 5
max_depth = 5
min_size = 10
measurementData = [0, 0, 0]
scores = evaluate_algorithm(newDataset, decision_tree, n_folds, measurementData, max_depth, min_size)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
print(f'assigns: {measurementData[0]} / comparisons: {measurementData[1]} / executed lines of code: {measurementData[2]}')