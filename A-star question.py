import math
from copy import deepcopy
import timeit


def hamming(current, goal):
	"""
	Calculates the Hamming distance between two game states
	:param current: - Current game state. 2D list with 0 as the empty tile
	:param goal: - Goal state. 2D list with 0 as the empty tile
	:return: - Integer distance
	"""
	distance = 0
	for i in range(len(goal)):
		for j in range(len(goal)):
			if current[i][j] == 0:
				continue
			elif current[i][j] != goal[i][j]:
				distance += 1

	return distance


def manhattan(current, goal):
	"""
	Calculates the manhattan distance of the current game state from the goal
	state
	:param current: - Current game state. 2D list with 0 as the empty tile
	:param goal: - Goal state. 2D list with 0 as the empty tile
	:return: - Integer distance
	"""
	distance = 0
	for n in range(1, len(goal) ** 2):
		for i in range(len(goal)):
			for j in range(len(goal)):
				if current[i][j] == n:
					ic = i
					jc = j
				if goal[i][j] == n:
					ig = i
					jg = j

		distance += abs(ic - ig) + abs(jc - jg)

	return distance


def legalMoves(state, previous):
	"""
	Calculates the list of legal moves that don't result in moves already
	obtained in path
	:param state: The game state
	:param previous: List of all game states used to get to state
	:return: List of all legal moves from state
	"""
	moves = []
	for i in range(len(state)):
		for j in range(len(state)):
			if state[i][j] == 0:
				pos = (i, j)

	if pos[0] != 0:
		new = deepcopy(state)
		new[pos[0] - 1][pos[1]], new[pos[0]][pos[1]] = new[pos[0]][pos[1]], new[pos[0] - 1][pos[1]]
		if new not in previous:
			moves.append(new)

	if pos[0] != len(state) - 1:
		new = deepcopy(state)
		new[pos[0] + 1][pos[1]], new[pos[0]][pos[1]] = new[pos[0]][pos[1]], new[pos[0] + 1][pos[1]]
		if new not in previous:
			moves.append(new)

	if pos[1] != 0:
		new = deepcopy(state)
		new[pos[0]][pos[1] - 1], new[pos[0]][pos[1]] = new[pos[0]][pos[1]], new[pos[0]][pos[1] - 1]
		if new not in previous:
			moves.append(new)

	if pos[1] != len(state) - 1:
		new = deepcopy(state)
		new[pos[0]][pos[1] + 1], new[pos[0]][pos[1]] = new[pos[0]][pos[1]], new[pos[0]][pos[1] + 1]
		if new not in previous:
			moves.append(new)

	return moves


def inversions(state):
	"""
	Calculates the inversions for a game state
	:param state: The game state - 2D list
	:return: The number of inversions
	"""
	list = []
	inversions = 0
	for row in state:
		list += row
	list.remove(0)
	for i in range(len(list) - 1):
		for j in range(i + 1, len(list)):
			if list[i] > list[j]:
				inversions += 1
	return inversions


def aStar(start, goal, heuristic):
	"""
	Uses A* search to find the shortest path from start to the goal state
	:param start: The start state
	:param goal: The goal state
	:param heuristic: Heuristic function to be used
	:return: A list of the optimal set of choices or -1 if the goal state is unobtainable from the start state
	"""

	# If both the start and goal state have odd or even inversions the it is solvable
	# otherwise it is insolvable
	if inversions(start) % 2 != inversions(goal) % 2:
		return -1
	if start == goal:
		return [goal]

	visited = [start]
	considered = [[[start, move], heuristic(move, goal), 1] for move in legalMoves(start, visited)]
	while True:
		considered.sort(key=lambda node: node[1] + node[2])
		path = considered[0][0]
		cost = considered[0][2] + 1
		considered.pop(0)
		if path[-1] == goal:
			return path
		for move in legalMoves(path[-1], visited):
			visited.append(move)
			considered.append([path + [move], heuristic(move, goal), cost])


if __name__ == "__main__":
	start = [[7, 2, 4],
	         [5, 0, 6],
	         [8, 3, 1]]
	goal = [[0, 1, 2],
	        [3, 4, 5],
	        [6, 7, 8]]

	# Allow user to input their own start and goal configurations
	while True:
		inp = input("Would you like to use the default start and goal states? Type Y or N\n")
		if inp.lower() == "y":
			break
		elif inp.lower() == "n":
			while True:
				print("Enter the start configuration separated by commas. For example 7,2,4,5,0,6,8,3,1 for a desired board of:")
				for row in start:
					print(row)
				inp = input()
				user_start = inp.split(',')
				for i in range(len(user_start)):
					user_start[i] = int(user_start[i])
				if 0 not in user_start:
					print("There must be a 0 in the configuration")
					continue
				n = math.sqrt(len(user_start))
				if n % 1 != 0:
					print("The entries must fill a square grid")
					continue
				n = int(n)
				start = [user_start[n*i:(i+1)*n] for i in range(n)]
				break

			while True:
				print("Enter the goal configuration separated by commas. For example 0,1,2,3,4,5,6,7,8 for a desired board of:")
				for row in goal:
					print(row)
				inp = input()
				user_goal = inp.split(',')
				ng = math.sqrt(len(user_goal))
				if n != ng:
					print("There must be the same number of entries as for the start configuration")
					continue
				for i in range(len(user_goal)):
					user_goal[i] = int(user_goal[i])
				goal = [user_goal[n*i:(i+1)*n] for i in range(n)]
				if sorted(user_start) != sorted(user_goal):
					print("start and end states must contain the same values")
					continue
				break
			else:
				continue
			break

	while True:
		inp = input("Enter h to use Hamming distance as the heuristic, m to use Manhattan distance or Q to quit\n")

		if inp.lower() == 'h':
			heuristic = hamming
			break

		elif inp.lower() == 'm':
			heuristic = manhattan
			break

		elif inp.lower() == 'q':
			exit(0)

	path = aStar(start, goal, heuristic)
	if path == -1:
		print("This configuration is not solvable")

	else:
		for node in path:
			for row in node:
				print(row)
			print()
		print("Number of moves needed:", len(path) - 1)

	'''Find the average run time of A* for both heuristic functions'''
	# manhattanTime = timeit.timeit('aStar(start, goal, manhattan)', globals=globals(), number=10)/10
	# print(manhattanTime)
	# hammingTime = timeit.timeit('aStar(start, goal, hamming)', globals=globals(), number=10)/10
	# print(hammingTime)
