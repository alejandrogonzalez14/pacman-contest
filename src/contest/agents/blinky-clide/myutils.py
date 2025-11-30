from game import Actions

def closest_food(approxQagent, pos, food):
    '''
    Returns distance and position of the closest food found from the position of the agent and
    the matrix of food where if a position is true, means that there is a food in that position.
    '''
    dist_food = [float('inf'), (0, 0)]
    for i in range(food.width):
        for j in range(food.height):
            if food[i][j]:
                dist_food = min(dist_food, [approxQagent.get_maze_distance(pos, (i, j)), (i, j)], key=lambda x:x[0])
    return dist_food


def closest_capsule(approxQagent, pos, capsules):
    '''
    Returns distance and position of the closest capsule found from the position of the agent.
    '''
    dist_capsule = float('inf')
    for c in capsules:
        dist_capsule = min(dist_capsule, approxQagent.get_maze_distance(pos, c))
    return dist_capsule


def count_food(food):
    '''
    Returns the number of food left.
    '''
    food_count = 0
    for i in range(food.width):
        for j in range(food.height):
            if food[i][j]:
                food_count += 1
    return food_count