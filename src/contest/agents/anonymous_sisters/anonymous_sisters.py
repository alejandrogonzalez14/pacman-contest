# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='CrazyAgentAttacker', second='CrazyAgentDefender', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def _init_(self, index, time_for_computing=.1):
        super()._init_(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}




class CrazyAgentAttacker(CaptureAgent):
    
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.total_food_eaten = 0
        self.food_eaten = 0
        self.patrol_target = None
        self.patrol_direction = 1   # 1 = moving up, -1 = moving down
    
    def choose_action(self, game_state):
        if self.red:
            if game_state.get_score() > 0:
                
                return self.choose_action_defend(game_state)
            else:
                
                return self.choose_action_attack(game_state)
        else:
            if game_state.get_score() < 0:
                
                return self.choose_action_defend(game_state)
            else:
                
                return self.choose_action_attack(game_state)


    # Attacking part
    def get_home_positions(self, game_state):
        if self.red:
                home_x_range = range(0, game_state.data.layout.width // 2)
        else:
            home_x_range = range(game_state.data.layout.width // 2, game_state.data.layout.width)
        home_positions = [(x, y) for x in home_x_range for y in range(game_state.data.layout.height)
                    if not game_state.has_wall(x, y)]
        return home_positions
    

    def choose_action_attack(self, game_state):
        """
        Choose the best action using A*
        """

        home_positions = self.get_home_positions(game_state)

        my_pos = game_state.get_agent_position(self.index)
        food_positions = self.get_food(game_state).as_list()
        defending_food_positions = self.get_food_you_are_defending(game_state).as_list()
        sorted_defending_food = sorted(
                defending_food_positions,
                key=lambda f: self.get_maze_distance(my_pos, f)
            )


        # If just got killed, set food eaten to 0
        if self.get_previous_observation() != None:
            if self.get_maze_distance(self.get_previous_observation().get_agent_position(self.index), my_pos) > 1:
                self.food_eaten = 0
            if len(self.get_food(self.get_previous_observation()).as_list()) - len(food_positions) == 1:
                self.food_eaten += 1
                self.total_food_eaten += 1

        enemy_indices = self.get_opponents(game_state)

        # Pair enemies with positions
        enemies = []
        for i in enemy_indices:
            pos = game_state.get_agent_position(i)
            if pos is not None:
                enemies.append((i, pos))

        # If there are enemies, find closest
        if enemies:
            closest_enemy_idx, closest_enemy = min(
                enemies,
                key=lambda item: self.get_maze_distance(my_pos, item[1])
            )

        
        path = None
        # Goal is home if he carries >= 18 food or an enemy is close
        if enemies != [] and self.get_maze_distance(my_pos, closest_enemy) <= 3:
            if  game_state.get_agent_state(closest_enemy_idx).is_pacman == True or game_state.get_agent_state(closest_enemy_idx).scared_timer >= 2:
                goal = closest_enemy
            else:
                goal = min(home_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))
            
            path = self.a_star_attack(game_state, goal)
            if path == []:
                goal = sorted_defending_food[0]
                path = self.a_star_attack(game_state, goal)

            if path != None and (len(path) == 1 or len(path) == 0):
                self.food_eaten = 0
        
        elif self.food_eaten >= 15 or game_state.data.timeleft <= 100:

            goal = min(home_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))
            path = self.a_star_attack(game_state, goal)

            if path == []:
                goal = sorted_defending_food[0]
                path = self.a_star_attack(game_state, goal)

            if path != None and (len(path) == 1 or len(path) == 0):
                self.food_eaten = 0
        
        # otherwise we hunt for the nearest food
        elif food_positions:
            sorted_food = sorted(
                food_positions,
                key=lambda f: self.get_maze_distance(my_pos, f)
            )
            i = 0
            goal = sorted_food[i]
            path = self.a_star_attack(game_state, goal)  
            while path == None and i < len(sorted_food):
                i += 1
                goal = sorted_food[i]
                path = self.a_star_attack(game_state, goal) 
        else:
            goal = min(home_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))
            path = self.a_star_attack(game_state, goal)
            if path != None and (len(path) == 1 or len(path) == 0):
                self.food_eaten = 0

        if not path:
            return 'Stop'

        return path[0]

    def a_star_attack(self, start_state, goal):
        import heapq
        tie_breaker = 0
        frontier = []
        heapq.heappush(frontier, (0.0, tie_breaker, start_state, []))
        expanded = []

        while frontier:
            cost, _, state, path = heapq.heappop(frontier)
            pos = state.get_agent_state(self.index).get_position()

            if pos == goal:
                if cost >= 1000:
                    return None
                else:
                    return path

            if pos in expanded:
                continue

            expanded.append(pos)

            for action in state.get_legal_actions(self.index):
                successor = state.generate_successor(self.index, action)
                successor_state = successor.get_agent_state(self.index)
                new_pos = successor_state.get_position()

                if new_pos in expanded:
                    continue
                
                # closest ghost
                

                g = len(path) + 1
                h = self.get_maze_distance(new_pos, goal)
                f = float(g) + h
                tie_breaker += 1
                heapq.heappush(frontier, (f, tie_breaker, successor, path + [action]))

        return None   
    
    # Defensive Part
    def choose_action_defend(self, game_state):
        """
        Choose the best action using A*
        """

        my_pos = game_state.get_agent_position(self.index)
        enemy_indices = self.get_opponents(game_state)
        enemy_positions = [
            (i, game_state.get_agent_position(i))
            for i in enemy_indices
            if game_state.get_agent_position(i) is not None
        ]

        # center border once
        height = game_state.data.layout.height
        mid_x = game_state.data.layout.width // 2
        my_pos = game_state.get_agent_state(self.index).get_position()

        if self.red:
            border_x = mid_x - 2
        else:
            border_x = mid_x

        border_positions = [
            (border_x, y)
            for y in range(3, height- 3)
            if not game_state.has_wall(border_x, y)
        ]

        # If patrol target not set, start at the lowest border tile
        if self.patrol_target is None:
            self.patrol_target = min(border_positions, key=lambda p: p[1])  # bottom
            self.patrol_direction = 1  # start moving upward

        bottom_border = min(border_positions, key=lambda p: p[1])
        top_border = max(border_positions, key=lambda p: p[1])

        if my_pos == self.patrol_target:
            if self.patrol_direction == 1:
                self.patrol_target = top_border
                self.patrol_direction = -1
            else:
                self.patrol_target = bottom_border
                self.patrol_direction = 1

        center_border = self.patrol_target

        home_positions = self.get_home_positions(game_state)
        flag = False
        is_pacman = False
        if self.red:
            if my_pos[0] > border_x:
                goal = min(home_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))
                flag = True
        else:
            if my_pos[0] < border_x:
                goal = min(home_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))
                flag = True

        if not flag:
            # If no visible enemies - defend center 
            if not enemy_positions:
                goal = center_border
            else:
                # prioritize non-pacman enemies 
                defender_targets = []
                im_scared = game_state.get_agent_state(self.index).scared_timer
                for enemy in enemy_positions:
                    if (im_scared  == 0 and game_state.get_agent_state(enemy[0]).is_pacman) or game_state.get_agent_state(enemy[0]).scared_timer >= 2:
                        defender_targets.append(enemy[1])

                if defender_targets:
                    # chase non-pacman enemies on our side
                    goal = min(defender_targets, key=lambda p: self.get_maze_distance(my_pos, p))
                    is_pacman = False
                else:
                    # all enemies are pacman → chase closest invader
                    goal = min(enemy_positions, key=lambda p: self.get_maze_distance(my_pos, p[1]))[1]
                    is_pacman = True

        path = self.a_star_defend(game_state, goal, is_pacman)

        if not path:
            return random.choice(game_state.get_legal_actions(self.index))

        return path[0]



    def a_star_defend(self, start_state, goal, is_pacman):
        import heapq

        frontier = []
        heapq.heappush(frontier, (0, 0, start_state, []))
        expanded = set()
        tie_breaker = 0


        while frontier:
            cost, _, state, path = heapq.heappop(frontier)
            pos = state.get_agent_state(self.index).get_position()

            if pos == goal:
                return path

            if pos in expanded:
                continue
            
            expanded.add(pos)

            for action in state.get_legal_actions(self.index):
                successor = state.generate_successor(self.index, action)
                new_pos = successor.get_agent_state(self.index).get_position()

                if new_pos in expanded:
                    continue

                # default: go to goal
                h = self.get_maze_distance(new_pos, goal)

                our_side = True
                if self.red:
                    if new_pos[0] > start_state.data.layout.width // 2 - 2:
                        our_side = False
                        h *= 2
                else:
                    if new_pos[0] < start_state.data.layout.width // 2:
                        our_side = False
                        h *= 2

                if is_pacman and our_side:
                    h *= -1  # running away from pacman enemies

                g = len(path) + 1
                f = g + h
                tie_breaker += 1

                heapq.heappush(frontier, (f, tie_breaker, successor, path + [action]))

        return None






class CrazyAgentDefender(CaptureAgent):
    
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.total_food_eaten = 0
        self.food_eaten = 0
        self.patrol_target = None
        self.patrol_direction = 1   # 1 = moving up, -1 = moving down
    
    def choose_action(self, game_state):
        return self.choose_action_defend(game_state)
            


    # Attacking part
    def get_home_positions(self, game_state):
        if self.red:
                home_x_range = range(0, game_state.data.layout.width // 2)
        else:
            home_x_range = range(game_state.data.layout.width // 2, game_state.data.layout.width)
        home_positions = [(x, y) for x in home_x_range for y in range(game_state.data.layout.height)
                    if not game_state.has_wall(x, y)]
        return home_positions
    

    def choose_action_attack(self, game_state):
        """
        Choose the best action using A*
        """

        home_positions = self.get_home_positions(game_state)

        my_pos = game_state.get_agent_position(self.index)
        food_positions = self.get_food(game_state).as_list()
        defending_food_positions = self.get_food_you_are_defending(game_state).as_list()
        sorted_defending_food = sorted(
                defending_food_positions,
                key=lambda f: self.get_maze_distance(my_pos, f)
            )


        # If just got killed, set food eaten to 0
        if self.get_previous_observation() != None:
            if self.get_maze_distance(self.get_previous_observation().get_agent_position(self.index), my_pos) > 1:
                self.food_eaten = 0
            if len(self.get_food(self.get_previous_observation()).as_list()) - len(food_positions) == 1:
                self.food_eaten += 1
                self.total_food_eaten += 1

        enemy_indices = self.get_opponents(game_state)

        # Pair enemies with positions
        enemies = []
        for i in enemy_indices:
            pos = game_state.get_agent_position(i)
            if pos is not None:
                enemies.append((i, pos))

        # If there are enemies, find closest
        if enemies:
            closest_enemy_idx, closest_enemy = min(
                enemies,
                key=lambda item: self.get_maze_distance(my_pos, item[1])
            )

        
        path = None
        # Goal is home if he carries >= 18 food or an enemy is close
        if enemies != [] and self.get_maze_distance(my_pos, closest_enemy) <= 3:
            if  game_state.get_agent_state(closest_enemy_idx).is_pacman == True or game_state.get_agent_state(closest_enemy_idx).scared_timer >= 2:
                goal = closest_enemy
            else:
                goal = min(home_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))
            
            path = self.a_star_attack(game_state, goal)
            if path == []:
                goal = sorted_defending_food[0]
                path = self.a_star_attack(game_state, goal)

            if path != None and (len(path) == 1 or len(path) == 0):
                self.food_eaten = 0
        
        elif self.food_eaten >= 15 or game_state.data.timeleft <= 100:

            goal = min(home_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))
            path = self.a_star_attack(game_state, goal)

            if path == []:
                goal = sorted_defending_food[0]
                path = self.a_star_attack(game_state, goal)

            if path != None and (len(path) == 1 or len(path) == 0):
                self.food_eaten = 0
        
        # otherwise we hunt for the nearest food
        elif food_positions:
            sorted_food = sorted(
                food_positions,
                key=lambda f: self.get_maze_distance(my_pos, f)
            )
            i = 0
            goal = sorted_food[i]
            path = self.a_star_attack(game_state, goal)  
            while path == None and i < len(sorted_food):
                i += 1
                goal = sorted_food[i]
                path = self.a_star_attack(game_state, goal) 
        else:
            goal = min(home_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))
            path = self.a_star_attack(game_state, goal)
            if path != None and (len(path) == 1 or len(path) == 0):
                self.food_eaten = 0

        if not path:
            return 'Stop'

        return path[0]

    def a_star_attack(self, start_state, goal):
        import heapq
        tie_breaker = 0
        frontier = []
        heapq.heappush(frontier, (0.0, tie_breaker, start_state, []))
        expanded = []

        while frontier:
            cost, _, state, path = heapq.heappop(frontier)
            pos = state.get_agent_state(self.index).get_position()

            if pos == goal:
                if cost >= 1000:
                    return None
                else:
                    return path

            if pos in expanded:
                continue

            expanded.append(pos)

            for action in state.get_legal_actions(self.index):
                successor = state.generate_successor(self.index, action)
                successor_state = successor.get_agent_state(self.index)
                new_pos = successor_state.get_position()

                if new_pos in expanded:
                    continue
                
                # closest ghost
                

                g = len(path) + 1
                h = self.get_maze_distance(new_pos, goal)
                f = float(g) + h
                tie_breaker += 1
                heapq.heappush(frontier, (f, tie_breaker, successor, path + [action]))

        return None   
    
    # Defensive Part
    def choose_action_defend(self, game_state):
        """
        Choose the best action using A*
        """

        my_pos = game_state.get_agent_position(self.index)
        enemy_indices = self.get_opponents(game_state)
        enemy_positions = [
            (i, game_state.get_agent_position(i))
            for i in enemy_indices
            if game_state.get_agent_position(i) is not None
        ]

        # center border once
        height = game_state.data.layout.height
        mid_x = game_state.data.layout.width // 2
        my_pos = game_state.get_agent_state(self.index).get_position()

        if self.red:
            border_x = mid_x  - 2
        else:
            border_x = mid_x

        border_positions = [
            (border_x, y)
            for y in range(3, height- 3)
            if not game_state.has_wall(border_x, y)
        ]

        # If patrol target not set, start at the lowest border tile
        if self.patrol_target is None:
            self.patrol_target = min(border_positions, key=lambda p: p[1])  # bottom
            self.patrol_direction = 1  # start moving upward

        bottom_border = min(border_positions, key=lambda p: p[1])
        top_border = max(border_positions, key=lambda p: p[1])

        if my_pos == self.patrol_target:
            if self.patrol_direction == 1:
                self.patrol_target = top_border
                self.patrol_direction = -1
            else:
                self.patrol_target = bottom_border
                self.patrol_direction = 1

        center_border = self.patrol_target

        home_positions = self.get_home_positions(game_state)
        flag = False
        is_pacman = False
        if self.red:
            if my_pos[0] > border_x:
                goal = min(home_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))
                flag = True
        else:
            if my_pos[0] < border_x:
                goal = min(home_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))
                flag = True

        if not flag:
            # If no visible enemies - defend center 
            if not enemy_positions:
                goal = center_border
            else:
                # prioritize non-pacman enemies 
                defender_targets = []
                im_scared = game_state.get_agent_state(self.index).scared_timer
                for enemy in enemy_positions:
                    if (im_scared == 0 and game_state.get_agent_state(enemy[0]).is_pacman) or game_state.get_agent_state(enemy[0]).scared_timer >= 2:
                        defender_targets.append(enemy[1])

                if defender_targets:
                    # chase non-pacman enemies on our side
                    goal = min(defender_targets, key=lambda p: self.get_maze_distance(my_pos, p))
                    is_pacman = False
                else:
                    # all enemies are pacman → chase closest invader
                    goal = min(enemy_positions, key=lambda p: self.get_maze_distance(my_pos, p[1]))[1]
                    is_pacman = True

        path = self.a_star_defend(game_state, goal, is_pacman)

        if not path:
            return random.choice(game_state.get_legal_actions(self.index))

        return path[0]



    def a_star_defend(self, start_state, goal, is_pacman):
        import heapq

        frontier = []
        heapq.heappush(frontier, (0, 0, start_state, []))
        expanded = set()
        tie_breaker = 0


        while frontier:
            cost, _, state, path = heapq.heappop(frontier)
            pos = state.get_agent_state(self.index).get_position()

            if pos == goal:
                return path

            if pos in expanded:
                continue
            
            expanded.add(pos)

            for action in state.get_legal_actions(self.index):
                successor = state.generate_successor(self.index, action)
                new_pos = successor.get_agent_state(self.index).get_position()

                if new_pos in expanded:
                    continue

                # default: go to goal
                h = self.get_maze_distance(new_pos, goal)

                our_side = True
                if self.red:
                    if new_pos[0] > start_state.data.layout.width // 2 - 2:
                        our_side = False
                        h *= 2
                else:
                    if new_pos[0] < start_state.data.layout.width // 2:
                        our_side = False
                        h *= 2

                if is_pacman and our_side:
                    h *= -1  # running away from pacman enemies

                g = len(path) + 1
                f = g + h
                tie_breaker += 1

                heapq.heappush(frontier, (f, tie_breaker, successor, path + [action]))

        return None