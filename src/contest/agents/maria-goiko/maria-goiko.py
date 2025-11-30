# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html


import random
import util
from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point

#we create a class to differentiate roles
class Role:
    OFFENSIVE = "Offensive"
    DEFENSIVE = "Defensive"

#we create a shared state dictionary to hold shared variables
shared_state = {
    'initial_food': 0,  # **MY COMMENT** Tracks initial food count at start of game
    'actual_food': 0,   # **MY COMMENT** Current food count on the map
    'food_eat': 0,      # **MY COMMENT** Counts food eaten by the offensive agent
    'have_pased_pos': (0.0,0.0),  # **MY COMMENT** Position tracking variable
    'first_iteration_back': False,  # **MY COMMENT** Flag for first return to base
    'timer': 0,         # **MY COMMENT** General purpose timer for game events
    'A_point': (0.0,0.0),  # **MY COMMENT** First patrol point for defensive agent
    'B_point': (0.0,0.0),  # **MY COMMENT** Second patrol point for defensive agent
    'food_timer': 0,    # **MY COMMENT** Timer to detect if offensive agent is stuck (no food eaten)
    'roles': {},        # **MY COMMENT** Dictionary mapping agent indices to their current roles
    'need_change': {},  # **MY COMMENT** Dictionary tracking which agents need a role change
    'role_timer': {},   # **MY COMMENT** Dictionary for role-specific timers
}

#function to set roles for both agents
def set_roles(agent1_index, role1, agent2_index, role2):
    shared_state['roles'][agent1_index] = role1
    shared_state['need_change'][agent1_index] = False
    shared_state['role_timer'][agent1_index] = 0
    shared_state['roles'][agent2_index] = role2
    shared_state['need_change'][agent2_index] = False
    shared_state['role_timer'][agent2_index] = 0

#agent role getter
def get_role(agent_index):
    return shared_state['roles'].get(agent_index, None)

#funtion to change roles based on conditions
def change_role(agent_index, agent_instance, n_food_left):
    #we check if the timer is greater than a threshold and if it's defensive to switch roles
    if shared_state['roles'][agent_index] == Role.DEFENSIVE and shared_state['role_timer'][agent_index] >= 98:
        shared_state['roles'][agent_index] = Role.OFFENSIVE
        shared_state['need_change'][agent_index] = False
        shared_state['role_timer'][agent_index] = 0
    #we check if it's offensive to switch back to defensive
    elif shared_state['roles'][agent_index] == Role.OFFENSIVE:
        shared_state['roles'][agent_index] = Role.DEFENSIVE
        shared_state['need_change'][agent_index] = False
        agent_instance.food_obtained(n_food_left)
        shared_state['role_timer'][agent_index] = 0
    else:
        shared_state['role_timer'][agent_index] += 1

#we change the roles of all agents
def change_signal():
    for agent_index in shared_state['need_change']:
        if not shared_state['need_change'][agent_index]:
            shared_state['need_change'][agent_index] = True


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    # Set initial roles and pass shared state
    set_roles(first_index, Role.OFFENSIVE, second_index, Role.DEFENSIVE)
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
        self.index = index

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        #we set the initial food count in shared state
        shared_state['initial_food'] = len(self.get_food(game_state).as_list())


    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
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

    #we reset food tracking variables when agent respawns
    def food_obtained(self, n_food):
        shared_state['initial_food'] = n_food
        shared_state['food_eat'] = 0
        shared_state['food_timer'] = 0

    def manage_food_counter(self, food_left):
        #we reset or update food eaten counter
        if food_left == shared_state['initial_food']:
            shared_state['food_eat'] = 0
            shared_state['actual_food'] = shared_state['initial_food']
        
        #we update actual food count and food eaten counter
        elif food_left < shared_state['actual_food']:
            shared_state['actual_food'] = food_left
            shared_state['food_eat'] += 1

    #we compute distance to safe zone
    def go_safe_zone(self, game_state, action):
        successor = self.get_successor(game_state, action)
        #we get the agent position after the action
        my_pos = successor.get_agent_state(self.index).get_position()
        #distance to safe zone
        min_distance = self.get_maze_distance(my_pos, self.start)
        return min_distance

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
        #we check if a role change is needed
        if shared_state['need_change'].get(self.index, False):
            change_role(self.index, self, len(self.get_food(game_state).as_list()))
        #we evaluate based on current role
        if get_role(self.index) == Role.OFFENSIVE:
            features = self.get_features_offensive(game_state, action)
            weights = self.get_weights_offensive(game_state, action)
        elif get_role(self.index) == Role.DEFENSIVE:
            features = self.get_features_defensive(game_state, action)
            weights = self.get_weights_defensive(game_state, action)
        else:
            features = self.get_features(game_state, action)
            weights = self.get_weights(game_state, action)
        #we return the dot product of features and weights
        return features * weights

    #we separate defensive and offensive features
    def get_features_defensive(self, game_state, action):
        #we initialize a counter
        features = util.Counter()
        #we simulate the successor state after the action
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        #we get the agent's position
        my_pos = my_state.get_position()
        #we value being on defense
        features['on_defense'] = 1
        walls = game_state.get_walls()
        map_width = walls.width
        map_height = walls.height
        #we calculate midline x coordinate
        midline_x = map_width // 2
        #we set patrol points if not already set
        if shared_state['A_point'] == (0.0,0.0):
            if self.red:
                A_point = (midline_x-4,map_height-3)
                B_point = (midline_x-4,2)
            else:
                A_point = (midline_x+1,map_height-3)
                B_point = (midline_x+1,2)
            if walls[A_point[0]][A_point[1]]:
                for i in range(1, A_point[1]):
                    if not walls[A_point[0]][A_point[1]-i]:
                        A_point = (A_point[0],A_point[1]-i)
                        break
            if walls[B_point[0]][B_point[1]]:
                for i in range(1, A_point[1]):
                    if not walls[B_point[0]][B_point[1]+i]:
                        B_point = (B_point[0],B_point[1]+i)
                        break
            shared_state['A_point'] = A_point
            shared_state['B_point'] = B_point
        #detect enemy pacman
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        #we find invaders
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        #if there are invaders, we compute distances
        if len(invaders) > 0:
            #we calcuate the distance to close enemies and get the closest one
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            closest_invader_distance = min(dists)
            features['invader_distance'] = min(dists)
            #if the closest invader is within 5 units, we chase or block it
            if closest_invader_distance <= 5:
                closest_invader = invaders[dists.index(closest_invader_distance)]
                target_pos = closest_invader.get_position()
                #if the invader is on our side, we chase it
                if target_pos[0] > midline_x:
                    if my_pos[0] <= midline_x:
                        features['chasing_invader'] = self.get_maze_distance(my_pos, target_pos)
                    else:
                        features['chasing_invader'] = -100
                else:
                    features['chasing_invader'] = self.get_maze_distance(my_pos, target_pos)
            else:
                features['patrol_distance'] = self.ghost_patrol(my_pos)
        #if no invaders, we patrol
        else:
            features['invader_distance'] = 0
            features['move_towards_invader'] = 0
            features['patrol_distance'] = self.ghost_patrol(my_pos)
        #we penalize stopping and reversing
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        return features

    #we return weights for defensive features
    def get_weights_defensive(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'patrol_distance': -5, 'move_towards_invader': 50, 'patrol_distance': -5, 'stop': -100, 'reverse': -2}

    #we define patrol behavior for defensive agent
    def ghost_patrol(self, my_pos):
        target_pos = getattr(self, "current_target", shared_state['A_point'])
        if my_pos == shared_state['A_point']:
            target_pos = shared_state['B_point']
        elif my_pos == shared_state['B_point']:
            target_pos = shared_state['A_point']
        self.current_target = target_pos
        return self.get_maze_distance(my_pos, target_pos)

    #we define offensive features
    def get_features_offensive(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        #we reward eating food
        features['successor_score'] = -len(food_list)
        #we get agent position
        my_pos = successor.get_agent_state(self.index).get_position()
        theres_ghost = False
        its_stuck = False
        #we check if the agent is stuck
        if shared_state['food_timer'] >= 250:
            theres_ghost = False
            shared_state['food_timer'] = 0
            its_stuck = True
        else:
            shared_state['food_timer'] += 1
        #we detect nearby ghosts
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None]
        #we compute distances to ghosts
        if ghosts:
            ghost_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in ghosts]  # **MY COMMENT** Distance to each ghost
            closest_ghost_distance = min(ghost_dists)
            danger_threshold = 5
            skip_threshold = 2
            #we react based on ghost proximity
            if closest_ghost_distance <= danger_threshold:
                features['ghost_distance'] = closest_ghost_distance
                theres_ghost = True
                features['safe_zone_distance'] = 0
            elif closest_ghost_distance <= skip_threshold:
                safe_zone_distance = self.go_safe_zone(game_state, action)
                features['safe_zone_distance'] = safe_zone_distance
                return features
            else:
                features['ghost_distance'] = 0
                features['safe_zone_distance'] = 0
        else:
            features['ghost_distance'] = 0
        #we manage timers and role changes
        if my_pos == self.start and shared_state['timer'] >= 20:
            change_signal()
        elif shared_state['timer'] < 20:
            shared_state['timer'] += 1
        if my_pos == shared_state['A_point']:
            its_stuck = False
        #we compute the distance to the nearest food
        if len(food_list) > 0:
            if shared_state['food_eat'] >= 1:
                min_distance = self.go_safe_zone(game_state, action)
                features['distance_to_food'] = min_distance
                mid_line = int(game_state.get_walls().width//2)
                if self.red:
                    if mid_line - 2 == my_pos[0]:
                        shared_state['food_eat'] = 0
                else:
                    if mid_line + 2 == my_pos[0]:
                        shared_state['food_eat'] = 0
            #we adjust behavior based on ghost presence and stuck status
            elif theres_ghost and not its_stuck:
                min_distance = max([self.get_maze_distance(my_pos, food) for food in food_list])
                features['distance_to_food'] = min_distance * 10
            elif its_stuck:
                min_distance = self.get_maze_distance(my_pos, shared_state['A_point'])
                features['distance_to_food'] = min_distance * 10
            else:
                min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
                features['distance_to_food'] = min_distance
        self.manage_food_counter(len(food_list))
        return features

    #we get weights for offensive features
    def get_weights_offensive(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1, 'ghost_distance': 1, 'safe_zone_distance': -10}

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
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

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