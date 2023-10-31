import copy
import itertools

import numpy as np


class Direction(object):
    """
    The four possible directions a player can be facing.
    """

    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)
    ALL_DIRECTIONS = INDEX_TO_DIRECTION = [NORTH, SOUTH, EAST, WEST]
    DIRECTION_TO_INDEX = {a: i for i, a in enumerate(INDEX_TO_DIRECTION)}
    OPPOSITE_DIRECTIONS = {NORTH: SOUTH, SOUTH: NORTH, EAST: WEST, WEST: EAST}
    DIRECTION_TO_NAME = {
        d: name
        for d, name in zip(
            [NORTH, SOUTH, EAST, WEST], ["NORTH", "SOUTH", "EAST", "WEST"]
        )
    }

    @staticmethod
    def get_adjacent_directions(direction):
        """Returns the directions within 90 degrees of the given direction.

        direction: One of the Directions, except not Direction.STAY.
        """
        if direction in [Direction.NORTH, Direction.SOUTH]:
            return [Direction.EAST, Direction.WEST]
        elif direction in [Direction.EAST, Direction.WEST]:
            return [Direction.NORTH, Direction.SOUTH]
        raise ValueError("Invalid direction: %s" % direction)

LLMActionSet = {
    'cramped_room' : {
        'onion_dispenser' : ['pick up onion from o0.', 'pick up onion from o1.'],
        'cooking_status': ['turn on c0.'],
        'cooker': ['place onion in c0.'], 
        'plate': ['pick up plate from p0.'],
        'delivery_area': ['place soup for delivery in d0.'],
        'cooked_soup': ['load soup on plate from c0.'],
        'collision_avoidance': ['move away.'],
        'wait': ['wait.'],
        'kitchen_counter_pick_onion': [f'pick up onion from k{i}.' for i in range(50)],
        'kitchen_counter_pick_plate': [f'pick up plate from k{i}.' for i in range(50)],
        'kitchen_counter_pick_soup': [f'pick up cooked soup from k{i}.' for i in range(50)],
        'kitchen_counter_place_onion': [f'place onion on k{idx}.' for idx in range(50)], 
        'kitchen_counter_place_plate': [f'place plate on k{idx}.' for idx in range(50)],
        
    },

    'no_counter_door' : {
        'onion_dispenser' : ['pick up onion from o0.', 'pick up onion from o1.'],
        'cooking_status': ['turn on c0.'],
        'cooker': ['place onion in c0.'], 
        'plate': ['pick up plate from p0.'],
        'delivery_area': ['place soup for delivery in d0.'],
        'cooked_soup': ['load soup on plate from c0.'],
        'collision_avoidance': ['move away.'],
        'wait': ['wait.'],
        'gate': ['open g0.', 'open g1.'],
        'kitchen_counter_pick_onion': [f'pick up onion from k{i}.' for i in range(50)],
        'kitchen_counter_pick_plate': [f'pick up plate from k{i}.' for i in range(50)],
        'kitchen_counter_pick_soup': [f'pick up cooked soup from k{i}.' for i in range(50)],
        'kitchen_counter_place_onion': [f'place onion on k{idx}.' for idx in range(50)], 
        'kitchen_counter_place_plate': [f'place plate on k{idx}.' for idx in range(50)],
    },
    'soup_passing_door' : {
        'onion_dispenser' : ['pick up onion from o0.', 'pick up onion from o1.'],
        
        'cooker': ['place onion in c0.', 'place onion in c1.'], 
        'plate': ['pick up plate from p0.', 'pick up plate from p1.'],
        'delivery_area': ['place soup for delivery in d0.'],
        'cooked_soup': ['load soup on plate from c0.', 'load soup on plate from c1.'],
        'collision_avoidance': ['move away.'],
        'wait': ['wait.'],
        'gate': ['open g0.'],
        'kitchen_counter_pick_onion': [f'pick up onion from k{i}.' for i in range(50)],
        'kitchen_counter_pick_plate': [f'pick up plate from k{i}.' for i in range(50)],
        'kitchen_counter_pick_soup': [f'pick up cooked soup from k{i}.' for i in range(50)],
        'kitchen_counter_place_onion': [f'place onion on k{idx}.' for idx in range(50)], 
        'kitchen_counter_place_plate': [f'place plate on k{idx}.' for idx in range(50)],
    },
    'soup_passing' : {
        'onion_dispenser' : ['pick up onion from o0.', 'pick up onion from o1.'],
        'cooker': ['place onion in c0.', 'place onion in c1.'], 
        'plate': ['pick up plate from p0.', 'pick up plate from p1.'],
        'delivery_area': ['place soup for delivery in d0.'],
        'cooked_soup': ['load soup on plate from c0.', 'load soup on plate from c1.'],
        'collision_avoidance': ['move away.'],
        'wait': ['wait.'],
        'storage_counter_pick_onion': ['pick up onion from s0.', 'pick up onion from s1.', 'pick up onion from s2.', 'pick up onion from s3.'],
        'storage_counter_pick_plate': ['pick up plate from s0.', 'pick up a plate from s1.', 'pick up a plate from s2.', 'pick up a plate from s3.'],
        'storage_counter_pick_soup': ['pick up cooked soup from s0.', 'pick up cooked soup from s1.', 'pick up cooked soup from s2.', 'pick up cooked soup from s3.'],
        'storage_counter_place_onion': ['place onion in s0.', 'place onion in s1.', 'place onion in s2.', 'place onion in s3.'],
        'storage_counter_place_plate': ['place plate in s0.', 'place plate in s1.', 'place plate in s2.', 'place plate in s3'],
        'storage_counter_place_soup': ['place cooked soup in s0.', 'place cooked soup in s1.', 'place cooked soup in s2.', 'place cooked soup in s3'],
        'kitchen_counter_pick_onion': [f'pick up onion from k{i}.' for i in range(50)],
        'kitchen_counter_pick_plate': [f'pick up plate from k{i}.' for i in range(50)],
        'kitchen_counter_pick_soup': [f'pick up cooked soup from k{i}.' for i in range(50)],
        'kitchen_counter_place_onion': [f'place onion on k{idx}.' for idx in range(50)], 
        'kitchen_counter_place_plate': [f'place plate on k{idx}.' for idx in range(50)],
        'kitchen_counter_place_soup': [f'place cooked soup on k{idx}.' for idx in range(50)],
    },
    'coordination_ring' : {
        'onion_dispenser' : ['pick up onion from o0.', 'pick up onion from o1.'],
        'cooking_status': ['turn on c0.', 'turn on c1.'],
        'cooker': ['place onion in c0.', 'place onion in c1.'], 
        'plate': ['pick up plate from p0.'],
        'delivery_area': ['place soup for delivery in d0.'],
        'cooked_soup': ['load soup on plate from c0.', 'load soup on plate from c1.'],
        'collision_avoidance': ['move away.'],
        'counter': ['place onion on k0', 'place plate on k0'],
        'wait': ['wait.'],
        'communication': ['send message: move away.', 'send message: send me a plate in the storage counter.', 'send message: send me an onion in the storage counter.', 'send acknowledgement: okay, i moved away.', 'send acknowledgement: okay, i placed an onion in the storage counter.', 'send acknowledgement: okay, i placed a plate in the storage counter.'],
        'kitchen_counter_pick_onion': [f'pick up onion from k{i}.' for i in range(50)],
        'kitchen_counter_pick_plate': [f'pick up plate from k{i}.' for i in range(50)],
        'kitchen_counter_pick_soup': [f'pick up cooked soup from k{i}.' for i in range(50)],
        'kitchen_counter_place_onion': [f'place onion on k{idx}.' for idx in range(50)], 
        'kitchen_counter_place_plate': [f'place plate on k{idx}.' for idx in range(50)],
        
    },
    'counter_circuit_o_1order' : {
        'onion_dispenser' : ['pick up onion from o0.', 'pick up onion from o1.'],
        'cooking_status': ['turn on c0.', 'turn on c1.'],
        'cooker': ['place onion in c0.', 'place onion in c1.'], 
        'plate': ['pick up plate from p0.'],
        'delivery_area': ['place soup for delivery in d0.'],
        'cooked_soup': ['load soup on plate from c0.', 'load soup on plate from c1.'],
        'collision_avoidance': ['move away.'],
        'counter': ['place onion on k0', 'place plate on k0'],
        'wait': ['wait.'],
        'storage_counter_pick_onion': ['pick up onion from s0.', 'pick up onion from s1.', 'pick up onion from s2.', 'pick up onion from s3.'],
        'storage_counter_pick_plate': ['pick up a plate from s0.', 'pick up a plate from s1.', 'pick up a plate from s2.', 'pick up a plate from s3.'],
        'storage_counter_place_onion': ['place onion in s0.', 'place onion in s1.', 'place onion in s2.', 'place onion in s3.'],
        'storage_counter_place_plate': ['place plate in s0.', 'place plate in s1.', 'place plate in s2.', 'place plate in s3'],
        # 'communication': ['send message: move out of the way.', 'send message: pass me an onion.', 'send message: pass me a plate.']
        'kitchen_counter_pick_onion': [f'pick up onion from k{i}.' for i in range(50)],
        'kitchen_counter_pick_plate': [f'pick up plate from k{i}.' for i in range(50)],
        'kitchen_counter_pick_soup': [f'pick up cooked soup from k{i}.' for i in range(50)],
        'kitchen_counter_place_onion': [f'place onion on k{idx}.' for idx in range(50)], 
        'kitchen_counter_place_plate': [f'place plate on k{idx}.' for idx in range(50)],
    },
    'asymmetric_advantages' : {
        'onion_dispenser' : ['pick up onion from o0.', 'pick up onion from o1.'],
        'cooking_status': ['turn on c0.', 'turn on c1.'],
        'cooker': ['place onion in c0.', 'place onion in c1.'], 
        'plate': ['pick up plate from p0.', 'pick up plate from p1.'],
        'delivery_area': ['place soup for delivery in d0.', 'place soup for delivery in d1.'],
        'cooked_soup': ['load soup on plate from c0.', 'load soup on plate from c1.'],
        'collision_avoidance': ['move away.'],
        'wait': ['wait.'],
        'kitchen_counter_pick_onion': [f'pick up onion from k{i}.' for i in range(50)],
        'kitchen_counter_pick_plate': [f'pick up plate from k{i}.' for i in range(50)],
        'kitchen_counter_pick_soup': [f'pick up cooked soup from k{i}.' for i in range(50)],
        'kitchen_counter_place_onion': [f'place onion on k{idx}.' for idx in range(50)], 
        'kitchen_counter_place_plate': [f'place plate on k{idx}.' for idx in range(50)],
    },
    'forced_coordination' : {
        'onion_dispenser' : ['pick up onion from o0.', 'pick up onion from o1.'],
        'cooking_status': ['turn on c0.', 'turn on c1.'],
        'cooker': ['place onion in c0.', 'place onion in c1.'], 
        'plate': ['pick up plate from p0.', 'pick up plate from p1.'],
        'delivery_area': ['place soup for delivery in d0.', 'place soup for delivery in d1.'],
        'cooked_soup': ['load soup on plate from c0.', 'load soup on plate from c1.'],
        'storage_counter_pick_onion': ['pick up onion from s0.', 'pick up onion from s1.', 'pick up onion from s2.'],
        'storage_counter_pick_plate': ['pick up a plate from s0.', 'pick up a plate from s1.', 'pick up a plate from s2.'],
        'storage_counter_place_onion': ['place onion in s0.', 'place onion in s1.', 'place onion in s2.'],
        'storage_counter_place_plate': ['place plate in s0.', 'place plate in s1.', 'place plate in s2.'],
        'collision_avoidance': ['move away.'],
        'wait': ['wait.'],
        'kitchen_counter_pick_onion': [f'pick up onion from k{i}.' for i in range(50)],
        'kitchen_counter_pick_plate': [f'pick up plate from k{i}.' for i in range(50)],
        'kitchen_counter_pick_soup': [f'pick up cooked soup from k{i}.' for i in range(50)],
        'kitchen_counter_place_onion': [f'place onion on k{idx}.' for idx in range(50)], 
        'kitchen_counter_place_plate': [f'place plate on k{idx}.' for idx in range(50)],
    },
    'bottleneck':
    {
       'onion_dispenser' : ['pick up onion from o0.',],
        'cooking_status': ['turn on c0.', 'turn on c1.'],
        'cooker': ['place onion in c0.', 'place onion in c1.'], 
        'plate': ['pick up plate from p0.'],
        'delivery_area': ['place soup for delivery in d0.'],
        'cooked_soup': ['load soup on plate from c0.', 'load soup on plate from c1.'],
        'collision_avoidance': ['move away.'],
        'counter': ['place onion on k0', 'place plate on k0'],
        'wait': ['wait.'],
        'communication': ['send message: move away.', 'send message: send me a plate in the storage counter.', 'send message: send me an onion in the storage counter.', 'send acknowledgement: okay, i moved away.', 'send acknowledgement: okay, i placed an onion in the storage counter.', 'send acknowledgement: okay, i placed a plate in the storage counter.'],
        'kitchen_counter_pick_onion': [f'pick up onion from k{i}.' for i in range(50)],
        'kitchen_counter_pick_plate': [f'pick up plate from k{i}.' for i in range(50)],
        'kitchen_counter_pick_soup': [f'pick up cooked soup from k{i}.' for i in range(50)],
        'kitchen_counter_place_onion': [f'place onion on k{idx}.' for idx in range(50)], 
        'kitchen_counter_place_plate': [f'place plate on k{idx}.' for idx in range(50)], 
    },

    'centre_pots':
    {
       'onion_dispenser' : ['pick up onion from o0.', 'pick up onion from o1.'],
        'cooking_status': ['turn on c0.'],
        'cooker': ['place onion in c0.'], 
        'plate': ['pick up plate from p0.'],
        'delivery_area': ['place soup for delivery in d0.'],
        'cooked_soup': ['load soup on plate from c0.'],
        'collision_avoidance': ['move away.'],
        'counter': ['place onion on k0', 'place plate on k0'],
        'wait': ['wait.'],
        'communication': ['send message: move away.', 'send message: send me a plate in the storage counter.', 'send message: send me an onion in the storage counter.', 'send acknowledgement: okay, i moved away.', 'send acknowledgement: okay, i placed an onion in the storage counter.', 'send acknowledgement: okay, i placed a plate in the storage counter.'],
        'kitchen_counter_pick_onion': [f'pick up onion from k{i}.' for i in range(50)],
        'kitchen_counter_pick_plate': [f'pick up plate from k{i}.' for i in range(50)],
        'kitchen_counter_pick_soup': [f'pick up cooked soup from k{i}.' for i in range(50)],
        'kitchen_counter_place_onion': [f'place onion on k{idx}.' for idx in range(50)], 
        'kitchen_counter_place_plate': [f'place plate on k{idx}.' for idx in range(50)], 
    },

    'centre_objects':
    {
       'onion_dispenser' : ['pick up onion from o0.'],
        'cooking_status': ['turn on c0.'],
        'cooker': ['place onion in c0.'], 
        'plate': ['pick up plate from p0.'],
        'delivery_area': ['place soup for delivery in d0.'],
        'cooked_soup': ['load soup on plate from c0.'],
        'collision_avoidance': ['move away.'],
        'counter': ['place onion on k0', 'place plate on k0'],
        'wait': ['wait.'],
        'communication': ['send message: move away.', 'send message: send me a plate in the storage counter.', 'send message: send me an onion in the storage counter.', 'send acknowledgement: okay, i moved away.', 'send acknowledgement: okay, i placed an onion in the storage counter.', 'send acknowledgement: okay, i placed a plate in the storage counter.'],
        'kitchen_counter_pick_onion': [f'pick up onion from k{i}.' for i in range(50)],
        'kitchen_counter_pick_plate': [f'pick up plate from k{i}.' for i in range(50)],
        'kitchen_counter_pick_soup': [f'pick up cooked soup from k{i}.' for i in range(50)],
        'kitchen_counter_place_onion': [f'place onion on k{idx}.' for idx in range(50)], 
        'kitchen_counter_place_plate': [f'place plate on k{idx}.' for idx in range(50)], 
    },

    'large_room':{
        'onion_dispenser' : ['pick up onion from o0.', 'pick up onion from o1.'],
        'cooking_status': ['turn on c0.'],
        'cooker': ['place onion in c0.'], 
        'plate': ['pick up plate from p0.'],
        'delivery_area': ['place soup for delivery in d0.'],
        'cooked_soup': ['load soup on plate from c0.'],
        'collision_avoidance': ['move away.'],
        'counter': ['place onion on k0', 'place plate on k0'],
        'wait': ['wait.'],
        'communication': ['send message: move away.', 'send message: send me a plate in the storage counter.', 'send message: send me an onion in the storage counter.', 'send acknowledgement: okay, i moved away.', 'send acknowledgement: okay, i placed an onion in the storage counter.', 'send acknowledgement: okay, i placed a plate in the storage counter.'],
        'kitchen_counter_pick_onion': [f'pick up onion from k{i}.' for i in range(50)],
        'kitchen_counter_pick_plate': [f'pick up plate from k{i}.' for i in range(50)],
        'kitchen_counter_pick_soup': [f'pick up cooked soup from k{i}.' for i in range(50)],
        'kitchen_counter_place_onion': [f'place onion on k{idx}.' for idx in range(50)], 
        'kitchen_counter_place_plate': [f'place plate on k{idx}.' for idx in range(50)], 
    },
}


class Action(object):
    """
    The six actions available in the OvercookedGridworld.

    Includes definitions of the actions as well as utility functions for
    manipulating them or applying them.
    """

    STAY = (0, 0)
    INTERACT = "interact"
    REACHED = "reached"
    WAIT = "wait."
    ALL_ACTIONS = INDEX_TO_ACTION = Direction.INDEX_TO_DIRECTION + [
        STAY,
        INTERACT,
    ]
    INDEX_TO_ACTION_INDEX_PAIRS = [
        v for v in itertools.product(range(len(INDEX_TO_ACTION)), repeat=2)
    ]
    ACTION_TO_INDEX = {a: i for i, a in enumerate(INDEX_TO_ACTION)}
    MOTION_ACTIONS = Direction.ALL_DIRECTIONS + [STAY]
    ACTION_TO_CHAR = {
        Direction.NORTH: "↑",
        Direction.SOUTH: "↓",
        Direction.EAST: "→",
        Direction.WEST: "←",
        STAY: "stay",
        INTERACT: INTERACT,
    }
    NUM_ACTIONS = len(ALL_ACTIONS)

    @staticmethod
    def move_in_direction(point, direction):
        """
        Takes a step in the given direction and returns the new point.

        point: Tuple (x, y) representing a point in the x-y plane.
        direction: One of the Directions.
        """
        assert direction in Action.MOTION_ACTIONS
        x, y = point
        dx, dy = direction
        return (x + dx, y + dy)

    @staticmethod
    def determine_action_for_change_in_pos(old_pos, new_pos):
        """Determines an action that will enable intended transition"""
        if old_pos == new_pos:
            return Action.STAY
        new_x, new_y = new_pos
        old_x, old_y = old_pos
        direction = (new_x - old_x, new_y - old_y)
        assert direction in Direction.ALL_DIRECTIONS
        return direction

    @staticmethod
    def sample(action_probs):
        return np.random.choice(
            np.array(Action.ALL_ACTIONS, dtype=object), p=action_probs
        )

    @staticmethod
    def argmax(action_probs):
        action_idx = np.argmax(action_probs)
        return Action.INDEX_TO_ACTION[action_idx]

    @staticmethod
    def remove_indices_and_renormalize(probs, indices, eps=0.0):
        probs = copy.deepcopy(probs)
        if len(np.array(probs).shape) > 1:
            probs = np.array(probs)
            for row_idx, row in enumerate(indices):
                for idx in indices:
                    probs[row_idx][idx] = eps
            norm_probs = probs.T / np.sum(probs, axis=1)
            return norm_probs.T
        else:
            for idx in indices:
                probs[idx] = eps
            return probs / sum(probs)

    @staticmethod
    def to_char(action):
        assert action in Action.ALL_ACTIONS
        return Action.ACTION_TO_CHAR[action]

    @staticmethod
    def joint_action_to_char(joint_action):
        assert all([a in Action.ALL_ACTIONS for a in joint_action])
        return tuple(Action.to_char(a) for a in joint_action)

    @staticmethod
    def uniform_probs_over_actions():
        num_acts = len(Action.ALL_ACTIONS)
        return np.ones(num_acts) / num_acts
