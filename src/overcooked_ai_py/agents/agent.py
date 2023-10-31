import itertools
import math
import os
from collections import defaultdict

import dill
import numpy as np

from overcooked_ai_py.mdp.actions import Action

class Agent(object):

    def action(self, state):
        return NotImplementedError()

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        pass

class AgentGroup(object):
    """
    AgentGroup is a group of N agents used to sample 
    joint actions in the context of an OvercookedEnv instance.
    """

    def __init__(self, *agents, allow_duplicate_agents=False):
        self.agents = agents
        self.n = len(self.agents)
        for i, agent in enumerate(self.agents):
            agent.set_agent_index(i)

        if not all(a0 is not a1 for a0, a1 in itertools.combinations(agents, 2)):
            assert allow_duplicate_agents, "All agents should be separate instances, unless allow_duplicate_agents is set to true"

    def joint_action(self, state):
        return tuple(a.action(state) for a in self.agents)

    def set_mdp(self, mdp):
        for a in self.agents:
            a.set_mdp(mdp)

    def reset(self):
        for a in self.agents:
            a.reset()


class AgentPair(AgentGroup):
    """
    AgentPair is the N=2 case of AgentGroup. Unlike AgentGroup,
    it supports having both agents being the same instance of Agent.
    
    NOTE: Allowing duplicate agents (using the same instance of an agent
    for both fields can lead to problems if the agents have state / history)
    """

    def __init__(self, *agents, allow_duplicate_agents=False): 
        super().__init__(*agents, allow_duplicate_agents=allow_duplicate_agents)
        assert self.n == 2
        self.a0, self.a1 = self.agents

        # if type(self.a0) is CoupledPlanningAgent and type(self.a1) is CoupledPlanningAgent:
        # print("If the two planning agents have same params, consider using CoupledPlanningPair instead to reduce computation time by a factor of 2")

    def joint_action(self, state):
        if self.a0 is self.a1:
            # When using the same instance of an agent for self-play, 
            # reset agent index at each turn to prevent overwriting it
            self.a0.set_agent_index(0)
            action_0 = self.a0.action(state)
            self.a1.set_agent_index(1)
            action_1 = self.a1.action(state)
            return (action_0, action_1)
        else:
            return super().joint_action(state)

class NNPolicy(object):
    """
    This is a common format for NN-based policies. Once one has wrangled the intended trained neural net
    to this format, one can then easily create an Agent with the AgentFromPolicy class.
    """

    def __init__(self):
        pass

    def multi_state_policy(self, states, agent_indices):
        """
        A function that takes in multiple OvercookedState instances and their respective agent indices and returns action probabilities.
        """
        raise NotImplementedError()

    def multi_obs_policy(self, states):
        """
        A function that takes in multiple preprocessed OvercookedState instatences and returns action probabilities.
        """
        raise NotImplementedError()


class AgentFromPolicy(Agent):
    """
    Defines an agent from a `state_policy` and `direct_policy` functions
    """
    
    def __init__(self, state_policy, direct_policy, stochastic=True, action_probs=False):
        """
        state_policy (fn): a function that takes in an OvercookedState instance and returns corresponding actions
        direct_policy (fn): a function that takes in a preprocessed OvercookedState instances and returns actions
        stochastic (Bool): Whether the agent should sample from policy or take argmax
        action_probs (Bool): Whether agent should return action probabilities or a sampled action
        """
        self.state_policy = state_policy
        self.direct_policy = direct_policy
        self.history = []
        self.stochastic = stochastic
        self.action_probs = action_probs

    def action(self, state):
        """
        The standard action function call, that takes in a Overcooked state
        and returns the corresponding action.

        Requires having set self.agent_index and self.mdp
        """
        self.history.append(state)
        try:
            return self.state_policy(state, self.mdp, self.agent_index, self.stochastic, self.action_probs)
        except AttributeError as e:
            raise AttributeError("{}. Most likely, need to set the agent_index or mdp of the Agent before calling the action method.".format(e))

    def direct_action(self, obs):
        """
        A action called optimized for multi-threaded environment simulations
        involving the agent. Takes in SIM_THREADS (as defined when defining the agent)
        number of observations in post-processed form, and returns as many actions.
        """
        return self.direct_policy(obs)

    def reset(self):
        self.history = []

class RandomAgent(Agent):
    """
    An agent that randomly picks motion actions.
    NOTE: Does not perform interact actions, unless specified
    """

    def __init__(
        self, sim_threads=None, all_actions=False, custom_wait_prob=None
    ):
        self.sim_threads = sim_threads
        self.all_actions = all_actions
        self.custom_wait_prob = custom_wait_prob

    def action(self, state):
        action_probs = np.zeros(Action.NUM_ACTIONS)
        legal_actions = list(Action.MOTION_ACTIONS)
        if self.all_actions:
            legal_actions = Action.ALL_ACTIONS
        legal_actions_indices = np.array(
            [Action.ACTION_TO_INDEX[motion_a] for motion_a in legal_actions]
        )
        action_probs[legal_actions_indices] = 1 / len(legal_actions_indices)

        if self.custom_wait_prob is not None:
            stay = Action.STAY
            if np.random.random() < self.custom_wait_prob:
                return stay, {"action_probs": Agent.a_probs_from_action(stay)}
            else:
                action_probs = Action.remove_indices_and_renormalize(
                    action_probs, [Action.ACTION_TO_INDEX[stay]]
                )

        return Action.sample(action_probs), {"action_probs": action_probs}

    def actions(self, states, agent_indices):
        return [self.action(state) for state in states]

    def direct_action(self, obs):
        return [np.random.randint(4) for _ in range(self.sim_threads)]


class StayAgent(Agent):
    def __init__(self, sim_threads=None):
        self.sim_threads = sim_threads

    def action(self, state):
        a = Action.STAY
        return a, {}

    def direct_action(self, obs):
        return [Action.ACTION_TO_INDEX[Action.STAY]] * self.sim_threads


class FixedPlanAgent(Agent):
    """
    An Agent with a fixed plan. Returns Stay actions once pre-defined plan has terminated.
    # NOTE: Assumes that calls to action are sequential (agent has history)
    """

    def __init__(self, plan):
        self.plan = plan
        self.i = 0

    def action(self, state):
        if self.i >= len(self.plan):
            return Action.STAY, {}
        curr_action = self.plan[self.i]
        self.i += 1
        return curr_action, {}

    def reset(self):
        super().reset()
        self.i = 0


class GreedyHumanModel(Agent):
    """
    Agent that at each step selects a medium level action corresponding
    to the most intuitively high-priority thing to do

    NOTE: MIGHT NOT WORK IN ALL ENVIRONMENTS, for example forced_coordination.layout,
    in which an individual agent cannot complete the task on their own.
    Will work only in environments where the only order is 3 onion soup.
    """

    def __init__(
        self,
        mlam,
        hl_boltzmann_rational=False,
        ll_boltzmann_rational=False,
        hl_temp=1,
        ll_temp=1,
        auto_unstuck=True,
    ):
        self.mlam = mlam
        self.mdp = self.mlam.mdp

        # Bool for perfect rationality vs Boltzmann rationality for high level and low level action selection
        self.hl_boltzmann_rational = hl_boltzmann_rational  # For choices among high level goals of same type
        self.ll_boltzmann_rational = (
            ll_boltzmann_rational  # For choices about low level motion
        )

        # Coefficient for Boltzmann rationality for high level action selection
        self.hl_temperature = hl_temp
        self.ll_temperature = ll_temp

        # Whether to automatically take an action to get the agent unstuck if it's in the same
        # state as the previous turn. If false, the agent is history-less, while if true it has history.
        self.auto_unstuck = auto_unstuck
        self.reset()

    def reset(self):
        super().reset()
        self.prev_state = None

    def actions(self, states, agent_indices):
        actions_and_infos_n = []
        for state, agent_idx in zip(states, agent_indices):
            self.set_agent_index(agent_idx)
            self.reset()
            actions_and_infos_n.append(self.action(state))
        return actions_and_infos_n

    def action(self, state):
        possible_motion_goals = self.ml_action(state)

        # Once we have identified the motion goals for the medium
        # level action we want to perform, select the one with lowest cost
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        chosen_goal, chosen_action, action_probs = self.choose_motion_goal(
            start_pos_and_or, possible_motion_goals
        )

        if (
            self.ll_boltzmann_rational
            and chosen_goal[0] == start_pos_and_or[0]
        ):
            chosen_action, action_probs = self.boltzmann_rational_ll_action(
                start_pos_and_or, chosen_goal
            )

        if self.auto_unstuck:
            # HACK: if two agents get stuck, select an action at random that would
            # change the player positions if the other player were not to move
            if (
                self.prev_state is not None
                and state.players_pos_and_or
                == self.prev_state.players_pos_and_or
            ):
                if self.agent_index == 0:
                    joint_actions = list(
                        itertools.product(Action.ALL_ACTIONS, [Action.STAY])
                    )
                elif self.agent_index == 1:
                    joint_actions = list(
                        itertools.product([Action.STAY], Action.ALL_ACTIONS)
                    )
                else:
                    raise ValueError("Player index not recognized")

                unblocking_joint_actions = []
                for j_a in joint_actions:
                    new_state, _ = self.mlam.mdp.get_state_transition(
                        state, j_a
                    )
                    if (
                        new_state.player_positions
                        != self.prev_state.player_positions
                    ):
                        unblocking_joint_actions.append(j_a)
                # Getting stuck became a possiblity simply because the nature of a layout (having a dip in the middle)
                if len(unblocking_joint_actions) == 0:
                    unblocking_joint_actions.append([Action.STAY, Action.STAY])
                chosen_action = unblocking_joint_actions[
                    np.random.choice(len(unblocking_joint_actions))
                ][self.agent_index]
                action_probs = self.a_probs_from_action(chosen_action)

            # NOTE: Assumes that calls to the action method are sequential
            self.prev_state = state
        return chosen_action, {"action_probs": action_probs}

    def choose_motion_goal(self, start_pos_and_or, motion_goals):
        """
        For each motion goal, consider the optimal motion plan that reaches the desired location.
        Based on the plan's cost, the method chooses a motion goal (either boltzmann rationally
        or rationally), and returns the plan and the corresponding first action on that plan.
        """
        if self.hl_boltzmann_rational:
            possible_plans = [
                self.mlam.motion_planner.get_plan(start_pos_and_or, goal)
                for goal in motion_goals
            ]
            plan_costs = [plan[2] for plan in possible_plans]
            goal_idx, action_probs = self.get_boltzmann_rational_action_idx(
                plan_costs, self.hl_temperature
            )
            chosen_goal = motion_goals[goal_idx]
            chosen_goal_action = possible_plans[goal_idx][0][0]
        else:
            (
                chosen_goal,
                chosen_goal_action,
            ) = self.get_lowest_cost_action_and_goal(
                start_pos_and_or, motion_goals
            )
            action_probs = self.a_probs_from_action(chosen_goal_action)
        return chosen_goal, chosen_goal_action, action_probs

    def get_boltzmann_rational_action_idx(self, costs, temperature):
        """Chooses index based on softmax probabilities obtained from cost array"""
        costs = np.array(costs)
        softmax_probs = np.exp(-costs * temperature) / np.sum(
            np.exp(-costs * temperature)
        )
        action_idx = np.random.choice(len(costs), p=softmax_probs)
        return action_idx, softmax_probs

    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(
                start_pos_and_or, goal
            )
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal
        return best_goal, best_action

    def boltzmann_rational_ll_action(
        self, start_pos_and_or, goal, inverted_costs=False
    ):
        """
        Computes the plan cost to reach the goal after taking each possible low level action.
        Selects a low level action boltzmann rationally based on the one-step-ahead plan costs.

        If `inverted_costs` is True, it will make a boltzmann "irrational" choice, exponentially
        favouring high cost plans rather than low cost ones.
        """
        future_costs = []
        for action in Action.ALL_ACTIONS:
            pos, orient = start_pos_and_or
            new_pos_and_or = self.mdp._move_if_direction(pos, orient, action)
            _, _, plan_cost = self.mlam.motion_planner.get_plan(
                new_pos_and_or, goal
            )
            sign = (-1) ** int(inverted_costs)
            future_costs.append(sign * plan_cost)

        action_idx, action_probs = self.get_boltzmann_rational_action_idx(
            future_costs, self.ll_temperature
        )
        return Action.ALL_ACTIONS[action_idx], action_probs

    def ml_action(self, state):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]

        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of greedy heuristics based on the current state.

        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]
        am = self.mlam

        counter_objects = self.mlam.mdp.get_counter_objects_dict(
            state, list(self.mlam.mdp.terrain_pos_dict["X"])
        )
        pot_states_dict = self.mlam.mdp.get_pot_states(state)

        if not player.has_object():
            ready_soups = pot_states_dict["ready"]
            cooking_soups = pot_states_dict["cooking"]

            soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
            other_has_dish = (
                other_player.has_object()
                and other_player.get_object().name == "dish"
            )

            if soup_nearly_ready and not other_has_dish:
                motion_goals = am.pickup_dish_actions(counter_objects)
            else:
                assert len(state.all_orders) == 1 and list(
                    state.all_orders[0].ingredients
                ) == ["onion", "onion", "onion"], (
                    "The current mid level action manager only support 3-onion-soup order, but got orders"
                    + str(state.all_orders)
                )
                next_order = list(state.all_orders)[0]
                soups_ready_to_cook_key = "{}_items".format(
                    len(next_order.ingredients)
                )
                soups_ready_to_cook = pot_states_dict[soups_ready_to_cook_key]
                if soups_ready_to_cook:
                    only_pot_states_ready_to_cook = defaultdict(list)
                    only_pot_states_ready_to_cook[
                        soups_ready_to_cook_key
                    ] = soups_ready_to_cook
                    # we want to cook only soups that has same len as order
                    motion_goals = am.start_cooking_actions(
                        only_pot_states_ready_to_cook
                    )
                else:
                    motion_goals = am.pickup_onion_actions(counter_objects)
                # it does not make sense to have tomato logic when the only possible order is 3 onion soup (see assertion above)
                # elif 'onion' in next_order:
                #     motion_goals = am.pickup_onion_actions(counter_objects)
                # elif 'tomato' in next_order:
                #     motion_goals = am.pickup_tomato_actions(counter_objects)
                # else:
                #     motion_goals = am.pickup_onion_actions(counter_objects) + am.pickup_tomato_actions(counter_objects)

        else:
            player_obj = player.get_object()

            if player_obj.name == "onion":
                motion_goals = am.put_onion_in_pot_actions(pot_states_dict)

            elif player_obj.name == "tomato":
                motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)

            elif player_obj.name == "dish":
                motion_goals = am.pickup_soup_with_dish_actions(
                    pot_states_dict, only_nearly_ready=True
                )

            elif player_obj.name == "soup":
                motion_goals = am.deliver_soup_actions()

            else:
                raise ValueError()

        motion_goals = [
            mg
            for mg in motion_goals
            if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                player.pos_and_or, mg
            )
        ]

        if len(motion_goals) == 0:
            motion_goals = am.go_to_closest_feature_actions(player)
            motion_goals = [
                mg
                for mg in motion_goals
                if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                    player.pos_and_or, mg
                )
            ]
            assert len(motion_goals) != 0

        return motion_goals


class SampleAgent(Agent):
    """Agent that samples action using the average action_probs across multiple agents"""

    def __init__(self, agents):
        self.agents = agents

    def action(self, state):
        action_probs = np.zeros(Action.NUM_ACTIONS)
        for agent in self.agents:
            action_probs += agent.action(state)[1]["action_probs"]
        action_probs = action_probs / len(self.agents)
        return Action.sample(action_probs), {"action_probs": action_probs}