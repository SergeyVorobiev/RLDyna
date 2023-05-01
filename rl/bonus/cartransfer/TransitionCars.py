import math

from rl.mdp.Action import Action
from rl.mdp.State import State
from rl.mdp.Transition import Transition


class TransitionCars(Transition):

    def __init__(self):
        self.request_first_l = 3
        self.request_second_l = 4
        self.return_first_l = 3
        self.return_second_l = 2
        self.max_cars = 20
        self.request_car_reward = 10
        self.move_cost_per_car = 2
        self.map_cache = {}
        self.__prepare_probability_map()
        self.reward_probability_table_first = self.__init_table(self.max_cars)
        self.reward_probability_table_second = self.__init_table(self.max_cars)

    def __init_table(self, max_cars):
        table = []
        for car in range(max_cars + 1):
            car_list = []
            table.append(car_list)
            for next_car in range(max_cars + 1):
                probability_reward = [None, None]
                car_list.append(probability_reward)
        return table

    def __prepare_probability_map(self):
        keys = [self.request_first_l, self.request_second_l, self.return_first_l, self.return_second_l]
        for l in keys:
            if self.map_cache.get(l) is None:
                array = []
                self.map_cache[l] = array
                for n in range(self.max_cars + 1):
                    array.append(self.__probability_formula(l, n))

    def __probability_formula(self, l, n):
        return math.pow(l, n) / math.factorial(n) * math.pow(math.e, -l)

    def __probability(self, l, n):
        # return math.pow(l, n) / math.factorial(n) * math.pow(math.e, -l)
        return self.map_cache[l][n]

    def __cant_do_action(self, expected_cars_in_first_location, expected_cars_in_second_location):
        return expected_cars_in_first_location < 0 or expected_cars_in_first_location > self.max_cars or expected_cars_in_second_location < 0 or expected_cars_in_second_location > self.max_cars

    def get_transition(self, s: State, a: Action, sn: State) -> (float, float):
        cars_in_first_location = s.info[0]
        cars_in_second_location = s.info[1]
        next_cars_in_first_location = sn.info[0]
        next_cars_in_second_location = sn.info[1]
        cars_in_first_location = cars_in_first_location - a.key
        cars_in_second_location = cars_in_second_location + a.key
        if self.__cant_do_action(cars_in_first_location, cars_in_second_location):
            return 0, 0
        first_probability, first_reward = self.__p_s_r(cars_in_first_location, next_cars_in_first_location,
                                                       self.request_first_l, self.return_first_l, True)
        second_probability, second_reward = self.__p_s_r(cars_in_second_location, next_cars_in_second_location,
                                                         self.request_second_l, self.return_second_l, False)
        probability = first_probability * second_probability
        reward = first_reward + second_reward - abs(a.key) * self.move_cost_per_car
        return probability, reward * probability

    def __p_s_r(self, available_cars, left_cars, request_lambda, return_lambda, is_first):
        # print("Available cars: " + str(available_cars) + " Left cars: " + str(left_cars))
        if is_first:
            table = self.reward_probability_table_first
        else:
            table = self.reward_probability_table_second
        result_list = table[available_cars][left_cars]
        if result_list[0] is not None:
            return result_list[0], result_list[1]
        probability = 0
        reward = 0
        if available_cars <= left_cars:
            for requested_cars in range(self.max_cars + 1):
                actual_requested_cars = min(requested_cars, available_cars)
                if left_cars == self.max_cars:
                    returned_probability = 0
                    start = left_cars - available_cars + requested_cars
                    start = min(start, self.max_cars)
                    # returned_cars_str = "(" + str(start) + " - " + str(self.max_cars) + ")"
                    for returned_cars in range(start, self.max_cars + 1):
                        returned_probability += self.__probability(return_lambda, returned_cars)
                else:
                    returned_cars = min(left_cars - available_cars + requested_cars, left_cars)
                    # returned_cars_str = "(" + str(returned_cars) + ")"
                    returned_probability = self.__probability(return_lambda, returned_cars)
                requested_probability = self.__probability(request_lambda, requested_cars)
                rr_probability = requested_probability * returned_probability
                # print("Requested cars: " + str(requested_cars) + " Actual requested cars: " + str(actual_requested_cars) + " Returned cars: " + returned_cars_str + " Probability: " + str(rr_probability))
                reward += actual_requested_cars * self.request_car_reward * requested_probability
                probability += rr_probability
        else:
            for requested_cars in range(available_cars - left_cars, self.max_cars + 1):
                actual_requested_cars = min(requested_cars, available_cars)
                returned_cars = min(requested_cars + left_cars - available_cars, left_cars)
                requested_probability = self.__probability(request_lambda, requested_cars)
                returned_probability = self.__probability(return_lambda, returned_cars)
                rr_probability = requested_probability * returned_probability
                # print("Requested cars: " + str(requested_cars) + " Actual requested cars: " + str(actual_requested_cars) + " Returned cars: " + str(returned_cars) + " Probability: " + str(rr_probability))
                reward += actual_requested_cars * self.request_car_reward * requested_probability
                probability += rr_probability
        # print("Sum probability: " + str(probability) + " Expected reward: " + str(reward))
        result_list[0] = probability
        result_list[1] = reward
        return probability, reward


def check(s1):
    areward = 0
    all_probability = 0
    for fe in range(21):
        for se in range(21):
            s2 = State(0, "s2", actions, info=[fe, se])
            probability, reward = c.get_transition(s1, Action(0, 0), s2)
            areward += reward
            # print(str(fs_end) + " - " + str(ss_end))
            # print(reward)
            all_probability += probability
    return all_probability, areward


if __name__ == '__main__':
    c = TransitionCars()
    fs_start = 6
    ss_start = 0
    max_cars = 20
    actions = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    for fs in range(20, 21):
        for ss in range(20, 21):
            s11 = State(0, "s1", actions, info=[fs, ss])
            all_probability, areward = check(s11)
            print("Prob: " + str(all_probability))
            print("Reward: " + str(areward))
