

class ActionBuilder(object):

    @staticmethod
    def build_by_keys(keys: []) -> {}:
        result = {}
        k = 0
        for key in keys:
            result[key] = Action(k, key)
            k += 1
        return result

    @staticmethod
    def build_by_key_names(key_names: []) -> {}:
        result = {}
        k = 0
        for key_name in key_names:
            key = key_name[0]
            name = key_name[1]
            result[key] = Action(k, key, name)
            k += 1
        return result


class Action(object):

    def __init__(self, a_id: int, key, name: str = None):
        if name is None:
            self.name: str = str(key)
        else:
            self.name: str = name
        self.a_id: int = a_id
        self.key: object = key
        self.visit_count: int = 0
        self.q: float = 0.0
        self.visited: bool = False

    def reset(self):
        self.visit_count = 0
        self.q = 0

    def default_clone(self, a_id: int = None):
        if a_id is None:
            a_id = self.a_id
        action: Action = Action(a_id, self.key, self.name)
        return action

    def full_clone(self, a_id: int = None):
        if a_id is None:
            a_id = self.a_id
        action: Action = Action(a_id, self.key, self.name)
        action.visit_count = self.visit_count
        action.q = self.q
        return action
