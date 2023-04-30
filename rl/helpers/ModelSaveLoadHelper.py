import os


class ModelSaveLoadHelper:

    @staticmethod
    def save_weights_h5(model, path):
        full = path + ".h5"
        dirn = os.path.dirname(full)
        if not os.path.exists(dirn):
            os.makedirs(dirn)
        try:
            model.save_weights(full)
        except Exception as e:
            print("Save weights ex: " + str(e))
            return

    @staticmethod
    def simple_save(model, path):
        try:
            model.save(path)
        except IOError as e:
            print("Save model ex: " + str(e))
            return

    @staticmethod
    def load_weights_h5(path, model_func):
        full = path + ".h5"
        try:
            if os.path.exists(full):
                model = model_func()
                model.load_weights(full)
            else:
                return None
            return model
        except IOError as e:
            return None
