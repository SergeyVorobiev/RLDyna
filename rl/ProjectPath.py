import os


class ProjectPath:

    rootPath = os.path.dirname(__file__)
    resources_path = os.path.join(rootPath, "resources")
    models_path = os.path.join(resources_path, "models")
    nn_models_path = os.path.join(models_path, "nn")
    table_models_path = os.path.join(models_path, "table")

    @staticmethod
    def join_to_resources_path(path):
        return os.path.join(ProjectPath.resources_path, path)

    @staticmethod
    def join_to_res_models_path(path):
        return os.path.join(ProjectPath.models_path, path)

    @staticmethod
    def join_to_nn_models_path(path):
        return os.path.join(ProjectPath.nn_models_path, path)

    @staticmethod
    def join_to_table_models_path(path):
        return os.path.join(ProjectPath.table_models_path, path)
