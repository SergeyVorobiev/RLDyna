from matplotlib import image

from rl.ProjectPath import ProjectPath


def save_input_as_image(pixel_array, name):
    try:
        image.imsave(ProjectPath.join_to_resources_path(name), pixel_array)
    except Exception as e:
        print("Save image exception: " + str(e))
