import os

from matplotlib import image

from rl.RootPath import rootPath

resources_path = os.path.join(rootPath, "resources")


def save_input_as_image(pixel_array, name):
    try:
        image.imsave(os.path.join(resources_path, name + '.png'), pixel_array)
    except Exception as e:
        print("Save image exception: " + str(e))
