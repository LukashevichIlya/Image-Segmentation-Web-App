import os

model_scheme = "FCN.png"
image_files = ["airplane.jpg", "dog-and-cat.jpg", "horse-riding.jpg", "people-cars.jpg", "person-sheep.png"]
image_names = ["Airplane", "Dog and cat", "Horse riding", "People and cars", "Person and sheep"]
image_folder = "static"

model_scheme_file = os.path.join(image_folder, model_scheme)
image_name_file_dict = {image_name: os.path.join(image_folder, file_name)
                        for image_name, file_name in zip(image_names, image_files)}
