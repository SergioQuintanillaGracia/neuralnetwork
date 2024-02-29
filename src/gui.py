from tkinter import filedialog, messagebox
from threading import Thread
import customtkinter as ctk
from PIL import Image
import bindings
import shutil
import json
import time
import os
import re

# Configure scaling and theme
scale = 1
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")
ctk.set_window_scaling(scale)
ctk.set_widget_scaling(scale)

# Create and configure the window
app = ctk.CTk()
app.geometry("940x560")
app.title("NeuralNetwork GUI")

# Variables
threads = 22
update_gen_label_interval = 25
update_model_accuracy_interval = 100
save_to_disk_interval = 250
image_limit = 1000
weights_mutation_range = 0.05
biases_mutation_range = 0.05
mutation_center_offset_range = 0.2
models_dir = "./networks/GUI_models/"
current_model_name: str = None
current_model_obj_names: list[str] = []
current_model_layers: list[int] = None
current_model_weights_path: str = None
current_model_biases_path: str = None
current_model_img_path: str = None
current_model_img_paths: list[str] = []
model_being_created_img_path: str = None

model_image_label: str = None
model_obj_labels: list = []
model_obj_labels_range: list[float] = (0.25, 0.85)
model_obj_labels_x: float = 0.756
model_model_optionmenu = None
train_model_optionmenu = None
train_obj_images_label = None
train_gen_label = None
train_gen_entry = None
train_base_obj_labels: list = []
train_base_obj_labels_range: list[float] = (0.26, 0.73)
train_base_obj_labels_x: float = 0.25
train_base_general_label = None
train_best_obj_labels: list = []
train_best_obj_labels_range: list[float] = (0.26, 0.73)
train_best_obj_labels_x: float = 0.75
train_best_general_label = None
manage_name_entry = None
manage_select_img_label = None
manage_obj_entry = None
manage_layers_entry = None

use_model_tab_name: str = "  Use Model  "
train_model_tab_name: str = "  Train Model  "
manage_models_tab_name: str = "  Manage Models  "
current_image_path: str = None
model_img_res_x: float = 380
model_img_res_y: float = 380
model_img_rel_pos_x: float = 0.263
model_img_rel_pos_y: float = 0.45

# Font definitions
large_font = ctk.CTkFont(family="", size=int(30/scale), weight="bold")
large_underlined_font = ctk.CTkFont(family="", size=int(30/scale), weight="bold", underline=True)
medium_large_underlined_font = ctk.CTkFont(family="", size=int(24/scale), weight="bold", underline=True)
medium_font = ctk.CTkFont(family="", size=int(18/scale), weight="bold")
small_font = ctk.CTkFont(family="", size=int(15/scale), weight="bold")
smaller_font = ctk.CTkFont(family="", size=int(12/scale))

# Define the tabview and add the tabs
tabview = ctk.CTkTabview(master=app, corner_radius=10, border_width=0)
tabview._segmented_button.configure(font=medium_font)
tabview.pack(expand=1, fill="both")
tabview.add(use_model_tab_name)
tabview.add(train_model_tab_name)
tabview.add(manage_models_tab_name)
tabview.set(train_model_tab_name)


# NEURAL NETWORK FUNCTIONS
def get_model_name_list() -> list[str]:
    it: iter = os.scandir(models_dir)
    dir_list: list[str] = [entry.name for entry in it if entry.is_dir()]
    return dir_list

def get_obj_to_img_directories(path) -> list[str]:
    img_paths: list[str] = []
    for obj in current_model_obj_names:
        img_paths.append(os.path.join(path, obj))
    print("Loaded image paths: ", end="")
    print(img_paths)
    return img_paths

def load_current_model_data() -> None:
    global current_model_obj_names, current_model_img_path, current_model_img_paths, current_model_layers, current_model_weights_path, current_model_biases_path
    # Open the model file, read the data, and store it in variables
    with open(models_dir + "models.json", 'r') as file:
        data = json.load(file)

    current_model_obj_names = data[current_model_name]["objNames"]
    current_model_img_path = data[current_model_name]["imgPath"]
    current_model_layers = data[current_model_name]["layers"]
    current_model_weights_path = data[current_model_name]["weightsPath"]
    current_model_biases_path = data[current_model_name]["biasesPath"]

    train_select_image_path(current_model_img_path)
    current_model_img_paths = get_obj_to_img_directories(current_model_img_path)

    bindings.loadModel(current_model_layers, current_model_weights_path, current_model_biases_path)

    print("Loaded data of model: " + current_model_name)

def load_current_model_labels() -> None:
    global model_obj_labels, train_base_obj_labels, train_best_obj_labels
    model_output_size = len(current_model_obj_names)

    # Delete previous model labels
    delete_model_labels()
    
    model_step = (model_obj_labels_range[1] - model_obj_labels_range[0]) / (model_output_size)
    train_base_step = (train_base_obj_labels_range[1] - train_base_obj_labels_range[0]) / (model_output_size)
    train_best_step = (train_best_obj_labels_range[1] - train_best_obj_labels_range[0]) / (model_output_size)
    min_step = 0.10
    model_step = model_step if model_step < min_step else min_step
    train_base_step = train_base_step if train_base_step < min_step else min_step
    train_best_step = train_best_step if train_best_step < min_step else min_step

    for i in range(model_output_size):
        model_obj_labels.append(ctk.CTkLabel(master=tabview.tab(use_model_tab_name), text = f"{current_model_obj_names[i]}: ___%", font=medium_font))
        model_obj_labels[-1].place(relx = model_obj_labels_x, rely = model_obj_labels_range[0] + model_step * i, anchor=ctk.CENTER)

        train_base_obj_labels.append(ctk.CTkLabel(master=tabview.tab(train_model_tab_name), text = f"{current_model_obj_names[i]} accuracy: ___%", font=small_font))
        train_base_obj_labels[-1].place(relx = train_base_obj_labels_x, rely = train_base_obj_labels_range[0] + train_base_step * i, anchor=ctk.CENTER)

        train_best_obj_labels.append(ctk.CTkLabel(master=tabview.tab(train_model_tab_name), text = f"{current_model_obj_names[i]} accuracy: ___%", font=small_font))
        train_best_obj_labels[-1].place(relx = train_best_obj_labels_x, rely = train_best_obj_labels_range[0] + train_best_step * i, anchor=ctk.CENTER)

def delete_model_labels():
    if model_obj_labels:
        for label in model_obj_labels:
            label.destroy()
        model_obj_labels.clear()
    if train_base_obj_labels:
        for label in train_base_obj_labels:
            label.destroy()
        train_base_obj_labels.clear()
    if train_best_obj_labels:
        for label in train_best_obj_labels:
            label.destroy()
        train_best_obj_labels.clear()

def update_default_image_path() -> None:
    if current_model_img_path:
        with open(models_dir + "models.json", 'r') as file:
            data = json.load(file)
        
        data[current_model_name]["imgPath"] = current_model_img_path

        with open (models_dir + "models.json", 'w') as file:
            json.dump(data, file, indent=2)
    else:
        print("Tried to set the default image path to null, path did not change")


# MODEL TAB FUNCTIONS
def model_optionmenu_func(choice) -> None:
    global current_model_name, current_image_path
    current_model_name = choice
    load_current_model_data()
    load_current_model_labels()

    if current_image_path:
        # Check if the new model can process the image
        image = Image.open(current_image_path)
        width, height = image.size
        if (width * height == current_model_layers[0]):
            # The model can process the image
            process_image(current_image_path)
        else:
            # The model cannot process the image
            model_image_label.configure(text="No image selected", image="")
            current_image_path = None

    print("Switched to model: " + choice)
    
def model_select_image() -> None:
    global current_image_path, model_image_label

    if not current_model_name:
        messagebox.showerror("Cannot select image", "Please, select a model before selecting an image.")
        return

    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Images", ("*.png", "*.jpg"))])
    
    if file_path:
        current_image_path = file_path
        original_image = Image.open(current_image_path)
        scaled_image = original_image.resize((model_img_res_x * scale, model_img_res_y * scale), Image.NEAREST)
        img = ctk.CTkImage(light_image=scaled_image, dark_image=scaled_image, size=(model_img_res_x, model_img_res_y))
        model_image_label.configure(text="", image=img)
        print("Loaded image: " + file_path)

        process_image(current_image_path)

def process_image(current_image_path: str) -> None:
    # Get the model answer and set the percentages accordingly
    model_answer = bindings.getModelAnswer(current_image_path, False)
    print("Model answer: ", end="")
    print(model_answer)
    set_model_answer(model_answer)

def set_model_answer(model_answer: list[float]):
    # Find the highest element's index and the total sum of the answers
    answer_sum: float = 0
    highest: float = -1
    highest_index: int = -1

    for i in range(len(model_answer)):
        answer_sum += model_answer[i]
        if model_answer[i] > highest:
            highest = model_answer[i]
            highest_index = i
    
    # Update the prediction labels with the model answer
    for i in range(len(model_answer)):
        if i != highest_index:
            model_obj_labels[i].configure(text = current_model_obj_names[i] + ": " + str(round(model_answer[i] / answer_sum * 100, 2)) + "%")
        else:
            model_obj_labels[i].configure(text = "> " + current_model_obj_names[i] + ": " + str(round(model_answer[i] / answer_sum * 100, 2)) + "%")


# TRAIN TAB FUNCTIONS
def train_select_image_path_ask() -> None:
    file_path = filedialog.askdirectory(title="Select the folder containing the training data directories")
    if file_path:
        train_select_image_path(file_path)

def train_select_image_path(file_path) -> None:
    global current_model_img_path, current_model_img_paths
    current_model_img_path = file_path
    current_model_img_paths = get_obj_to_img_directories(current_model_img_path)
    update_default_image_path()
    train_obj_images_label.configure(text=current_model_img_path.split('/')[-1])

def train_model_loop() -> None:
    global current_model_obj_names, current_model_img_paths
    bindings.loadModel(current_model_layers, current_model_weights_path, current_model_biases_path)
    bindings.initializeTrainer(models_dir + current_model_name + "/training", weights_mutation_range, biases_mutation_range, threads)
    print("Images: ", end=" ")
    print(current_model_img_paths)
    bindings.initializeCache(current_model_img_paths)
    generations: int = int(train_gen_entry.get())
    update_gen_label(0, generations)
    update_training_model_information(True)
    train_gen_entry.place_forget()
    train_gen_label.place(relx=0.25, rely=0.86, anchor=ctk.CENTER)
    time.sleep(0.1)

    for i in range(generations):
        save_to_disk = False

        if (i + 1) % save_to_disk_interval == 0 or i + 1 == generations:
            save_to_disk = True

        bindings.trainModel(current_model_obj_names, current_model_img_paths,
                            mutation_center_offset_range, i + 1, save_to_disk, True, True, image_limit)

        if (i + 1) % update_model_accuracy_interval == 0 or i + 1 == generations:
            update_training_model_information(False)
        
        if (i + 1) % update_gen_label_interval == 0:
            update_gen_label(i + 1, generations)
            time.sleep(0.1)
    
    train_gen_entry.place(relx=0.25, rely=0.85, anchor=ctk.CENTER)
    train_gen_label.place_forget()

def update_training_model_information(also_set_to_base: bool = False) -> None:
    accuracy_string: str = bindings.getAccuracyString(current_model_obj_names, current_model_img_paths, image_limit)
    split_accuracy_string: list[str] = accuracy_string.split(" | ")
    obj_accuracy_list: list[float] = [float(el.split(": ")[1][:-1]) for el in split_accuracy_string[:-1]]
    general_accuracy: float = float(split_accuracy_string[-1].split(": ")[1][:-1])

    for i in range(len(obj_accuracy_list)):
        train_best_obj_labels[i].configure(text = f"{current_model_obj_names[i]} accuracy: {round(obj_accuracy_list[i], 4)}%")
    train_best_general_label.configure(text = f"General accuracy: {round(general_accuracy, 4)}%")

    if also_set_to_base:
        for i in range(len(obj_accuracy_list)):
            train_base_obj_labels[i].configure(text = f"{current_model_obj_names[i]} accuracy: {round(obj_accuracy_list[i], 4)}%")
        train_base_general_label.configure(text = f"General accuracy: {round(general_accuracy, 4)}%")

def update_gen_label(gen: int, max_gen: int) -> None:
    global train_gen_label
    train_gen_label.configure(text = str(gen) + " / " + str(max_gen))

def train_model() -> None:
    global train_gen_entry

    if not current_model_name:
        messagebox.showerror("Couldn't start training", "Training cannot start, as there is no model selected.")
        return
    
    if not current_model_img_path:
        messagebox.showerror("Couldn't start training", "Training cannot start, as no image folder has been selected.")
        return

    # Check if the train_gen_entry input is correct
    if not train_gen_entry.get().isdigit() or not int(train_gen_entry.get()) >= 1:
        messagebox.showerror("Couldn't start training", "Please, enter a valid generation number.")
        return

    print("Deleting previous training files")
    for filename in os.listdir(models_dir + current_model_name + "/training/"):
        file_path = os.path.join(models_dir + current_model_name + "/training/", filename)
        if (os.path.isfile(file_path)):
            os.unlink(file_path)

    print("Starting model training")
    t = Thread(target=train_model_loop)
    t.start()

def get_latest_model_paths(path):
    paths: list[str] = []
    latest_weight_file = None
    latest_weight_file_gen = -1

    it: iter = os.scandir(path)
    weights_path_list: list[str] = [entry.name for entry in it if entry.is_file() and entry.name.endswith(".weights")]

    for weight_path in weights_path_list:
        num: int = int(weight_path[3:weight_path.index(".weights")])

        if num > latest_weight_file_gen:
            latest_weight_file_gen = num
            latest_weight_file = weight_path
    
    paths.append(models_dir + current_model_name + "/training/" + latest_weight_file)
    paths.append(models_dir + current_model_name + "/training/" + latest_weight_file.replace(".weights", ".bias"))

    return paths

def update_to_latest_model() -> None:
    model_paths = get_latest_model_paths(models_dir + current_model_name + "/training")
    # Copy the weights and biases to the model folder
    shutil.move(model_paths[0], models_dir + current_model_name + "/default.weights")
    shutil.move(model_paths[1], models_dir + current_model_name + "/default.bias")

    # Update the training labels to reflect the base model change
    for i in range(len(train_base_obj_labels)):
        train_base_obj_labels[i].configure(text=train_best_obj_labels[i].cget("text"))

    train_base_general_label.configure(text=train_best_general_label.cget("text"))

    load_current_model_data()


# MANAGE TAB FUNCTIONS
def update_optionmenus() -> None:
    # Update OptionMenus model lists
    model_optionmenu_var.set("Select Model")
    model_name_list = get_model_name_list()
    model_model_optionmenu.configure(values=model_name_list)
    train_model_optionmenu.configure(values=model_name_list)
    manage_model_optionmenu.configure(values=model_name_list)

def create_model() -> None:
    global manage_name_entry, manage_obj_textbox, manage_layers_entry, model_being_created_img_path

    model_name = manage_name_entry.get()
    model_obj_names_str: str = manage_obj_textbox.get("0.0", "end").strip()
    layers_str: str = manage_layers_entry.get()

    if not (model_name and model_being_created_img_path and model_obj_names_str and layers_str):
        messagebox.showerror("Couldn't create model", "The model could not be created. Please, fill every required field.")
        return

    # Convert the layer string to a list of ints
    layers_list_str: list[str] = re.split(r',\s*', layers_str)
    layers_list: list[int] = [int(s) for s in layers_list_str]

    if len(layers_list) < 2:
        messagebox.showerror("Couldn't create model", "A model with less than 2 layers cannot be created.")
        return

    # Process the data
    model_obj_names = re.split(r',\s*', model_obj_names_str)
    weightsPath = models_dir + model_name + "/default.weights"
    biasesPath = models_dir + model_name + "/default.bias"

    if len(model_obj_names) < 2:
        messagebox.showerror("Couldn't create model", "A model with less than 2 objects cannot be created.")
        return

    # Open the model file and read the data
    with open(models_dir + "models.json", 'r') as file:
        data = json.load(file)
    
    if model_name in data:
        messagebox.showerror("Couldn't create model", "A model with that name already exists.")
        return

    data[model_name] = {
        "objNames": model_obj_names,
        "imgPath": model_being_created_img_path,
        "layers": layers_list,
        "weightsPath": weightsPath,
        "biasesPath": biasesPath
    }

    # Add the data of the new model to the json file
    with open(models_dir + "models.json", 'w') as file:
        json.dump(data, file, indent=2)
    
    # Create model folders
    os.makedirs(os.path.join(models_dir, model_name, "training"))

    # Initialize the model with random weights and biases
    bindings.initializeModelFiles(layers_list, weightsPath, biasesPath)

    update_optionmenus()

def ask_delete_model() -> None:
    global model_optionmenu_var, model_model_optionmenu, train_model_optionmenu
    model_to_delete = model_optionmenu_var.get()

    # Select Model is the placeholder used for when there is no model selected. The delete button
    # should not try to delete anything when the model name is "Select Model".
    if model_to_delete == "Select Model":
        return

    response = messagebox.askyesno("Delete model", f"You are about to permanently delete \"{model_to_delete}\". Do you want to proceed?")
    
    if response:
        # Remove the model's folder
        shutil.rmtree(os.path.join(models_dir, model_to_delete))
        
        # Remove the model from models.json
        with open(os.path.join(models_dir, "models.json")) as file:
            data = json.load(file)
        
        del data[model_to_delete]

        # Write the updated json data to models.json
        with open(os.path.join(models_dir, "models.json"), 'w') as file:
            json.dump(data, file, indent=2)
        
        update_optionmenus()

        # Set model tab labels to their default values
        delete_model_labels()

def select_default_image_path() -> None:
    global model_being_created_img_path, manage_select_img_label
    file_path = filedialog.askdirectory(title="Select the folder containing the training data directories")
    if file_path:
        model_being_created_img_path = file_path
        manage_select_img_label.configure(text=model_being_created_img_path.split('/')[-1])


# MODEL TAB CODE
model_optionmenu_var = ctk.StringVar(value="Select Model")
model_model_optionmenu = ctk.CTkOptionMenu(master=tabview.tab(use_model_tab_name), width=260, height=34, font=small_font, values=get_model_name_list(), command=model_optionmenu_func, variable=model_optionmenu_var)
model_model_optionmenu.place(relx=0.172, rely=0.92, anchor=ctk.CENTER)

model_select_image_button = ctk.CTkButton(master=tabview.tab(use_model_tab_name), width=150, height=34, text="Select Image", font=small_font, command=model_select_image)
model_select_image_button.place(relx=0.41, rely=0.92, anchor=ctk.CENTER)

model_image_label = ctk.CTkLabel(master=tabview.tab(use_model_tab_name), text="No image selected", font=small_font)
model_image_label.place(relx=model_img_rel_pos_x, rely=model_img_rel_pos_y, anchor=ctk.CENTER)

model_predictions_label = ctk.CTkLabel(master=tabview.tab(use_model_tab_name), text="MODEL PREDICTIONS", font=large_underlined_font)
model_predictions_label.place(relx=0.756, rely=0.12, anchor=ctk.CENTER)


# TRAIN TAB CODE
train_model_optionmenu = ctk.CTkOptionMenu(master=tabview.tab(train_model_tab_name), width=340, height=34, font=small_font, values=get_model_name_list(), command=model_optionmenu_func, variable=model_optionmenu_var)
train_model_optionmenu.place(relx=0.25, rely=0.06, anchor=ctk.CENTER)

train_select_obj_images_button = ctk.CTkButton(master=tabview.tab(train_model_tab_name), width=150, height=34, text="Select training images", font=small_font, command=train_select_image_path_ask)
train_select_obj_images_button.place(relx=0.6, rely=0.06, anchor=ctk.CENTER)

train_obj_images_label = ctk.CTkLabel(master=tabview.tab(train_model_tab_name), text="None selected", font=smaller_font, text_color="gray", height=10)
train_obj_images_label.place(relx=0.6, rely=0.12, anchor=ctk.CENTER)


train_base_model_label = ctk.CTkLabel(master=tabview.tab(train_model_tab_name), text="Base Model", font=medium_large_underlined_font)
train_base_model_label.place(relx=0.25, rely=0.18, anchor=ctk.CENTER)

train_base_general_label = ctk.CTkLabel(master=tabview.tab(train_model_tab_name), text="General accuracy: ___%", font=medium_font)
train_base_general_label.place(relx=0.25, rely=0.75, anchor=ctk.CENTER)

train_best_model_label = ctk.CTkLabel(master=tabview.tab(train_model_tab_name), text="Best Model", font=medium_large_underlined_font)
train_best_model_label.place(relx=0.75, rely=0.18, anchor=ctk.CENTER)

train_best_general_label = ctk.CTkLabel(master=tabview.tab(train_model_tab_name), text="General accuracy: ___%", font=medium_font)
train_best_general_label.place(relx=0.75, rely=0.75, anchor=ctk.CENTER)


train_button = ctk.CTkButton(master=tabview.tab(train_model_tab_name), width=120, height=34, text="Train", font=medium_font, command=train_model)
train_button.place(relx=0.25, rely=0.92, anchor=ctk.CENTER)

train_gen_label = ctk.CTkLabel(master=tabview.tab(train_model_tab_name), text="", font=smaller_font, text_color="gray", height=10)

train_gen_entry = ctk.CTkEntry(master=tabview.tab(train_model_tab_name), placeholder_text="Generations", width = 120, font=small_font, justify="center")
train_gen_entry.place(relx=0.25, rely=0.85, anchor=ctk.CENTER)

train_update_to_best_button = ctk.CTkButton(master=tabview.tab(train_model_tab_name), width=120, height=34, text="Update base model to latest", font=medium_font, command=update_to_latest_model)
train_update_to_best_button.place(relx=0.672, rely=0.92, anchor=ctk.CENTER)


# MANAGE MODELS TAB CODE
manage_add_model_label = ctk.CTkLabel(master=tabview.tab(manage_models_tab_name), text="CREATE A MODEL", font=large_underlined_font)
manage_add_model_label.place(relx=0.25, rely=0.1, anchor=ctk.CENTER)

manage_name_entry = ctk.CTkEntry(master=tabview.tab(manage_models_tab_name), placeholder_text="Model name", width = 200, font=small_font, justify="center")
manage_name_entry.place(relx=0.25, rely=0.2, anchor=ctk.CENTER)

manage_select_img_dir_button = ctk.CTkButton(master=tabview.tab(manage_models_tab_name), width=120, height=34, text="Select default image folder", font=medium_font, command=select_default_image_path)
manage_select_img_dir_button.place(relx=0.25, rely=0.31, anchor=ctk.CENTER)

manage_select_img_label = ctk.CTkLabel(master=tabview.tab(manage_models_tab_name), text="None selected", font=smaller_font, text_color="gray", height=10)
manage_select_img_label.place(relx=0.25, rely=0.37, anchor=ctk.CENTER)

manage_obj_label = ctk.CTkLabel(master=tabview.tab(manage_models_tab_name), text="Object names (separated by ',')", font=small_font)
manage_obj_label.place(relx=0.25, rely=0.44, anchor=ctk.CENTER)

manage_obj_textbox = ctk.CTkTextbox(master=tabview.tab(manage_models_tab_name), width = 260, height = 120, font=small_font, wrap="word")
manage_obj_textbox.place(relx=0.25, rely=0.6, anchor=ctk.CENTER)

manage_layers_entry = ctk.CTkEntry(master=tabview.tab(manage_models_tab_name), placeholder_text="Model layers (separated by ',')", width = 260, font=small_font, justify="center")
manage_layers_entry.place(relx=0.25, rely=0.79, anchor=ctk.CENTER)

manage_add_model_button = ctk.CTkButton(master=tabview.tab(manage_models_tab_name), width=120, height=34, text="Create model", font=medium_font, command=create_model)
manage_add_model_button.place(relx=0.25, rely=0.89, anchor=ctk.CENTER)

manage_remove_model_label = ctk.CTkLabel(master=tabview.tab(manage_models_tab_name), text="REMOVE A MODEL", font=large_underlined_font)
manage_remove_model_label.place(relx=0.75, rely=0.1, anchor=ctk.CENTER)

manage_model_optionmenu = ctk.CTkOptionMenu(master=tabview.tab(manage_models_tab_name), width=340, height=34, font=small_font, values=get_model_name_list(), command=model_optionmenu_func, variable=model_optionmenu_var)
manage_model_optionmenu.place(relx=0.75, rely=0.25, anchor=ctk.CENTER)

manage_delete_model_button = ctk.CTkButton(master=tabview.tab(manage_models_tab_name), width=120, height=34, text="Delete model", font=medium_font, command=ask_delete_model)
manage_delete_model_button.place(relx=0.75, rely=0.43, anchor=ctk.CENTER)


app.mainloop()
