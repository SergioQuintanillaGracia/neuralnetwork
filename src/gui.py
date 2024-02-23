import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import bindings
import os
import json
from threading import Thread
import shutil

# Configure scaling and theme
scale = 2
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")
ctk.set_window_scaling(scale)
ctk.set_widget_scaling(scale)

# Create and configure the window
app = ctk.CTk()
app.geometry("940x560")
app.title("NeuralNetwork GUI")

# Variables
models_dir = "./networks/GUI_models/"
current_model_name: str = None
current_model_obj1_name: str = None
current_model_obj2_name: str = None
current_model_layers: list[int] = None
current_model_weights_path: str = None
current_model_biases_path: str = None
current_model_obj1_image_path: str = None
current_model_obj2_image_path: str = None

model_image_label: str = None
model_obj_1_label: str = None
model_obj_2_label: str = None
train_obj1_images_label = None
train_obj2_images_label = None
train_gen_label = None
train_gen_entry = None
manage_name_entry = None
manage_obj1_entry = None
manage_obj2_entry = None
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
tabview.set(manage_models_tab_name)


# NEURAL NETWORK FUNCTIONS (temporary at the moment)
def get_model_name_list() -> list[str]:
    it: iter = os.scandir(models_dir)
    dir_list: list[str] = [entry.name for entry in it if entry.is_dir()]
    return dir_list

def load_current_model_data() -> None:
    global current_model_obj1_name, current_model_obj2_name, current_model_layers, current_model_weights_path, current_model_biases_path
    # Open the model file, read the data, and store it in variables
    with open(models_dir + "models.json", 'r') as file:
        data = json.load(file)
    
    current_model_obj1_name = data[current_model_name]["obj1Name"]
    current_model_obj2_name = data[current_model_name]["obj2Name"]
    current_model_layers = data[current_model_name]["layers"]
    current_model_weights_path = models_dir + data[current_model_name]["weightsPath"]
    current_model_biases_path = models_dir + data[current_model_name]["biasesPath"]

    bindings.loadModel(current_model_layers, current_model_weights_path, current_model_biases_path)
    
    print("Loaded data of model: " + current_model_name)

# MODEL TAB FUNCTIONS
def model_optionmenu_func(choice) -> None:
    global current_model_name, model_obj_1_label, model_obj_2_label
    current_model_name = choice
    load_current_model_data()
    model_obj_1_label.configure(text = current_model_obj1_name + ": ___%")
    model_obj_2_label.configure(text = current_model_obj2_name + ": ___%")
    print("Switched to model: " + choice)
    
def model_select_image() -> None:
    global current_image_path, model_image_label, model_obj_1_label, model_obj_2_label
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Images", ("*.png", "*.jpg"))])
    current_image_path = file_path
    original_image = Image.open(current_image_path)
    scaled_image = original_image.resize((model_img_res_x * scale, model_img_res_y * scale), Image.NEAREST)
    img = ctk.CTkImage(light_image=scaled_image, dark_image=scaled_image, size=(model_img_res_x, model_img_res_y))
    model_image_label.configure(text="", image=img)
    print("Loaded image: " + file_path)

    # Get the model answer and set the percentages accordingly
    model_answer = bindings.getModelAnswer(current_image_path, False)
    model_obj_1_label.configure(text = current_model_obj1_name + ": " + str(round((1 - model_answer[0]) * 100, 2)) + "%")
    model_obj_2_label.configure(text = current_model_obj2_name + ": " + str(round(model_answer[0] * 100, 2)) + "%")


# TRAIN TAB FUNCTIONS
def train_select_obj1_image_path() -> None:
    global current_model_obj1_image_path
    file_path = filedialog.askdirectory(title="Select object 1 training image folder")
    current_model_obj1_image_path = file_path
    train_obj1_images_label.configure(text=current_model_obj1_image_path.split('/')[-1])

def train_select_obj2_image_path() -> None:
    global current_model_obj2_image_path
    file_path = filedialog.askdirectory(title="Select object 2 training image folder")
    current_model_obj2_image_path = file_path
    train_obj2_images_label.configure(text=current_model_obj2_image_path.split('/')[-1])

def train_model_loop() -> None:
    global current_model_obj1_name, current_model_obj1_image_path, current_model_obj2_name, current_model_obj2_image_path
    bindings.loadModel(current_model_layers, current_model_weights_path, current_model_biases_path)
    bindings.initializeTrainer(models_dir + current_model_name + "/training", 0, 0.1, 18)
    bindings.initializeCache(current_model_obj1_image_path, current_model_obj2_image_path)
    generations: int = int(train_gen_entry.get())
    train_gen_entry.place_forget()
    train_gen_label.place(relx=0.25, rely=0.86, anchor=ctk.CENTER)

    for i in range(generations):
        save_to_disk = False

        if (i + 1) % 1000 == 0 or i + 1 == generations:
            save_to_disk = True
            update_gen_label(i + 1, generations)

        bindings.trainModel(current_model_obj1_name, current_model_obj1_image_path, current_model_obj2_name,
                            current_model_obj2_image_path, 0.2, 5, i + 1, save_to_disk, True, -1)
    
    train_gen_entry.place(relx=0.25, rely=0.85, anchor=ctk.CENTER)
    train_gen_label.place_forget()

def update_gen_label(gen: int, max_gen: int) -> None:
    global train_gen_label
    train_gen_label.configure(text = str(gen) + " / " + str(max_gen))

def train_model() -> None:
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


# MANAGE TAB FUNCTIONS
def create_model() -> None:
    global manage_name_entry, manage_obj1_entry, manage_obj2_entry, manage_layers_entry
    # Open the model file and read the data
    with open(models_dir + "models.json", 'r') as file:
        data = json.load(file)
    
    layers_list_str: list[str] = current_model_layers.get().split(",")
    layers_list: list[int] = [int(s) for s in layers_list_str]

    data[manage_name_entry.get()] = {
        "obj1Name": manage_obj1_entry.get(),
        "obj2Name": manage_obj2_entry.get(),
        "layers": layers_list,
        "weightsPath": models_dir + manage_name_entry.get() + "/default.weights",
        "biasesPath": models_dir + manage_name_entry.get() + "/default.bias"
    }

    


# MODEL TAB CODE
model_optionmenu_default_opt = ctk.StringVar(value="Select Model")
model_model_optionmenu = ctk.CTkOptionMenu(master=tabview.tab(use_model_tab_name), width=260, height=34, font=small_font, values=get_model_name_list(), command=model_optionmenu_func, variable=model_optionmenu_default_opt)
model_model_optionmenu.place(relx=0.172, rely=0.92, anchor=ctk.CENTER)

model_select_image_button = ctk.CTkButton(master=tabview.tab(use_model_tab_name), width=150, height=34, text="Select Image", font=small_font, command=model_select_image)
model_select_image_button.place(relx=0.41, rely=0.92, anchor=ctk.CENTER)

model_image_label = ctk.CTkLabel(master=tabview.tab(use_model_tab_name), text="No image selected")
model_image_label.place(relx=model_img_rel_pos_x, rely=model_img_rel_pos_y, anchor=ctk.CENTER)

model_predictions_label = ctk.CTkLabel(master=tabview.tab(use_model_tab_name), text="MODEL PREDICTIONS", font=large_underlined_font)
model_predictions_label.place(relx=0.756, rely=0.2, anchor=ctk.CENTER)

model_obj_1_label = ctk.CTkLabel(master=tabview.tab(use_model_tab_name), text="Object 1: ___%", font=medium_font)
model_obj_1_label.place(relx=0.756, rely=0.36, anchor=ctk.CENTER)

model_obj_2_label = ctk.CTkLabel(master=tabview.tab(use_model_tab_name), text="Object 2: ___%", font=medium_font)
model_obj_2_label.place(relx=0.756, rely=0.5, anchor=ctk.CENTER)


# TRAIN TAB CODE
train_model_optionmenu = ctk.CTkOptionMenu(master=tabview.tab(train_model_tab_name), width=340, height=34, font=small_font, values=get_model_name_list(), command=model_optionmenu_func, variable=model_optionmenu_default_opt)
train_model_optionmenu.place(relx=0.25, rely=0.06, anchor=ctk.CENTER)

train_select_obj1_images_button = ctk.CTkButton(master=tabview.tab(train_model_tab_name), width=150, height=34, text="Select obj1 images", font=small_font, command=train_select_obj1_image_path)
train_select_obj1_images_button.place(relx=0.6, rely=0.06, anchor=ctk.CENTER)

train_obj1_images_label = ctk.CTkLabel(master=tabview.tab(train_model_tab_name), text="None selected", font=smaller_font, text_color="gray", height=10)
train_obj1_images_label.place(relx=0.6, rely=0.12, anchor=ctk.CENTER)

train_select_obj2_images_button = ctk.CTkButton(master=tabview.tab(train_model_tab_name), width=150, height=34, text="Select obj2 images", font=small_font, command=train_select_obj2_image_path)
train_select_obj2_images_button.place(relx=0.85, rely=0.06, anchor=ctk.CENTER)

train_obj2_images_label = ctk.CTkLabel(master=tabview.tab(train_model_tab_name), text="None selected", font=smaller_font, text_color="gray", height=10)
train_obj2_images_label.place(relx=0.85, rely=0.12, anchor=ctk.CENTER)

train_button = ctk.CTkButton(master=tabview.tab(train_model_tab_name), width=120, height=34, text="Train", font=medium_font, command=train_model)
train_button.place(relx=0.25, rely=0.92, anchor=ctk.CENTER)

train_gen_label = ctk.CTkLabel(master=tabview.tab(train_model_tab_name), text="", font=smaller_font, text_color="gray", height=10)

train_gen_entry = ctk.CTkEntry(master=tabview.tab(train_model_tab_name), placeholder_text="Generations", width = 100, font=small_font, justify="center")
train_gen_entry.place(relx=0.25, rely=0.85, anchor=ctk.CENTER)

train_update_to_best_button = ctk.CTkButton(master=tabview.tab(train_model_tab_name), width=120, height=34, text="Update base model to latest", font=medium_font, command=update_to_latest_model)
train_update_to_best_button.place(relx=0.672, rely=0.92, anchor=ctk.CENTER)

# MANAGE MODELS TAB CODE
manage_add_model_label = ctk.CTkLabel(master=tabview.tab(manage_models_tab_name), text="CREATE A MODEL", font=large_underlined_font)
manage_add_model_label.place(relx=0.25, rely=0.1, anchor=ctk.CENTER)

manage_name_entry = ctk.CTkEntry(master=tabview.tab(manage_models_tab_name), placeholder_text="Model name", width = 200, font=small_font, justify="center")
manage_name_entry.place(relx=0.25, rely=0.25, anchor=ctk.CENTER)

manage_obj1_entry = ctk.CTkEntry(master=tabview.tab(manage_models_tab_name), placeholder_text="Object 1 name", width = 200, font=small_font, justify="center")
manage_obj1_entry.place(relx=0.25, rely=0.39, anchor=ctk.CENTER)

manage_obj2_entry = ctk.CTkEntry(master=tabview.tab(manage_models_tab_name), placeholder_text="Object 2 name", width = 200, font=small_font, justify="center")
manage_obj2_entry.place(relx=0.25, rely=0.53, anchor=ctk.CENTER)

manage_layers_entry = ctk.CTkEntry(master=tabview.tab(manage_models_tab_name), placeholder_text="Model layers (separated by ',')", width = 240, font=small_font, justify="center")
manage_layers_entry.place(relx=0.25, rely=0.67, anchor=ctk.CENTER)

manage_add_model_button = ctk.CTkButton(master=tabview.tab(manage_models_tab_name), width=120, height=34, text="Create model", font=medium_font, command=create_model)
manage_add_model_button.place(relx=0.25, rely=0.85, anchor=ctk.CENTER)


manage_remove_model_label = ctk.CTkLabel(master=tabview.tab(manage_models_tab_name), text="REMOVE A MODEL", font=large_underlined_font)
manage_remove_model_label.place(relx=0.75, rely=0.1, anchor=ctk.CENTER)


app.mainloop()