import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import bindings
import os
import json
from threading import Thread
import time

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

use_model_tab_name: str = "  Use Model  "
train_model_tab_name: str = "  Train Model  "
model_info_tab_name: str = "  Model Information  "
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
tabview.add(model_info_tab_name)
tabview.set(train_model_tab_name)


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
    generations: int = 5

    for i in range(generations):
        bindings.trainModel(current_model_obj1_name, current_model_obj1_image_path, current_model_obj2_name,
                            current_model_obj2_image_path, 0.15, 5, i + 1, (i + 1) % 20 == 0, True, -1)
        update_gen_label(i + 1, generations)

def update_gen_label(gen: int, max_gen: int) -> None:
    global train_gen_label
    train_gen_label.configure(text = str(gen) + " / " + str(max_gen))

def train_model() -> None:
    print("Starting model training")
    t = Thread(target=train_model_loop)
    t.start()


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
train_button.place(relx=0.172, rely=0.92, anchor=ctk.CENTER)

train_gen_label = ctk.CTkLabel(master=tabview.tab(train_model_tab_name), text="", font=smaller_font, text_color="gray", height=10)
train_gen_label.place(relx=0.172, rely=0.86, anchor=ctk.CENTER)


app.mainloop()