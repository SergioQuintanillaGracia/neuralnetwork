import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import bindings
import os

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
current_model_name = None
current_model_obj1_name = "Circle"          # Set manually, until it can be automatically loaded
current_model_obj2_name = "Circumference"    # Set manually, until it can be automatically loaded

model_image_label = None
model_obj_1_label = None
model_obj_2_label = None

model_tab_name = "Model"
train_tab_name = "Train"
current_image_path = None
model_img_res_x = 380
model_img_res_y = 380
model_img_rel_pos_x = 0.263
model_img_rel_pos_y = 0.45

# Font definitions
large_font = ctk.CTkFont(family="", size=int(30/scale), weight="bold")
large_underlined_font = ctk.CTkFont(family="", size=int(30/scale), weight="bold", underline=True)
medium_font = ctk.CTkFont(family="", size=int(20/scale), weight="bold")
small_font = ctk.CTkFont(family="", size=int(15/scale), weight="bold")

# Define the tabview and add the tabs
tabview = ctk.CTkTabview(master=app, corner_radius=10, border_width=0)
tabview._segmented_button.configure(font=medium_font)
tabview.pack(expand=1, fill="both")
tabview.add(model_tab_name)
tabview.add(train_tab_name)
tabview.set(model_tab_name)

# Load the model (temporary, until models can be selected)
bindings.loadModel([256, 96, 48, 1], "./networks/circles_circumf_16x16/256_96_48_1/progress/2881.4_3000.weights", "./networks/circles_circumf_16x16/256_96_48_1/progress/2881.4_3000.bias")

# NEURAL NETWORK FUNCTIONS (temporary at the moment)
def get_model_name_list():
    return ["model1 (placeholder)", "model2 (placeholder)", "model 3 (placeholder)", "model4 (placeholder)"]

def load_current_model_data():
    # Open the model file, read the data, and store it in variables
    print("Loaded data of model: " + current_model)

# MODEL TAB FUNCTIONS
def model_optionmenu_func(choice):
    global current_model
    current_model = choice
    load_current_model_data()
    print("Switched to model: " + choice)
    

def model_select_image():
    global model_image_label, model_obj_1_label, model_obj_2_label
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Images", ("*.png", "*.jpg"))])
    current_image = file_path
    original_image = Image.open(current_image)
    scaled_image = original_image.resize((model_img_res_x * scale, model_img_res_y * scale), Image.NEAREST)
    img = ctk.CTkImage(light_image=scaled_image, dark_image=scaled_image, size=(model_img_res_x, model_img_res_y))
    model_image_label.destroy()
    model_image_label = ctk.CTkLabel(master=tabview.tab(model_tab_name), text="", image=img)
    model_image_label.place(relx=model_img_rel_pos_x, rely=model_img_rel_pos_y, anchor=ctk.CENTER)
    print("Loaded image: " + file_path)

    # Get the model answer and set the percentages accordingly
    model_answer = bindings.getModelAnswer(current_image, False)
    model_obj_1_label.configure(text = current_model_obj1_name + ": " + str(round((1 - model_answer[0]) * 100, 2)) + "%")
    model_obj_2_label.configure(text = current_model_obj2_name + ": " + str(round(model_answer[0] * 100, 2)) + "%")


# MODEL TAB CODE
model_optionmenu_default_opt = ctk.StringVar(value="Select Model")
model_optionmenu = ctk.CTkOptionMenu(master=tabview.tab(model_tab_name), width=260, height=34, font=small_font, values=get_model_name_list(), command=model_optionmenu_func, variable=model_optionmenu_default_opt)
model_optionmenu.place(relx=0.172, rely=0.92, anchor=ctk.CENTER)

model_select_image_button = ctk.CTkButton(master=tabview.tab(model_tab_name), width=150, height=34, text="Select Image", font=small_font, command=model_select_image)
model_select_image_button.place(relx=0.41, rely=0.92, anchor=ctk.CENTER)

model_image_label = ctk.CTkLabel(master=tabview.tab(model_tab_name), text="No image selected")
model_image_label.place(relx=model_img_rel_pos_x, rely=model_img_rel_pos_y, anchor=ctk.CENTER)

model_predictions_label = ctk.CTkLabel(master=tabview.tab(model_tab_name), text="MODEL PREDICTIONS", font=large_underlined_font)
model_predictions_label.place(relx=0.756, rely=0.2, anchor=ctk.CENTER)

model_obj_1_label = ctk.CTkLabel(master=tabview.tab(model_tab_name), text="Object 1: ___%", font=medium_font)
model_obj_1_label.place(relx=0.756, rely=0.36, anchor=ctk.CENTER)

model_obj_2_label = ctk.CTkLabel(master=tabview.tab(model_tab_name), text="Object 2: ___%", font=medium_font)

model_obj_2_label.place(relx=0.756, rely=0.5, anchor=ctk.CENTER)


# TRAIN TAB FUNCTIONS



# TRAIN TAB CODE



app.mainloop()