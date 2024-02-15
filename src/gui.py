import customtkinter as ctk

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
ctk.set_window_scaling(2)
ctk.set_widget_scaling(2)

model_tab_name = "Model"
train_tab_name = "Train"

app = ctk.CTk()
app.geometry("800x600")

tabview = ctk.CTkTabview(master=app)
tabview.add(model_tab_name)
tabview.add(train_tab_name)
tabview.set(model_tab_name)

def button_function():
    print("button pressed")

button = ctk.CTkButton(master=tabview.tab(model_tab_name), text="Epic button", command=button_function)
button.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)

app.mainloop()