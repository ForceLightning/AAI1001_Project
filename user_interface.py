import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ctypes
import pandas as pd

#boolean, True - show layout 1, False - show layout 2 
b_toggle_layout = True

def init_window():
     #init window
    window = tk.Tk()
    window.title("Electrocardiogram")

    layout_1 = tk.LabelFrame(window, text = "Layout 1 (Landing)", padx = 5, pady = 5)
    layout_1.pack()
    layout_2 = tk.LabelFrame(window, text = "Layout 2 (Prediction)", padx = 5, pady = 5)
    
    return window, layout_1 , layout_2

def toggle_layout():
    #get global boolean
    global b_toggle_layout
    #swap truth value
    b_toggle_layout = not b_toggle_layout 
    
    #change layout accordingly
    if b_toggle_layout is True:
        layout_1.pack()
        layout_2.pack_forget()
    else:
        layout_1.pack_forget()
        layout_2.pack()




def init_layout_1(parent):
    #label - top centre of the window
    label = tk.Label(parent, text = "Electrocardiogram")
    label.config(font=("Courier", 32))
    label.pack()

    #buttons 
    btn_select = tk.Button(parent, text="Select CSV File", command=process_data)
    btn_select.pack(pady=20)

    # Setting of Screen Width
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)
    window.geometry(f"{window_width}x{window_height}")
    input_frame = ttk.Frame(parent)
    input_frame.pack(padx=10, pady=10)


    #THIS PART IS FOR TESTING ONLY (TO SEE IF GRAPH CAN BE SHOWN)
    data_label = ttk.Label(input_frame, text="Enter data (comma-separated):")
    data_label.pack(side=tk.LEFT)
    data_entry = ttk.Entry(input_frame, width=30)
    data_entry.pack(side=tk.LEFT)
    plot_button = ttk.Button(input_frame, text="Plot Graph", command=process_data)
    plot_button.pack(side=tk.LEFT)


    return data_entry

def init_layout_2(parent):
    #label - top centre of the window
    label = tk.Label(parent, text = "Electrocardiogram Result")
    label.config(font=("Courier", 32))
    label.pack()

    
    # Where the graph will appear
    graph_label = tk.Label(parent, text = "Here is where the graph will appear")
    graph_label.pack()

    #holder in hireachy 
    pred_holder = ttk.Frame(parent)
    pred_holder.pack()

    #children
    graph_frame = ttk.Frame(pred_holder)
    #graph_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    graph_frame.grid(row=1,column=0)

    text_prediction = tk.Text(pred_holder)
    text_prediction.insert(tk.END, "Your heartbeat was predicted to have contained : {}-type heartbeats, at a {}% accuracy.".format("REPLACE THIS WITH PREDICTED HEARTBEAT", "arbitrary number"))
    #text_prediction.pack()
    text_prediction.grid(row=1,column=1)

    #TODO : insert a table showing all predictions and all accuracies

    #button to return to previous page
    return_btn = tk.Button(parent ,text="Return", command=toggle_layout)
    return_btn.pack()

    return graph_frame


def plot_graph(data):
    toggle_layout()

    x_values = list(range(1, len(data) + 1))
    y_values = data

    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, marker='o')
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Time')
    ax.set_title('Electrocardiogram Results')

    
    for widget in graph_frame.winfo_children():
        widget.destroy()

   
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def process_data(): 
    
    ##########This is the part that reads the csv data but I am not sure which value to take for plotting##############
    # file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    # if not file_path:
    #     return

    # try:
    #     df = pd.read_csv(file_path)
    #     if 'Value' in df.columns:  
    #         data_list = df['Value'].tolist()
    #         plot_graph(data_list)
    #     else:
    #         print("Error: CSV file must contain a column named 'Value' with the data.")
    # except Exception as e:
    #     print("Error reading the CSV file:", e)

    data = data_entry.get()
    try:
        data_list = [float(x) for x in data.split(',')]
        plot_graph(data_list)
    except ValueError:
        print("Invalid input. Please enter a comma-separated list of numbers.")


if __name__ == "__main__":
    window, layout_1, layout_2 = init_window()
    data_entry = init_layout_1(layout_1)
    graph_frame = init_layout_2(layout_2)

    #TODO : load model here
    
    #TODO : call predict in/ together with plotgraph, and change layout_2 accordingly

    window.mainloop()
