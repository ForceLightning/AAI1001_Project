import ctypes
from functools import partial
from typing import Tuple

import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection

import numpy as np
import wfdb
from wfdb.processing import gqrs_detect
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.utils import *
from scripts.GradCAM1D import GradCAM as GradCAM1D

# boolean, True - show layout 1, False - show layout 2
b_toggle_layout = True
# integer, for use when displaying the heartbeats
i_index = 0
dataset = None
model_outputs = None


def init_window():
    # init window
    window = tk.Tk()
    window.title("Electrocardiogram")

    layout_1 = tk.LabelFrame(window, text="Layout 1 (Landing)", padx=5, pady=5)
    layout_1.pack()
    layout_2 = tk.LabelFrame(
        window, text="Layout 2 (Prediction)", padx=5, pady=5)

    return window, layout_1, layout_2

#for use to toggle between landing page and display page
def toggle_layout():
    # get global boolean
    global b_toggle_layout
    # swap truth value
    b_toggle_layout = not b_toggle_layout

    # change layout accordingly
    if b_toggle_layout is True:
        layout_1.pack()
        layout_2.pack_forget()
    else:
        layout_1.pack_forget()
        layout_2.pack()

def previous_beat():    
    global i_index
    i_index -= 1
    i_index %= len(dataset)
    update_layout_2(i_index)
    return

def next_beat():
    global i_index
    i_index += 1
    i_index %= len(dataset)
    update_layout_2(i_index)
    return


#landing page
def init_layout_1(parent):
    # label - top centre of the window
    label = tk.Label(parent, text="Electrocardiogram")
    label.config(font=("Courier", 32))
    label.pack()

    # buttons
    btn_select = tk.Button(
        parent, text="Select ECG File", command=process_data)
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

    # THIS PART IS FOR TESTING ONLY (TO SEE IF GRAPH CAN BE SHOWN)
    data_label = ttk.Label(input_frame, text="Enter data (comma-separated):")
    data_label.pack(side=tk.LEFT)
    data_entry = ttk.Entry(input_frame, width=30)
    data_entry.pack(side=tk.LEFT)
    plot_button = ttk.Button(
        input_frame, text="Plot Graph", command=process_data)
    plot_button.pack(side=tk.LEFT)

    return data_entry

#results page
def init_layout_2(parent):
    # label - top centre of the window
    label = tk.Label(parent, text="Electrocardiogram Result")
    label.config(font=("Courier", 32))
    label.pack()

    # Where the graph will appear
    graph_label = tk.Label(parent, text="Here is where the graph will appear")
    graph_label.pack()

    # holder in hireachy
    pred_holder = ttk.Frame(parent)
    pred_holder.pack()

    # children
   
    #graph, heartbeat + explainability?, left side
    graph_frame = ttk.Frame(pred_holder)
    graph_frame.grid(row=0, column=0)
    
    #text explaination, right side
    text_prediction = tk.Text(pred_holder, height=2)
    #text_prediction.delete("start","end")
    text_prediction.insert(tk.END, f"Heartbeat {i_index} out of {0}.") #x = max value of dataset
    text_prediction.insert(tk.END, "Your heartbeat was predicted to have contained : {}-type heartbeats, at a {}% confidence.".format(
        "REPLACE THIS WITH PREDICTED HEARTBEAT", "arbitrary number"))
    # text_prediction.pack()

    text_prediction.grid(row=1, column=0)

    #holder for left/right buttons
    button_holder = ttk.Frame(pred_holder)
    #button_holder.pack()
    button_holder.grid(row=2,column=0)
    

    #left/right buttons, to access previous/next heartbeat
    btn_left = tk.Button(button_holder, text="Previous Beat", command=previous_beat)    
    btn_left.grid(row=1, column=0)
    btn_right = tk.Button(button_holder, text="Next Beat", command=next_beat)
    btn_right.grid(row=1, column=1)

    # TODO : insert a table showing all predictions and all accuracies


    # button to return to previous page
    return_btn = tk.Button(pred_holder, text="Return", command=toggle_layout)
    #return_btn.pack()
    return_btn.grid(row=3,column=0)

    return graph_frame, text_prediction

def update_layout_2(index):
    #update graph_frame 
    fig, ax = plt.subplots()
    plt.close("all")
    plot_explainability(dataset[index], model_outputs["cams"][index], fig, ax)

    for widget in graph_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    #update text_prediction
    pred = model_outputs["preds"][index]
    label = one_hot_decode(pred)
    proba = model_outputs["probas"][index][pred]
    proba *= 100
    
    #update text_prediction
    text_prediction.delete(1.0, tk.END)
    text_prediction.insert(tk.END, f"Heartbeat {index} out of {len(dataset)}.") #x = max value of dataset
    text_prediction.insert(tk.END, f"Your heartbeat was predicted to have contained : {label}-type heartbeats, at a {proba:.2f}% confidence.")


def plot_graph(proba, pred, cam): #TODO: change according maybe
    # * Possibly take in 2 numpy.ndarray rather than just
    # * data (for the heartbeat and the cam)
    # * alternatively, take in an index and
    # * all model outputs (probas, cams)
    # * and plot the graph for that index
    # * using the global dataset variable
    toggle_layout()
    
    x_values = list(range(1, len(proba) + 1))
    #various y values of the different types of heartbeats
    N_values = proba[:,0]
    S_values = proba[:,1]
    V_values = proba[:,2]
    F_values = proba[:,3]
    Q_values = proba[:,4]
    #get the most likely type of heartbeat each heartbeat is 
    highest_values = np.max(proba, axis=1)

    #print("NVALUES")
    #print(N_values)

    #y_values = proba
    #print("xVALUES")
    #print(x_values)

    fig, ax = plt.subplots()
    
    # Possibly use the following line to plot the graph, TODO: not finished, stuck on plat_explainability
    #fig_explain, ax_explain = plt.subplots()
    plot_explainability(dataset[i_index], cam[i_index], fig, ax)
    #canvas_explain = FigureCanvasTkAgg(fig_explain, master=graph_explain_frame)
    #canvas_explain.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    # see below for the function definition

    # Get the number of columns in the input array
    #num_columns = proba.shape[1]

    # Create a color map for differentiating plots
    #color_map = plt.cm.get_cmap('tab10')

    # Plot each column with a different color
    #for j in range(num_columns):
    #    column_data = proba[:, j]
    #    color = color_map(j)  # Get a different color for each column
    #    ax.plot(column_data, label=f"Column {j}", color=color)

    # Plot the highest value for each row with a marker
    #ax.scatter(range(len(highest_values)), highest_values, color='black', marker='o', label="Highest Value")

    #print(len(dataset.__getitem__(5).numpy()))
    #print(dataset.__getitem__(5).numpy())
    
    #x_values = list(range(1, len(dataset.__getitem__(i_index).numpy()) + 1))
    #y_values = dataset.__getitem__(i_index).numpy()
    #ax.plot(x_values, y_values)

    #TODO : change accordingly
    #ax.set_xlabel('Amplitude')
    #ax.set_ylabel('Time')
    #ax.set_title('Electrocardiogram Results')
    
    pred = model_outputs["preds"][i_index]
    label = one_hot_decode(pred)
    proba = model_outputs["probas"][i_index][pred]
    proba *= 100
    
    #update text_prediction
    text_prediction.delete(1.0, tk.END)
    text_prediction.insert(tk.END, f"Heartbeat {i_index} out of {len(dataset)}.") #x = max value of dataset
    text_prediction.insert(tk.END, f"Your heartbeat was predicted to have contained : {label}-type heartbeats, at a {proba:.2f}% confidence.")

    #???? maybe no need
    for widget in graph_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def plot_explainability(heartbeat: np.ndarray, cam: np.ndarray, fig: plt.Figure, ax: plt.Axes, length_time: float = 1.0, fs: int = 360, cmap: str = "inferno"):
    """Plots the Grad-CAM heatmap for a heartbeat.

    Args:
        heartbeat (numpy.ndarray): heartbeat signal of shape (signal_length,)
        cam (numpy.ndarray): Grad-CAM heatmap of shape (signal_length,)
        fig (matplotlib.pyplot.Figure): figure to plot on
        ax (matplotlib.pyplot.Axes): axes to plot on
        length_time (float): length of the heartbeat signal in seconds
        fs (int): sampling frequency of the heartbeat signal
        cmap (str): name of the matplotlib colormap to use

    Usage:
        >>> heartbeat = np.random.randn(360)
        >>> cam = np.random.randn(360)
        >>> fig, ax = plt.subplots()
        >>> plot_explainability(heartbeat, cam, fig, ax)
    """
    x = np.arange(0, length_time, length_time/fs)
    points = np.array([x, heartbeat]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(cam.min(), cam.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(cam.T)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax, label="Grad-CAM Score (Normalised)")
    ax.set_ylim(heartbeat.min(), heartbeat.max())
    return


def process_data():
    # TODO : check on data entry method

    ########## This is the part that reads the csv data but I am not sure which value to take for plotting##############
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

    file_path = filedialog.askopenfilename(
        filetypes=[("WFDB Data Files", "*.dat")])
    if not file_path:
        return

    file_path = file_path.replace('.dat', '')
    sample = wfdb.rdrecord(file_path)
    qrs_locs = gqrs_detect(sample.p_signal[:, 0], sample.fs)
    heartbeats = []
    # Splits the data into 360-sample heartbeats.
    for loc in qrs_locs:
        if loc < 180 or loc > len(sample.p_signal) - 180:
            continue
        heartbeats.append(sample.p_signal[loc - 180:loc + 180, 0])
    heartbeats = np.array(heartbeats)

    #create global dataset for global reference
    global dataset
    dataset = ECGInferenceSet(heartbeats, partial(
        hb_transform, add_noise=False))

    #calling predict and getting results
    proba, pred, cam = model_predict(model, target_layer)
    
    global model_outputs
    model_outputs = {
        "probas": proba,
        "preds": pred,
        "cams": cam
    }

    #plot graph
    plot_graph(proba, pred, cam)
    # data = data_entry.get()
    # try:
    #     data_list = [float(x) for x in data.split(',')]
    #     plot_graph(data_list)
    # except ValueError:
    #     print("Invalid input. Please enter a comma-separated list of numbers.")

    #return proba, pred, cam 


def model_load(
    model_path: str = "./models/prototyping6/tcn_fold_10/best.pth"
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Loads the TCN model

    Args:
        model_path (str, optional): model checkpoint path. Defaults to "./models/prototyping6/tcn_fold_10/best.pth".

    Returns:
        tuple: (model, target_layer) where model is the TCN model and target_layer is the last convolutional layer of the TCN.
    """
    model = TCN(360, 5, [32]*9, 2, 0.125, use_skip_connections=True)
    model.load_state_dict(torch.load(model_path))
    target_layer = model.tcn.network[-1].conv2
    return model, target_layer


def model_predict(
    model: torch.nn.Module,
    target_layer: torch.nn.Module
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicts the heartbeat types and generates the Grad-CAM heatmap for each heartbeat.

    Args:
        model (torch.nn.Module): the TCN model
        target_layer (torch.nn.Module): target layer for Grad-CAM

    Returns:
        (numpy.ndarray, numpy.ndarray): (probas, cams) where probas is the predicted probabilities of each heartbeat type and cams is the Grad-CAM heatmap for each heartbeat.
            probas: probabilities of each heartbeat type after softmax, shape (num_samples, num_classes)
            cams: Grad-CAM heatmap for each heartbeat, shape (num_samples, signal_length)
    """
    use_cuda = torch.cuda.is_available() # check if GPU exists
    batch_size, _ = get_memory_usage(model, 15, use_cuda=use_cuda)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, pin_memory=True)
    with GradCAM1D(model=model, target_layer=target_layer, use_cuda=use_cuda) as cam:
        preds, probas, cams = [], [], []
        for _, inputs in tqdm(enumerate(loader)):
            model.eval()
            with torch.no_grad():
                if use_cuda:
                    inputs = inputs.cuda()
                output = model(inputs).cpu().detach()
                proba = nn.functional.softmax(output, dim=1)
                pred = proba.argmax(dim=1)
                preds.append(pred.numpy())
                probas.append(proba.numpy())
            greyscale_cams = cam(inputs, target_category=pred)
            cams.append(greyscale_cams)
        preds = np.concatenate(preds)
        cams = np.concatenate(cams)
        probas = np.concatenate(probas)
    return probas, preds, cams


if __name__ == "__main__":
    window, layout_1, layout_2 = init_window()
    data_entry = init_layout_1(layout_1)
    graph_frame, text_prediction = init_layout_2(layout_2)

    # TODO : load model here
    file_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(file_path)
    file_name = folder_path + "/models/prototyping6/tcn_fold_10/best.pth" 

    model, target_layer = model_load(model_path=file_name)

    # TODO : call model_predict in/ together with plotgraph, and change layout_2 accordingly
    # proba, cams = model_predict(model, target_layer)

    window.mainloop()
