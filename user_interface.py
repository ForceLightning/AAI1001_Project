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
#global data variables
dataset = None
model_outputs = None

def init_window():
    # init window
    window = tk.Tk()
    window.title("Electrocardiogram")

    #init layout
    layout_1 = tk.LabelFrame(window)
    layout_1.pack(padx=10, pady=10)
    layout_2 = tk.LabelFrame(window)

    #return to be referenced and edited later
    return window, layout_1, layout_2

#for use to toggle between landing page and display page
def toggle_layout():
    # get global boolean
    global b_toggle_layout
    # swap truth value
    b_toggle_layout = not b_toggle_layout

    # change layout accordingly
    if b_toggle_layout is True:
        layout_1.pack(padx=10, pady=10)
        layout_2.pack_forget()
    else:
        layout_1.pack_forget()
        layout_2.pack(padx=10, pady=10)

def previous_beat():    
    #decrement index and update layout
    global i_index
    i_index -= 1
    i_index %= len(dataset)
    update_layout_2(i_index)
    return

def next_beat():
    #increment index and update layout
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

    # button to prompt user to select ECG file
    btn_select = tk.Button(parent, text="Select ECG File", command=process_data)
    btn_select.pack(pady=20)


#results page
def init_layout_2(parent):
    # label - top centre of the window
    label = tk.Label(parent, text="Electrocardiogram Result")
    label.config(font=("Courier", 32))
    label.pack()

    # holder in hireachy
    pred_holder = ttk.Frame(parent)
    pred_holder.pack()

    # children to pred holder
    graph_holders = ttk.Frame(pred_holder)
    graph_holders.grid(row=0, column=0)

    #children to graph_holder
    #graph, heartbeat + explainability, left side
    graph_frame = ttk.Frame(graph_holders)
    graph_frame.grid(row=0, column=0, padx=5, pady=5)
    #graph table, table of predictions and confidence levels, right side
    graph_proba_frame = ttk.Frame(graph_holders)
    graph_proba_frame.grid(row=0, column=1, padx=5, pady=5)

    #text explaination
    text_prediction = tk.Text(pred_holder, height=3)
    text_prediction.grid(row=1, column=0)

    #holder for left/right buttons
    button_holder = ttk.Frame(pred_holder)
    #button_holder.pack()
    button_holder.grid(row=2,column=0)
    

    #left/right buttons, to access previous/next heartbeat
    btn_left = tk.Button(button_holder, text="Previous Beat", command=previous_beat)    
    btn_left.grid(row=1, column=0, padx=5, pady=5)
    btn_right = tk.Button(button_holder, text="Next Beat", command=next_beat)
    btn_right.grid(row=1, column=1, padx=5, pady=5)

    # button to return to previous page
    return_btn = tk.Button(pred_holder, text="Return", command=toggle_layout)
    return_btn.grid(row=3,column=0, padx=5, pady=5)

    return graph_frame, text_prediction, graph_proba_frame

def update_layout_2(index):
    #update graph_frame 
    fig, ax = plt.subplots()
    plt.close("all")
    plot_explainability(dataset[index].numpy(), model_outputs["cams"][index], fig, ax)

    for widget in graph_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    #update graph_proba_frame
    fig_proba, ax_proba = plt.subplots()

    x_values = ["N", "S", "V", "F", "Q"]
    y_values = model_outputs["probas"][index] * 100
    plt.bar(x_values, y_values)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.xlabel("Types of Heartbeats")
    plt.ylabel("Confidence Level (%)")

    for widget in graph_proba_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig_proba, master=graph_proba_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    #update text_prediction
    pred = model_outputs["preds"][index]
    label = label_decode(int(pred), ["Non-Ectopic", "Supraventricular Ectopic", "Ventricular Ectopic", "Fusion", "Unknown"])
    proba = model_outputs["probas"][index][pred]
    proba *= 100
    
    text_prediction.delete(1.0, tk.END)
    text_prediction.insert(tk.END, f"Heartbeat {index + 1} of {len(dataset)}.\n") 
    text_prediction.insert(tk.END, f"This heartbeat was predicted to be {label}, with a {proba:.2f}% confidence.")


def plot_graph():
    #change to results page and update it
    toggle_layout()   
    update_layout_2(i_index)


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

    #create global dataset for referencing
    global dataset
    dataset = ECGInferenceSet(heartbeats, partial(
        hb_transform, add_noise=False))

    #calling predict and getting results
    proba, pred, cam = model_predict(model, target_layer)
    
    #create global outputs for referencing
    global model_outputs
    model_outputs = {
        "probas": proba,
        "preds": pred,
        "cams": cam
    }

    #activate results page and update it
    plot_graph()


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
    #init GUI
    window, layout_1, layout_2 = init_window()
    init_layout_1(layout_1)
    graph_frame, text_prediction, graph_proba_frame = init_layout_2(layout_2)

    #loading model here
    file_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(file_path)
    file_name = folder_path + "/models/prototyping6/tcn_fold_10/best.pth" 
    model, target_layer = model_load(model_path=file_name)
    
    #tkinter main loop
    window.protocol("WM_DELETE_WINDOW", lambda root=window:root.destroy())
    window.mainloop()
