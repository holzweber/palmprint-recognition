import tkinter.filedialog
from PalmprintRecognitionNetwork import predictAndShow
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

canvas = None


def plot():
    global canvas
    if canvas is not None:
        canvas.get_tk_widget().destroy()
    path = tkinter.filedialog.askopenfilename(filetypes=[("Image File", '.tiff .jpg .png')])
    if path is None:
        return
    trueLabel = path.split('/')[-2]
    fig = predictAndShow(path, trueLabel)
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    # frame = Frame(window);
    canvas = FigureCanvasTkAgg(fig, window)

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()
    canvas.draw()



# the main Tkinter window
window = Tk()

# setting the title
window.title('Palmprint Recognition')

# dimensions of the main window
window.geometry("1000x800")

# button that displays the plot
plot_button = Button(master=window,
                     command=plot,
                     height=1,
                     width=15,
                     text="Open Image File")

# place the button
# in main window
plot_button.pack()

# run the gui
window.mainloop()
