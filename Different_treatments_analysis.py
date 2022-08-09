import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
import glob
import tifffile as tiff
def convert_to_gs(image):
    '''
    :param image: 2D array.
    :return: gs_image - 2D array image in grayscale values range.
    '''
    #convert to grayscale
    imin = image.min()
    imax = image.max()
    a = 255 / (imax - imin)
    b = 255 - a * imax
    gs_image = (a * image + b).astype(np.uint8)
    return gs_image

def count_cells(img):
    '''
    :param img: 2D array of grayscale image.
    :return: cells_detected. int - number of cells detected in the given image.
    '''
    gray = convert_to_gs(img)
    _, b = np.histogram(gray)
    thresh_value = b[1] * 0.5
    _, th = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel)
    sure_bg = cv2.dilate(opening,kernel)
    sure_bg = np.uint8(sure_bg)
    dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2, 3)
    _, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 10
    markers[unknown == 255] = 0
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)
    color[markers == -1] = [0, 255, 255]

    regions = measure.regionprops(markers)
    cells_detected = len(regions)
    return cells_detected

def get_cells_info(frames):
    '''
    :param frames: list of arrays, each array represents a frame from the tiff file
    :return: cells_count - list of cells counted. Each cells_count[i] represents number of cells in frames[i]
            x_axis - array. Each value represents the time (in hours) of frame taken from the tiff file.
            rates - list. Represents the rates (dx/dt) of hair cells appearance in current treatment.
    '''
    cells_count = []
    x_axis = []
    for i in range(len(frames)):
        n_cells = count_cells(frames[i])
        cells_count.append(n_cells)
        frame_time = i * number_of_frames * 0.25
        x_axis.append(frame_time)
    rates = []
    for i in range(len(cells_count)- 1):
        dx = cells_count[i+1] - cells_count[i]
        dt = x_axis[i+1] - x_axis[i]
        rate = dx / dt
        rates.append(rate)
    rates = np.mean(rates)

    return cells_count, x_axis, rates
class Tiff_movie():
    def __init__(self, movie_path):
        self.movie_path = movie_path
        movie_tiff = tiff.imread(movie_path)
        self.movie_file = movie_tiff
        self.size = movie_tiff.shape

    def get_frame(self, time, channel):
        try:
            frame = self.movie_file[time][channel]
            return frame
        except IndexError:
            print("Invalid time/channel. Channel max value:{}, time max value:{}".format(self.movie_file.shape[1]-1, self.movie_file.shape[0]-1))
            return None

    def get_every_x_frame(self, channel, x):
        frames = []
        l = self.size[0]
        for i in range(0, l, x):
            frame = self.get_frame(i, channel)
            frames.append(frame)
        return frames
movies = []
movies_cells_counts = []
threshes = []
treatments = []
all_rates = []
number_of_frames = 12  # each frame is 0.25h
for filepath in glob.iglob('Movies/*.tif'):
    treatment = filepath[7:-4]
    movie = Tiff_movie(filepath)
    movies.append(movie)
    frames = movie.get_every_x_frame(0, number_of_frames)
    cells_count, x_axis, rate = get_cells_info(frames)
    arr_cell = np.array(cells_count)
    arr_cell = arr_cell/arr_cell[0]
    movies_cells_counts.append({treatment: [arr_cell, x_axis]})
    all_rates.append(np.array(rate))
    treatments.append(treatment)

## plot #1
fig, ax = plt.subplots(ncols=3)
fig.set_size_inches(18, 7)
fig.canvas.manager.set_window_title('Plot #1')
fig.suptitle('Hair cells population over time under different treatments')
fig.text(0.5, 0.04, 'No. of hours', ha='center', va='center')
fig.text(0.06, 0.5, 'Ratio between New hair cells to initial no. of hair cells', ha='center', va='center', rotation='vertical')
i = 0
colors = ['coral', 'seagreen', 'royalblue']
for item in movies_cells_counts:
    for key, value in item.items():
        title = key
        cells = value[0]
        axis_x = value[1]
        ax[i].plot(axis_x, cells, color=colors[i])
        ax[i].set_title(title)
        i += 1

##plot #2
x_pos = np.arange(len(treatments))
fig, ax = plt.subplots()
fig.canvas.manager.set_window_title('Plot #2')
ax.bar(x_pos, all_rates, align='center', ecolor='black', capsize=10, color='DarkSlateBlue')
ax.set_ylabel('Average rate - [cells/hour]')
ax.set_xticks(x_pos)
ax.set_xticklabels(treatments)
ax.set_title('Average rate of hair cells emergent under different treatments')
plt.show()
