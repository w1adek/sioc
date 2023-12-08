import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from sklearn import metrics

"""Functions to be interpolated"""
def f1(x):
    return np.sin(x)
def f2(x):
    return np.sin(x**-1)
def f3(x):
    return np.sign(np.sin(8*x))

current_function = f1

"""Interpolation kernels"""
def h1(x):
    return np.where((x > 0) & (x < 1), 1, 0)
def h2(x):
    return np.where((x >= -0.5) & (x < 0.5), 1, 0)
def h3(x):
    return np.where((x >= -1) & (x <= 1), 1-np.abs(x), 0)
def h4(x):
    return np.sinc(x)

current_kernel = h1

"""Plotting sampled and interpolated functions"""
x_samples = np.linspace(-np.pi, np.pi, num=10)
y_samples = current_function(x_samples)

#x_normal_distrb = np.random.normal(0, np.pi/2, size=10) # Normal Distribution

x_interp = np.linspace(-np.pi, np.pi, num=20)
x_shifted = (x_interp - x_samples[:, np.newaxis]) / np.diff(x_samples)[0]
y_interp = np.dot(y_samples, b=current_kernel(x_shifted))

plt.style.use('seaborn-v0_8-darkgrid')

fig, ax = plt.subplots(figsize=(7, 4))
plt.title("Original and Interpolated functions")
plt.subplots_adjust(bottom=0.25, right=0.7)

samples, = ax.plot(x_samples, y_samples, '.r', label='samples')
interp, = ax.plot(x_interp, y_interp, '-g', label='interpolation')
plt.legend(loc=(1.06, 0))

mse = metrics.mean_squared_error(current_function(x_interp), y_interp)
mse_text = plt.text(4.2, -0.6, f'MSE: {mse:.4f}%')

"""Slider that updates values on sample axes"""
x_samples_slider = Slider(
    plt.axes([0.15, 0.15, 0.55, 0.03]),
    'Samples',
    valmin=10,
    valmax=100,
    valstep=10,
    valinit=10
    )

"""Slider that updates values on interpolation axes"""
x_interp_slider = Slider(
    plt.axes([0.15, 0.09, 0.55, 0.03]),
    'Iterpolation\nscale factor',
    valmin=1,
    valmax=20,
    valstep=1,
    valinit=2
    )

def update(val):
    current_sval = x_samples_slider.val
    x_samples = np.linspace(-np.pi, np.pi, num=current_sval)
    y_samples = current_function(x_samples)
    samples.set_data(x_samples, y_samples)
    
    current_ival = x_interp_slider.val
    x_interp = np.linspace(-np.pi, np.pi, num=current_ival * current_sval)
    x_shifted = (x_interp - x_samples[:, np.newaxis]) / np.diff(x_samples)[0]
    y_interp = np.dot(y_samples, b=current_kernel(x_shifted))
    interp.set_data(x_interp, y_interp)
    
    mse = metrics.mean_squared_error(current_function(x_interp), y_interp)
    mse_text.set_text(f'MSE: {mse:.4f}%')
    
    fig.canvas.draw()
    
x_samples_slider.on_changed(update)
x_interp_slider.on_changed(update)

"""Radio buttons that changes function"""
select_function_button = RadioButtons(
    plt.axes([0.71, 0.68, 0.24, 0.2]),
    ('sin(x)', 'sin(x^-1)', 'sign(sin(8x))'),
    activecolor='r'
    )

def select_function(label):
    global current_function
    dict = {
        'sin(x)': f1,
        'sin(x^-1)': f2,
        'sign(sin(8x))': f3
        }
    current_function = dict[label]
    update(None)
    
select_function_button.on_clicked(select_function)

"""Radio buttons that changes interpolation kernel"""
select_kernel_button = RadioButtons(
    plt.axes([0.71, 0.46, 0.24, 0.2]),
    ('sample and hold', 'nearest neighbour', 'linear', 'sinc'),
    activecolor='g'
    )

def select_kernel(label):
    global current_kernel
    dict = {
        'sample and hold': h1,
        'nearest neighbour': h2,
        'linear': h3,
        'sinc': h4
        }
    current_kernel = dict[label]
    update(None)
    
select_kernel_button.on_clicked(select_kernel)

plt.show()