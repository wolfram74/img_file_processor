import numpy
import matplotlib.pyplot as pyplot
import cv2
import scipy.optimize as opt
import time

def twoD_Gaussian(x_y, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = x_y
    xo = float(xo)
    yo = float(yo)
    a = (numpy.cos(theta)**2)/(2*sigma_x**2) + (numpy.sin(theta)**2)/(2*sigma_y**2)
    b = -(numpy.sin(2*theta))/(4*sigma_x**2) + (numpy.sin(2*theta))/(4*sigma_y**2)
    c = (numpy.sin(theta)**2)/(2*sigma_x**2) + (numpy.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*numpy.exp(
         - (
            a*((x-xo)**2)
            + 2*b*(x-xo)*(y-yo)
            + c*((y-yo)**2)
            )
         )
    return g.ravel()


def converter(img_data, output_path):
    shape = img_data.shape
    dpi = 2**7
    img_size=(
        float(shape[1])/dpi,
        float(shape[0])/dpi
        )
    fig = pyplot.figure()
    fig.set_size_inches(img_size)
    ax = pyplot.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    pyplot.set_cmap('gray')
    ax.imshow(img_data, aspect='equal')
    pyplot.savefig(output_path, dpi=dpi)
    return

def load_data(file_path):
    shape = (2**10, 2**9*3) # matrix size (rows of data, width of row)
    #accompanying .inf file lists width then height
    dtype = numpy.dtype('>u2') # big-endian unsigned integer (16bit)
    fid = open(file_path, 'rb')
    data = numpy.fromfile(fid, dtype)
    data_out = data.reshape(shape)
    return data_out

def blob_finding(file_path):
    im = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # custom params
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = True;
    params.blobColor = 255; #focus on bright blobs
    params.filterByInertia = True
    params.minInertiaRatio = 0.04 # allow for oblate shapes
    params.filterByArea = True
    params.minArea = 10

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    # print(ver)
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else :
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(im)
    return keypoints

def sort_blobs(blob_points):
    ordered_indices = []
    inter = (lambda x: (int(x.pt[0]), int(x.pt[1])))
    sorted_points = sorted(blob_points, key=(lambda x: x.pt[1]))
    top10 = sorted_points[:10]
    bot10 = sorted_points[10:]
    top10 = sorted(top10, key=(lambda x: x.pt[0]))
    bot10 = sorted(bot10, key=(lambda x: x.pt[0]))
    top10 = map(inter, top10)
    bot10 = map(inter, bot10)
    return [top10, bot10]

def process_blob(center, data):
    minx = center[0]-30
    maxx = center[0]+30
    miny = center[1]-20
    maxy = center[1]+20
    subset = data[miny:maxy, minx:maxx]
    #save png of each individual blob
    # test = './%d-%d.png' % center
    # converter(subset, test)
    gaussian_params = fit_blob_to_gaussian(subset)
    return gaussian_params[0]*gaussian_params[1]*gaussian_params[2]

def fit_blob_to_gaussian(data_set):
    blob_shape = data_set.shape
    x = numpy.linspace(0,blob_shape[1],blob_shape[1])
    y = numpy.linspace(0, blob_shape[0], blob_shape[0])
    x, y = numpy.meshgrid(x, y)
    # print(x[:5])
    # image_plot(data_set)
    initial_guess = (10000,30,20,20,40,0,10)
    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data_set.ravel(), p0=initial_guess)
    # fit_dat = twoD_Gaussian((x,y), *popt).reshape(blob_shape)
    # image_plot(fit_dat)
    return popt

def save_results(data, img_dest):
    txt_dest = img_dest[:-3]+'txt'
    target = open(txt_dest, 'w')
    target.write(img_dest)
    target.write('\n')
    for row in data:
        target.write('\t'.join([str(el) for el in row]))
        target.write('\n')
    return

# fit quantities
# [ 3.23329653e+04  3.93196841e+01  2.07697020e+01  2.00387675e+01
#  -8.21284563e+00 -5.46652970e-03 -1.21646729e+03]
