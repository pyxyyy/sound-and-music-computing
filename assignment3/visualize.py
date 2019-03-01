"""
Usage: python visualize.py
Performs visualization of feature pairs given an .arff input file as laid out in assignment3.pdf
"""

import scipy.io.arff
import matplotlib as mpl
mpl.use('TkAgg')
import pylab


def plot(data, xattr, yattr, clsattr, title):
    """
    Generates a music vs speech scatter plot based on the given data.
    :param data: data read from .arff file
    :param xattr: attribute name of x axis
    :param yattr: attribute name of y axis
    :param clsattr: attribute name of class i.e. music, speech
    :param title: title of generated plot
    :return:
    """
    zcr = data[xattr]
    par = data[yattr]
    xspeech = []
    yspeech = []
    xmusic = []
    ymusic = []

    for i, x in enumerate(data[clsattr]):
        x = x.decode('UTF-8')
        if x == 'speech':
            xspeech.append(zcr[i])
            yspeech.append(par[i])
        elif x == 'music':
            xmusic.append(zcr[i])
            ymusic.append(par[i])

    pylab.figure()
    pylab.xlabel(xattr)
    pylab.ylabel(yattr)
    pylab.title(title)
    pylab.scatter(xmusic, ymusic, marker='o', label='Music', color='magenta')
    pylab.scatter(xspeech, yspeech, marker='x', label='Speech', color='blue')
    pylab.legend()
    pylab.show()


def visualize():
    data, meta = scipy.io.arff.loadarff('features.arff')
    plot(data=data, xattr='ZCR_MEAN_TIME', yattr='PAR_MEAN_TIME', clsattr='class', title='ZCR_PAR_MEAN')
    plot(data=data, xattr='ZCR_STD_TIME', yattr='PAR_STD_TIME', clsattr='class', title='ZCR_PAR_STD')


if __name__ == "__main__":
    visualize()
