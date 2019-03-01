"""
usage: python gen_arff.py
generates arff file and graphs as laid out in assignment2.pdf
"""

import math
import numpy
import scipy.io.wavfile
import scipy.signal.windows
import scipy.fftpack
import matplotlib as mpl
mpl.use('TkAgg')
import pylab


# setting some magic constants
FLOAT_TRANSFORMATION = 32768.0
BUFFER_LENGTH = 1024
OVERLAP = 0.5
HOPSIZE = int(OVERLAP * BUFFER_LENGTH)
NUM_MEL_FREQ_FILTERS = 26


def calc_uncorr_sample_std_dev(x, mean):
    sq_devs = ((numpy.array(x) - ([mean] * len(x))) ** 2).sum()
    sample_var = sq_devs / len(x)
    return math.sqrt(sample_var)


def apply_preemphasis_window(bdata):
    filtered = []
    for t in range(len(bdata)):
        yt = bdata[t] if t == 0 else bdata[t] - 0.95 * bdata[t-1]
        filtered.append(yt)
    return filtered


def mel(freq):
    return (math.log(1 + (float(freq) / 700), math.e)) * 1127


def freq(mel):
    return (math.exp(float(mel)/1127) - 1) * 700


def apply_mel_freq_filtering(hwindow, frate, nfft):
    min_mel = mel(0)
    max_mel = mel(float(frate) / 2)
    # let mel freq filters cover entire frequency range
    frange = numpy.linspace(min_mel, max_mel, NUM_MEL_FREQ_FILTERS + 2)
    hertz = numpy.array([freq(x) for x in frange])
    # build fft bins
    fft_bins = float(BUFFER_LENGTH) * (hertz / frate)
    # fbanks is a matrix of filter banks x filter bins
    fbanks = numpy.zeros((NUM_MEL_FREQ_FILTERS, nfft))
    for fbank in range(NUM_MEL_FREQ_FILTERS):
        # find x-axis points of filters, convert to int fft bins
        left = int(numpy.floor(fft_bins[fbank]))
        top = int(numpy.round(fft_bins[fbank+1]))
        right = int(numpy.ceil(fft_bins[fbank+2]))
        # linearly interpolate intermediate regions
        for val in range(left, top):
            fbanks[fbank, val] = float((val - left)) / (top - left)
        for val in range(top, right):
            fbanks[fbank, val] = float((right - val)) / (right - top)

    # plot triangular windows
    # plot_triangular(22050, fbanks, True)
    # plot_triangular(22050, fbanks, False)

    # compute filter bank energies from filter banks
    return numpy.dot(hwindow, fbanks.T)


def plot_triangular(frate, fbanks, isfull):
    # calculate num of fft bins
    num_fft_bins = int(BUFFER_LENGTH / 2) + 1
    # plot
    pylab.figure()
    pylab.xlabel('Frequency (Hz)')
    pylab.xlim([0, 12000])
    pylab.ylabel('Amplitude')
    pylab.ylim([0, 1])
    pylab.title('26 Triangular MFCC filters, 22050Hz signal, window size 1024')
    for fbank in range(NUM_MEL_FREQ_FILTERS):
        x_range = [val * frate * 0.5 / num_fft_bins for val in range(num_fft_bins)]
        y_range = fbanks[fbank]
        if isfull:
            pylab.plot(x_range, y_range)
        else:
            pylab.plot(x_range, y_range, marker='o')
    if isfull:
        pylab.axis([0, 12000, 0, 1.0])
    else:
        pylab.axis([0, 300, 0, 1.0])
    pylab.show()


def main():
    fout = open("results.arff", "w")

    # write .arff header
    fout.write("@RELATION music_speech\n")
    fout.write("@ATTRIBUTE BIN_A_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_B_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_C_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_D_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_E_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_F_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_G_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_H_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_I_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_J_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_K_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_L_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_M_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_N_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_O_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_P_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_Q_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_R_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_S_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_T_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_U_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_V_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_W_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_X_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_Y_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_Z_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_A_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_B_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_C_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_D_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_E_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_F_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_G_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_H_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_I_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_J_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_K_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_L_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_M_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_N_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_O_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_P_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_Q_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_R_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_S_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_T_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_U_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_V_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_W_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_X_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_Y_STD NUMERIC\n")
    fout.write("@ATTRIBUTE BIN_Z_STD NUMERIC\n")
    fout.write("@ATTRIBUTE class {music,speech}\n")
    fout.write("@DATA\n")

    with open("music_speech.mf", "r") as fin:
        for line in fin:
            fname, flabel = line.strip().split('\t')
            print(fname, flabel)
            # load wav file
            frate, fdata = scipy.io.wavfile.read(fname)
            # convert data to floats
            ffloats = fdata / FLOAT_TRANSFORMATION
            # only keep complete buffers
            rmdr = len(ffloats) % HOPSIZE
            num_buffers = math.floor(len(ffloats) / HOPSIZE) if rmdr == 0 else math.floor(len(ffloats) / HOPSIZE) - 1
            # split data into buffers
            buffers = []
            # store dct's for mean and std computation
            dcts = []
            # calculate mfcc's for each buffer
            for i in range(num_buffers):
                start = i * HOPSIZE
                end = start + BUFFER_LENGTH
                buffer_data = ffloats[start:end]
                buffers.append(buffer_data)
                # apply preemphasis filter
                filtered = apply_preemphasis_window(buffer_data)
                # apply hamming window
                hammed = filtered * scipy.signal.windows.hamming(BUFFER_LENGTH)
                # apply fft; trash negative values and get absolute
                ffted = scipy.fftpack.fft(hammed)
                ffted = abs(ffted[:int(len(ffted) / 2)+1])
                # apply mel freq filtering, then log
                energies_lg = numpy.log10(apply_mel_freq_filtering(ffted, frate, len(ffted)))
                # apply DCT
                dcted = scipy.fftpack.dct(energies_lg)
                dcts.append(dcted)
            # find mean
            ftvec = numpy.mean(dcts, axis=0)
            # find uncorrected sample std dev
            for fb in range(NUM_MEL_FREQ_FILTERS):
                col = numpy.array(dcts)[:, fb]
                ftvec = numpy.append(ftvec, calc_uncorr_sample_std_dev(col, ftvec[fb]))
            # keep at least 6 dp's
            output = ",".join([format(ft, "0.6f") for ft in ftvec])
            # write results as .arff
            output = output + "," + flabel + "\n"
            fout.write(output)

    fout.close()


if __name__ == "__main__":
    main()
