"""
usage: python gen_arff.py
generates arff file as laid out in assignment1.pdf
"""

import math
import numpy
import scipy.io.wavfile
import scipy.signal.windows
import scipy.fftpack


# set some magic constants
FLOAT_TRANSFORMATION = 32768.0
BUFFER_LENGTH = 1024
HOPSIZE = 512
SRO_L = 0.85


def calc_rms(buffer):
    buffer_squares = buffer ** 2
    return math.sqrt(sum(buffer_squares) / len(buffer_squares))


def calc_zcr(buffer):
    return (buffer[:-1] * buffer[1:] < 0).sum() / (len(buffer) - 1)


def calc_uncorr_sample_std_dev(x, mean):
    sq_devs = ((numpy.array(x) - ([mean] * len(x))) ** 2).sum()
    sample_var = sq_devs / len(x)
    return math.sqrt(sample_var)


def calc_time_domain_feats(buffers):
    rms = []
    zcr = []
    for buffer in buffers:
        # calculate buffer RMS
        brms = calc_rms(buffer)
        rms.append(brms)
        # calculate buffer ZCR
        bzcr = calc_zcr(buffer)
        zcr.append(bzcr)
    # calculate mean
    mean_rms = numpy.mean(rms)
    mean_zcr = numpy.mean(zcr)
    # calculate uncorrected sample std dev
    ussd_rms = calc_uncorr_sample_std_dev(rms, mean_rms)
    ussd_zcr = calc_uncorr_sample_std_dev(zcr, mean_zcr)
    return mean_rms, mean_zcr, ussd_rms, ussd_zcr


def get_spectral_buffer(buffer):
    windowed = buffer * scipy.signal.windows.hamming(BUFFER_LENGTH)
    ffted = scipy.fftpack.fft(windowed)
    return ffted[:int(len(ffted) / 2)+1]


def calc_sc(spectral_buffer):
    k = [i for i in range(len(spectral_buffer))]
    sc_denom = sum(abs(spectral_buffer))
    sc_numer = sum(k * abs(spectral_buffer))
    return sc_numer / sc_denom


def calc_sro(spectral_buffer):
    sro_rhs = SRO_L * sum(abs(spectral_buffer))
    for r in range(len(spectral_buffer)):
        if sum(abs(spectral_buffer[:r])) >= sro_rhs:
            return r-1


def calc_sfm(spectral_buffer):
    k = [i for i in range(len(spectral_buffer))]
    sfm_denom = sum(abs(spectral_buffer)) / len(spectral_buffer)
    sfm_numer = 0
    for ik in k:
        sfm_numer += math.log(abs(spectral_buffer[ik]), math.e)
    sfm_numer /= len(spectral_buffer)
    sfm_numer = math.exp(sfm_numer)
    return sfm_numer / sfm_denom


def calc_freq_domain_feats(buffers):
    sc = []
    sro = []
    sfm = []
    for buffer in buffers:
        # get spectral buffer
        spectral_buffer = get_spectral_buffer(buffer)
        # calculate buffer SC
        bsc = calc_sc(spectral_buffer)
        sc.append(bsc)
        # calculate buffer SRO
        bsro = calc_sro(spectral_buffer)
        sro.append(bsro)
        # calculate buffer SFM
        bsfm = calc_sfm(spectral_buffer)
        sfm.append(bsfm)
    # calculate mean
    mean_sc = numpy.mean(sc)
    mean_sro = numpy.mean(sro)
    mean_sfm = numpy.mean(sfm)
    # calculate uncorrected sample std dev
    ussd_sc = calc_uncorr_sample_std_dev(sc, mean_sc)
    ussd_sro = calc_uncorr_sample_std_dev(sro, mean_sro)
    ussd_sfm = calc_uncorr_sample_std_dev(sfm, mean_sfm)
    return mean_sc, mean_sro, mean_sfm, ussd_sc, ussd_sro, ussd_sfm


def main():
    fout = open("results.arff", "w")

    # write .arff header
    fout.write("@RELATION music_speech\n")
    fout.write("@ATTRIBUTE RMS_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE ZCR_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE SC_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE SRO_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE SFM_MEAN NUMERIC\n")
    fout.write("@ATTRIBUTE RMS_STD NUMERIC\n")
    fout.write("@ATTRIBUTE ZCR_STD NUMERIC\n")
    fout.write("@ATTRIBUTE SC_STD NUMERIC\n")
    fout.write("@ATTRIBUTE SRO_STD NUMERIC\n")
    fout.write("@ATTRIBUTE SFM_STD NUMERIC\n")
    fout.write("@ATTRIBUTE class {music,speech}\n")
    fout.write("@DATA\n")

    with open("../data/music_speech.mf", "r") as fin:
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
            for i in range(num_buffers):
                start = i * HOPSIZE
                end = start + BUFFER_LENGTH
                buffer_data = ffloats[start:end]
                buffers.append(buffer_data)
            # perform calculations
            mean_rms, mean_zcr, ussd_rms, ussd_zcr = calc_time_domain_feats(buffers)
            mean_sc, mean_sro, mean_sfm, ussd_sc, ussd_sro, ussd_sfm = calc_freq_domain_feats(buffers)

            # write results as .arff
            output = "{:0.6f},{:0.6f},{:0.6f},{:0.6f},{:0.6f},{:0.6f},{:0.6f},{:0.6f},{:0.6f},{:0.6f},{}\n".format(
                mean_rms, mean_zcr, mean_sc, mean_sro, mean_sfm, ussd_rms, ussd_zcr, ussd_sc, ussd_sro, ussd_sfm,
                flabel)
            fout.write(output)

    fout.close()


if __name__ == "__main__":
    main()
