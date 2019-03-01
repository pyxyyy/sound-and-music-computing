"""
usage: python synthesis.py
(a) synthesises a wav file modeled after scale.wav
(b) lengthens one of the notes in scale.wav
as laid out in assignment4.pdf
"""

import numpy
import scipy.io.wavfile
import pylab

fs = 44100
fin = '../data/scale.wav'
fout = 'sin_scale.wav'
fout_long = 'long_scale.wav'

f0 = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # c4
f02 = [523.25,  587.33,  659.25,  698.46,  783.99,  880.00,  987.77,  1046.50]  # c5
f03 = [783.99,  880.00,  987.77, 1046.50, 1174.66, 1318.51, 1479.98,  1567.98]  # g5
f04 = [1046.50,  1174.66,  1318.51,  1396.91,  1567.98,  1760.00,  1975.53,  2093.00]  # c6
f05 = [1318.51, 1479.98, 1661.22, 1760.00, 1975.53,  2217.46, 2489.02,  2637.02]  # e6
f06 = [1567.98,  1760.00, 1975.53,  2093.00,  2349.32, 2637.02,  2959.96, 3135.96]  # g6

period = [0.592, 0.581, 0.546, 0.557, 0.557, 0.581, 0.534, 0.743]


def build_signal(f, t, a, n):
    samples = numpy.arange(t * fs) / fs
    signal = a * numpy.sin(2 * numpy.pi * f * samples) / n
    signal *= 32767
    signal = numpy.int16(signal)
    return signal


def main():

    # 1. pitch tracking and wav file synthesis
    scale = numpy.int16(numpy.zeros(0))
    amp = 0.1
    ns = 6
    empty = build_signal(1/0.3, 0.3, 0, 1)

    # build f0 and some harmonic partials for each note in scale
    for i in range(len(f0)):
        harmonics = build_signal(f0[i], period[i], amp, ns) + build_signal(f02[i], period[i], amp/4, ns) \
                    + build_signal(f03[i], period[i], amp/8, ns) + build_signal(f04[i], period[i], amp/16, ns) \
                    + build_signal(f05[i], period[i], amp/32, ns) + build_signal(f06[i], period[i], amp/64, ns)

        """
        # phase shift to remove pop sound
        """

        # fade to remove pop sound
        fade_in_val = 10
        fade_out_val = 20000
        fade_in = numpy.arange(.9, 1., .1 / float(fade_in_val))
        fade_out = numpy.arange(1., .1, -.9 / float(fade_out_val))
        harmonics[:len(fade_in)] = numpy.multiply(harmonics[:len(fade_in)], fade_in)
        harmonics[-len(fade_out):] = numpy.multiply(harmonics[-len(fade_out):], fade_out)
        scale = numpy.concatenate([scale, harmonics])

    # pad signal start with silence to match scale.wav
    scale = numpy.concatenate([empty, scale])
    scipy.io.wavfile.write(fout, fs, scale)

    # 2. note length modification (first note)
    frate, fraw = scipy.io.wavfile.read(fin)
    fdata = fraw / 32768.0
    pylab.figure()
    pylab.xlim(36000, 37000)
    pylab.ylim(-0.006, 0.005)
    pylab.plot(fdata)
    pylab.show()

    idx_st = int(36193.5)
    idx_en = int(36358.0)
    long_scale = fdata[:idx_st+1]
    num_repetitions = (2 * 60 * 1000) / float(idx_en-idx_st)
    # repeat long_scale (one period) for 2 seconds
    for repetition in range(int(num_repetitions)):
        long_scale = numpy.concatenate([long_scale, fdata[idx_st:idx_en+1]])
    long_scale = numpy.concatenate([long_scale, fdata[idx_en+1:]])
    scipy.io.wavfile.write(fout_long, fs, long_scale)


if __name__ == '__main__':
    main()
