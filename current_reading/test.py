#!/usr/bin/env python
#
# Simple script showing how to stream data from a device in continuous mode,
# use Ctrl-C to exit.

from __future__ import print_function

from signal import signal, SIG_DFL, SIGINT
import sys, time, traceback
import numpy as np
from pysmu import Session, Mode


# If stdout is a terminal continuously overwrite a single line, otherwise
# output each line individually.
if sys.stdout.isatty():
    output = lambda s: sys.stdout.write("\r" + s)
else:
    output = print


if __name__ == '__main__':
    # don't throw KeyboardInterrupt on Ctrl-C
    signal(SIGINT, SIG_DFL)

    session = Session()

    if session.devices:
        # Grab the first device from the session.
        dev = session.devices[0]

        # Set both channels to high impedance mode.
        chan_a = dev.channels['A']
        chan_b = dev.channels['B']
        chan_a.mode = Mode.HI_Z
        chan_b.mode = Mode.HI_Z

        # Ignore read buffer sample drops when printing to stdout.
        dev.ignore_dataflow = sys.stdout.isatty()

        # Start a continuous session.
        session.start(0)
    
        while True:
            # Read incoming samples from both channels which are in HI-Z mode
            # by default in a blocking fashion.
            now=time.time()
            samples = np.array(dev.read(100, -1))   ###read 100 samples
            print(np.average(samples[:,0,0]))
            # print(time.time()-now)
    else:
        print('no devices attached')