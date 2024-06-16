import RobotRaconteur as RR
RRN=RR.RobotRaconteurNode.s
import sys, time, traceback, threading
import numpy as np
from pysmu import Session, Mode

current_reading_interface="""
service experimental.current_reading

object current_obj
	wire double current [readonly]
end object
"""

class current_reading(object):
	def __init__(self):
		self._streaming=False
		self._capture_lock = threading.Lock()
		self.session = Session()
		if self.session.devices:
			# Grab the first device from the session.
			self.dev = self.session.devices[0]

			# Set both channels to high impedance mode.
			chan_a = self.dev.channels['A']
			chan_b = self.dev.channels['B']
			chan_a.mode = Mode.HI_Z
			chan_b.mode = Mode.HI_Z

			# Ignore read buffer sample drops when printing to stdout.
			self.dev.ignore_dataflow = sys.stdout.isatty()

			# Start a continuous session.
			self.session.start(0)
		else:
			raise Exception("No device found")
	
	def start_streaming(self):
		if (self._streaming):
			raise RR.InvalidOperationException("Already streaming")
		self._streaming=True
		t=threading.Thread(target=self.read_current)
		t.start()

	def stop_streaming(self):
		if (not self._streaming):
			raise RR.InvalidOperationException("Not streaming")
		self._streaming=False

	def read_current(self):
		while(self._streaming):
			with self._capture_lock:
				samples = np.array(self.dev.read(100, -1))   ###read 100 samples
				self.current.OutValue=np.average(1e3*samples[:,0,0])

def main():
	with RR.ServerNodeSetup("experimental.current_reading", 12182):
		#Register the service type
		RRN.RegisterServiceType(current_reading_interface)

		current_reading_inst=current_reading()
		
		#Register the service
		RRN.RegisterService("Current","experimental.current_reading.current_obj",current_reading_inst)

		current_reading_inst.start_streaming()
		try:
			input('press enter to quit')
		except:
			traceback.print_exc()
		finally:
			current_reading_inst.stop_streaming()

if __name__ == '__main__':
	main()