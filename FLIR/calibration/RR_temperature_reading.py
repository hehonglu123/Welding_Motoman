import RobotRaconteur as RR
RRN=RR.RobotRaconteurNode.s
import sys, time, traceback, threading
import numpy as np
from pysmu import Session, Mode
from thermal_couple_conversion import voltage_to_temperature

temperature_reading_interface="""
service experimental.temperature_reading

object temperature_obj
	wire double temperature [readonly]
end object
"""

class temperature_reading(object):
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
	
	def start_streaming(self):
		if (self._streaming):
			raise RR.InvalidOperationException("Already streaming")
		self._streaming=True
		t=threading.Thread(target=self.read_temperature)
		t.start()

	def stop_streaming(self):
		if (not self._streaming):
			raise RR.InvalidOperationException("Not streaming")
		self._streaming=False

	def read_temperature(self):
		while(self._streaming):
			with self._capture_lock:
				samples = np.array(self.dev.read(100, -1))   ###read 100 samples
				sample=np.average(samples[:,0,0])
				# print(sample)
				# self.temperature.OutValue=voltage_to_temperature(1e6*sample)
				self.temperature.OutValue=1e6*sample

def main():
	with RR.ServerNodeSetup("experimental.temperature_reading", 12182):
		#Register the service type
		RRN.RegisterServiceType(temperature_reading_interface)

		temperature_reading_inst=temperature_reading()
		
		#Register the service
		RRN.RegisterService("Temperature","experimental.temperature_reading.temperature_obj",temperature_reading_inst)

		temperature_reading_inst.start_streaming()
		try:
			input('press enter to quit')
		except:
			traceback.print_exc()
		finally:
			temperature_reading_inst.stop_streaming()

if __name__ == '__main__':
	main()