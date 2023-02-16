from RobotRaconteur.Client import *
import cv2
import matplotlib.pyplot as plt
import time
from contextlib import suppress
import threading

# Capture frames using scanning procedure and save to project

#Project name must be unique!

project_name = "test_scan_temp"

c = RRN.ConnectService('rr+tcp://localhost:64238?service=scanner')

number_of_frames_to_capture = 10000

scan_proc_settings_type = RRN.GetStructureType("experimental.artec_scanner.ScanningProcedureSettings", c)
artec_consts = RRN.GetConstants("experimental.artec_scanner", c)
desc = scan_proc_settings_type()
desc.max_frame_count = number_of_frames_to_capture
desc.initial_state = artec_consts["ScanningState"]["record"]
desc.pipeline_configuration = artec_consts["ScanningPipeline"]["map_texture"] | \
    artec_consts["ScanningPipeline"]["find_geometry_key_frame"] | artec_consts["ScanningPipeline"]["register_frame"] | \
    artec_consts["ScanningPipeline"]["convert_textures"]
desc.capture_texture = artec_consts["CaptureTextureMethod"]["every_n_frame"]
desc.capture_texture_frequency = 10
desc.ignore_registration_errors = True

class ScanningProcedureRunner:
    def __init__(self, gen):
        self.gen = gen
        self.evt = threading.Event()
        self.err = None
        self.res = None
        self.gen.AsyncNext(None,self._handle_next)

    def _handle_next(self, res, err):
        print(f"_handle_next {res} {err}")
        if err:
            self.err = err
            self.evt.set()
            return
        if res.action_status == 3:
            self.res = res
            self.evt.set()
            return
        self.gen.AsyncNext(None,self._handle_next)
    
    def get_res(self, timeout = 60):
        self.evt.wait(timeout)
        return self.res
        


scan_gen = c.run_scanning_procedure(desc)

input("Press enter to start scanning")

scan_run = ScanningProcedureRunner(scan_gen)

input("Press enter to stop scanning")
scan_gen.Close()

scan_res = scan_run.get_res()

model_handle = scan_res.model_handle

print(model_handle.scan_count)

exit()

try:
    c.model_save(model_handle, project_name)
finally:
    c.model_free(model_handle)

