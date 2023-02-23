from RobotRaconteur.Client import *
import threading
import time
from contextlib import suppress

class ContinuousScanner():
    def __init__(self,RRC) -> None:
        
        self.RRC=RRC
        self.capture_t = threading.Thread(target=self.capture_thread,daemon=True)
        self.capture_flag=False

        ### data
        self.scan_handles=[]
        self.timestamps=[]

    def capture_thread(self):
        
        # data
        scans=[]
        timestamps=[]

        st = time.perf_counter()
        while self.capture_flag:
            scans.append(self.RRC.capture_deferred(False))
            timestamps.append(time.perf_counter()-st)
        
        self.scan_handles=scans
        self.timestamps=timestamps

    def start_capture(self):

        self.capture_flag=True
        self.capture_t.start()
    
    def end_capture(self):
        
        self.capture_flag=False
        time.sleep(1)
    
    def get_capture(self):

        prepare_gen = self.RRC.deferred_capture_prepare_stl(self.scan_handles)
        with suppress(RR.StopIterationException):
            prepare_res = prepare_gen.Next()
        
        scans=[]
        for i in range(len(self.scan_handles)):
            stl_mesh = self.RRC.getf_deferred_capture(self.scan_handles[i])
            scans.append(stl_mesh)

        return scans,self.timestamps



if __name__=='__main__':
    
    c = RRN.ConnectService('rr+tcp://192.168.55.27:64238?service=scanner')

    cscanner = ContinuousScanner(c)

    cscanner.start_capture()
    time.sleep(5)
    cscanner.end_capture()
    scans_meshes,timestamps=cscanner.get_capture()

    print(scans_meshes[0].vertices)