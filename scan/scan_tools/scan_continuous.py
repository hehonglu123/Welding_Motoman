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

        while self.capture_flag:
            scans.append(self.RRC.capture_deferred(False))
        
        self.scan_handles=scans

    def start_capture(self):

        self.capture_flag=True
        self.capture_t.start()
    
    def end_capture(self):
        
        self.capture_flag=False
        time.sleep(1)
    
    def get_capture(self):

        prepare_gen = self.RRC.deferred_capture_prepare(self.scan_handles)
        with suppress(RR.StopIterationException):
            prepare_res = prepare_gen.Next()
        
        scans=[]
        timestamps=[]
        for i in range(len(self.scan_handles)):
            try:
                stl_mesh = self.RRC.getf_deferred_capture(self.scan_handles[i])
                scans.append(stl_mesh)
                framestamp=self.RRC.getf_deferred_capture_stamps(self.scan_handles[i])
                timestamps.append(framestamp.seconds+framestamp.micro_seconds*1e-6)
            except:
                print("Fail to capture frame",i)
                continue

        return scans,timestamps



if __name__=='__main__':
    
    c = RRN.ConnectService('rr+tcp://localhost:64238?service=scanner')

    cscanner = ContinuousScanner(c)

    cscanner.start_capture()
    time.sleep(10)
    cscanner.end_capture()

    st=time.perf_counter()
    scans_meshes,timestamps=cscanner.get_capture()
    dt=time.perf_counter()-st
    print("prepare mesh dt:",dt)

    print(timestamps)