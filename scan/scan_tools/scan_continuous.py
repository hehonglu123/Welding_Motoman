from RobotRaconteur.Client import *
import threading
import time

class ContinuousScanner():
    def __init__(self,RRC) -> None:
        
        self.RRC=RRC
        self.capture_t = threading.Thread(target=self.capture_thread,daemon=True)
        self.capture_flag=False

        ### data
        self.scans=[]
        self.timestamps=[]

    def capture_thread(self):
        
        # data
        scans=[]
        timestamps=[]

        st = time.perf_counter()
        while self.capture_flag:
            scans.append(self.RRC.capture_deferred(False))
            timestamps.append(time.perf_counter()-st)

    def start_capture(self):

        self.capture_flag=True
        self.capture_t.start()
    
    def end_capture(self):
        
        self.capture_flag=False
        time.sleep(0.01)
    
    def get_capture(self):

        return self.scans,self.timestamps



if __name__=='__main__':
    
    c = RRN.ConnectService('rr+tcp://localhost:64238?service=scanner')
    cscanner = ContinuousScanner(c)

    cscanner.start_capture()
    time.sleep(5)
    cscanner.end_capture()
    scans,timestamps=cscanner.get_capture()