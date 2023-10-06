import RobotRaconteur as RR
RRN = RR.RobotRaconteurNode.s
import RobotRaconteurCompanion as RRC
import numpy as np
import time
from RobotRaconteurCompanion.Util.SensorDataUtil import SensorDataUtil
sys.path.append('../scan/scan_process/')
from scanProcess import *

import sys
sys.path.append('../process/')

class PCDProcesser(object):
    def __init__(self) -> None:
        scan_process = ScanProcess(None,None)

def main():
    
    with RR.ServerNodeSetup("PCDProcess_Service", 2355):      #setup RR node with service name and port
        #Register the service type
        RRN.RegisterServiceTypeFromFile("robdef/experimental.pcd_processer.robdef") 
        
        pcp_obj = PCDProcesser()
        RRN.RegisterService("pcdprocesser_service","experimental.pcd_processer.PCProcesser",pcp_obj)


        print("PCD Processer start")
        input("Press enter to quit")

if __name__ == "__main__":
    main()