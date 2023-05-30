import sys
sys.path.append('../toolbox/')
# from robot_def import *
from dx200_motion_program_exec_client import *
from RobotRaconteur.Client import *
import cv2
import matplotlib.pyplot as plt
import time
from contextlib import suppress
import threading


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


# q1=np.array([43.5893,72.1362,45.2749,-84.0966,24.3644,94.2091])
# q2=np.array([34.6291,55.5756,15.4033,-28.8363,24.0298,3.6855])
# q3=np.array([27.3821,51.3582,-19.8428,-21.2525,71.6314,-62.8669])

# target2=['MOVJ',[-15,180],[-15,160],[-15,140],1,0]
# target2J_1=['MOVJ',[-15,180],1,0]
# target2J_2=['MOVJ',[-15,140],1,0]
# target2J_3=['MOVJ',[-15,100],1,0]

# robot_client=MotionProgramExecClient(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=[1.435355447016790322e+03,1.300329111270902331e+03,1.422225409601069941e+03,9.699560942607320158e+02,9.802408285708806943e+02,4.547552630640436178e+02],pulse2deg_2=[1994.3054,1376.714])
scan_client=RRN.ConnectService('rr+tcp://localhost:64238?service=scanner')


# robot_client.MoveJ(q1, 1,0,target2=target2J_1)
# robot_client.ProgEnd()
# robot_client.execute_motion_program()

# robot_client=MotionProgramExecClient(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=[1.435355447016790322e+03,1.300329111270902331e+03,1.422225409601069941e+03,9.699560942607320158e+02,9.802408285708806943e+02,4.547552630640436178e+02],pulse2deg_2=[1994.3054,1376.714])
# robot_client.MoveL(q2, 10,0,target2=target2J_2)
# robot_client.MoveL(q3, 10,0,target2=target2J_3)
# robot_client.ProgEnd()




now=time.time()

output_project_name = "test1234567891011"

number_of_frames_to_capture = 300

scan_proc_settings_type = RRN.GetStructureType("experimental.artec_scanner.ScanningProcedureSettings", scan_client)
artec_consts = RRN.GetConstants("experimental.artec_scanner", scan_client)
desc = scan_proc_settings_type()
desc.max_frame_count = number_of_frames_to_capture
desc.initial_state = artec_consts["ScanningState"]["record"]
desc.pipeline_configuration = artec_consts["ScanningPipeline"]["map_texture"] | \
    artec_consts["ScanningPipeline"]["find_geometry_key_frame"] | artec_consts["ScanningPipeline"]["register_frame"] | \
    artec_consts["ScanningPipeline"]["convert_textures"]
desc.capture_texture = artec_consts["CaptureTextureMethod"]["every_n_frame"]
desc.capture_texture_frequency = 10
desc.ignore_registration_errors = True

scan_gen = scan_client.run_scanning_procedure(desc)
scan_run = ScanningProcedureRunner(scan_gen)


# robot_client.execute_motion_program()

scan_gen.Close()

scan_res = scan_run.get_res()

model_handle = scan_res.model_handle



# Build up list of algorithms to run on data
algs = []

# Valid algoritms to use with run_algorithms are:
# AutoAlignAlgorithm, FastFusionAlgorithm, FastMeshSimplificationAlgorithm, GlobalRegistrationAlgorithm,
# LoopClosureAlgorithm, MeshSimplificationAlgorithm, OutliersRemovalAlgorithm, PoissonFusionAlgorithm,
# SerialRegistrationAlgorithm, SmallObjectsFilterAlgorithm, TexturizationAlgorithm
# 
# See http://docs.artec-group.com/sdk/2.0/namespaceartec_1_1sdk_1_1algorithms.html
# See https://github.com/johnwason/artec_scanner_robotraconteur_driver/blob/master/robdef/experimental.artec_scanner.robdef

# Apply serial registration
ser_reg = scan_client.initialize_algorithm(model_handle, "SerialRegistrationAlgorithm")
ser_reg.data.registration_type = artec_consts["SerialRegistrationType"]["rough"]
algs.append(ser_reg)

# Proceed with global registration

# Global registration does not seem to be completing? Hanging with 100% CPU usage
# glb_reg = scan_client.initialize_algorithm(model_handle, "GlobalRegistrationAlgorithm")
# glb_reg.data.registration_type = artec_consts["GlobalRegistrationType"]["geometry"]
# algs.append(glb_reg)

# Apply outliers removal
out_rem = scan_client.initialize_algorithm(model_handle, "OutliersRemovalAlgorithm")
algs.append(out_rem)

# Apply fast fusion
fast_fus = scan_client.initialize_algorithm(model_handle, "FastFusionAlgorithm")
fast_fus.data.resolution = 2
algs.append(fast_fus)

# Run the algorithms
alg_gen = scan_client.run_algorithms(model_handle, algs)

with suppress(RR.StopIterationException):
    while True:
        alg_res = alg_gen.Next()
        print(f"alg_res: {alg_res.action_status}, {alg_res.current_algorithm}")

# Get and save the result
output_model_handle = alg_res.output_model_handle

print(output_model_handle)

try:
    # Save the model to an artec project file
    scan_client.model_save(output_model_handle, output_project_name)

    # Retrieve and save the composite mesh result from algorithms
    model = scan_client.get_models(output_model_handle)
    container = model.get_composite_container()
    stl_data = container.getf_composite_mesh_stl(0)
    with open("test_alg_res.stl", "wb") as f:
        f.write(stl_data.tobytes())

finally:
    # Free the model from memory
    scan_client.model_free(model_handle)
    scan_client.model_free(output_model_handle)

print('Time Elapsed: ',time.time()-now)