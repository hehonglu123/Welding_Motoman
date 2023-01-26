from RobotRaconteur.Client import *
import cv2
import matplotlib.pyplot as plt
import time
from contextlib import suppress

output_project_name = "test_alg_3"

# Test running some algorithms
# Based on scanning-and-process-sample.cpp
# http://docs.artec-group.com/sdk/2.0/scanning-and-process-sample_8cpp-example.html
c = RRN.ConnectService('rr+tcp://localhost:64238?service=scanner')

# Load saved data to use. test_scan10 is a full scan
input_model_handle = c.model_load("test_scan10")

artec_consts = RRN.GetConstants("experimental.artec_scanner", c)

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
ser_reg = c.initialize_algorithm(input_model_handle, "SerialRegistrationAlgorithm")
ser_reg.data.registration_type = artec_consts["SerialRegistrationType"]["rough"]
algs.append(ser_reg)

# Proceed with global registration

# Global registration does not seem to be completing? Hanging with 100% CPU usage
# glb_reg = c.initialize_algorithm(input_model_handle, "GlobalRegistrationAlgorithm")
# glb_reg.data.registration_type = artec_consts["GlobalRegistrationType"]["geometry"]
# algs.append(glb_reg)

# Apply outliers removal
out_rem = c.initialize_algorithm(input_model_handle, "OutliersRemovalAlgorithm")
algs.append(out_rem)

# Apply fast fusion
fast_fus = c.initialize_algorithm(input_model_handle, "FastFusionAlgorithm")
fast_fus.data.resolution = 2
algs.append(fast_fus)

# Run the algorithms
alg_gen = c.run_algorithms(input_model_handle, algs)

with suppress(RR.StopIterationException):
    while True:
        alg_res = alg_gen.Next()
        print(f"alg_res: {alg_res.action_status}, {alg_res.current_algorithm}")

# Get and save the result
output_model_handle = alg_res.output_model_handle

print(output_model_handle)

try:
    # Save the model to an artec project file
    c.model_save(output_model_handle, output_project_name)

    # Retrieve and save the composite mesh result from algorithms
    model = c.get_models(output_model_handle)
    container = model.get_composite_container()
    stl_data = container.getf_composite_mesh_stl(0)
    with open("test_alg_res.stl", "wb") as f:
        f.write(stl_data.tobytes())

finally:
    # Free the model from memory
    c.model_free(input_model_handle)
    c.model_free(output_model_handle)



