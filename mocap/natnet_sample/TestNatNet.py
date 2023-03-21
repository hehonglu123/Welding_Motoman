import sys
import time
from NatNetClient import NatNetClient
import DataDescriptions
import MoCapData

def print_configuration(natnet_client):
    print("Connection Configuration:")
    print("  Client:          %s"% natnet_client.local_ip_address)
    print("  Server:          %s"% natnet_client.server_ip_address)
    print("  Command Port:    %d"% natnet_client.command_port)
    print("  Data Port:       %d"% natnet_client.data_port)

    if natnet_client.use_multicast:
        print("  Using Multicast")
        print("  Multicast Group: %s"% natnet_client.multicast_address)
    else:
        print("  Using Unicast")

    #NatNet Server Info
    application_name = natnet_client.get_application_name()
    nat_net_requested_version = natnet_client.get_nat_net_requested_version()
    nat_net_version_server = natnet_client.get_nat_net_version_server()
    server_version = natnet_client.get_server_version()

    print("  NatNet Server Info")
    print("    Application Name %s" %(application_name))
    print("    NatNetVersion  %d %d %d %d"% (nat_net_version_server[0], nat_net_version_server[1], nat_net_version_server[2], nat_net_version_server[3]))
    print("    ServerVersion  %d %d %d %d"% (server_version[0], server_version[1], server_version[2], server_version[3]))
    print("  NatNet Bitstream Requested")
    print("    NatNetVersion  %d %d %d %d"% (nat_net_requested_version[0], nat_net_requested_version[1],\
       nat_net_requested_version[2], nat_net_requested_version[3]))
    #print("command_socket = %s"%(str(natnet_client.command_socket)))
    #print("data_socket    = %s"%(str(natnet_client.data_socket)))

def print_mocap_data(mocap):

    if mocap is not None:
        out_str = "Frame # %3.1d\n"%(mocap.prefix_data.frame_number)
        out_str += "Rigid Body Count: %3.1d\n"%(len(mocap.rigid_body_data.rigid_body_list))
        for i in range(len(mocap.rigid_body_data.rigid_body_list)):
            rigid_body = mocap.rigid_body_data.rigid_body_list[i]
            out_str += "  Rigid Body ID: %3.1d\n"%(rigid_body.id_num)
            out_str += "  Pos:[%3.2f, %3.2f, %3.2f]\n"%(rigid_body.pos[0],rigid_body.pos[1],rigid_body.pos[2])
            out_str += "  Rot:[%3.2f, %3.2f, %3.2f]\n"%(rigid_body.rot[0],rigid_body.rot[1],rigid_body.rot[2])
            out_str += "  Marker Count: %3.1d\n"%(len(rigid_body.rb_marker_list))
        out_str += "Labeled Markers Count: %3.1d\n"%(len(mocap.labeled_marker_data.labeled_marker_list))
        for i in range(len(mocap.labeled_marker_data.labeled_marker_list)):
            lbmarker = mocap.labeled_marker_data.labeled_marker_list[i]
            marker_id,model_id = lbmarker.get_marker_id()
            out_str += "  Maker ID: %3.1d\n"%(marker_id)
            out_str += "  Model ID: %3.1d\n"%(model_id) # which rigid body it's belongs to.
            out_str += "  Pos:[%3.2f, %3.2f, %3.2f]\n"%(lbmarker.pos[0],lbmarker.pos[1],lbmarker.pos[2])
            out_str += "  Size: %3.2f\n"%(lbmarker.size)
            out_str += "  Error (Residual): %3.2f\n"%(lbmarker.residual)
        out_str += "===============================\n"
    else:
        out_str = 'No Mocap Data. Perhaps Motiv is not streaming.'
    print(out_str)

optionsDict = {}
optionsDict["clientAddress"] = "127.0.0.1"
optionsDict["serverAddress"] = "127.0.0.1"
optionsDict["use_multicast"] = True

## setup NatNetClient obj
streaming_client = NatNetClient()
streaming_client.set_client_address(optionsDict["clientAddress"])
streaming_client.set_server_address(optionsDict["serverAddress"])
streaming_client.set_use_multicast(optionsDict["use_multicast"])

is_running = streaming_client.run_mocap()
if not is_running:
    print("ERROR: Could not start streaming client.")
    try:
        sys.exit(1)
    except SystemExit:
        print("...")
    finally:
        print("exiting")

time.sleep(1)
if streaming_client.connected() is False:
    print("ERROR: Could not connect properly.  Check that Motive streaming is on.")
    try:
        sys.exit(2)
    except SystemExit:
        print("...")
    finally:
        print("exiting")

print_configuration(streaming_client)
print("\n")

## point to mocap data
print("Print Mocap Data Every 1 sec....")
time.sleep(1)
print_mocap_data(streaming_client.mocap_data)
time.sleep(1)
print_mocap_data(streaming_client.mocap_data)
time.sleep(1)
print_mocap_data(streaming_client.mocap_data)

print("Get Data Description")

streaming_client.shutdown()