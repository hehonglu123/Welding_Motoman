import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import LinearNDInterpolator
import pickle
from general_robotics_toolbox import *
from PH_interp import *
from calib_analytic_grad import *
import datetime
import time
import os, pathlib

from motoman_def import *
from Models import *

def LinearNDInterpNearestExtrap(points, values):
    q2 = points[:, 0]
    q3 = points[:, 1]
    f = LinearNDInterpolator(points, values)

    # this inner function will be returned to a user
    def new_f(q2q3):
        # evaluate the Linear interpolator. Out-of-bounds values are nan.
        zz = f(q2q3)
        if np.isnan(zz).any():
            # for each nan point, find its nearest neighbor
            inds = np.argmin(np.linalg.norm(points - q2q3, ord=2, axis=1))
            # ... and use its value
            zz= np.array([values[inds]])
        return zz
    
    print("Init interpolation done")
    return new_f

def test_fourier_accuracy(weights, data_q, data_T,robot,param_nominal):

    basis_func=[]
    basis_func.append(lambda q2,q3,a: np.sin(a*q2))
    basis_func.append(lambda q2,q3,a: np.sin(a*q3))
    basis_func.append(lambda q2,q3,a: np.cos(a*q2))
    basis_func.append(lambda q2,q3,a: np.cos(a*q3))
    basis_func.append(lambda q2,q3,a: np.sin(a*(q2+q3)))
    basis_func.append(lambda q2,q3,a: np.cos(a*(q2+q3)))
    
    p_error_all = []
    for i,q in enumerate(data_q):
        this_basis = []
        for a in range(1,3):
            for func in basis_func:
                # print(np.degrees([q[0],q[1]]))
                # print(func(q[0],q[1],a))
                # input("========================")
                this_basis.append(func(q[1],q[2],a))
        this_basis.append(1) # constant function

        pred_PH = weights@this_basis + param_nominal
        
        robot = get_PH_from_param(pred_PH,robot,unit='radians')
        T_pred = robot.fwd(q)
        p_error = np.linalg.norm(T_pred.p - data_T[i][:3])
        p_error_all.append(p_error)
    return p_error_all

def test_fwd_interp_accuracy(model, vae_model, train_q, data_q, data_T,robot,param_nominal):
    
    p_error_all = []
    for i,q in enumerate(data_q):
        # first, interpolate q2q3 to get latent vector
        q2q3 = np.array([q[1],q[2]])
        latent_vec_hat = model(torch.tensor(q2q3, dtype=torch.float32))
        # then, predict delta PH using the latent vector and the decoder  
        pred_PH = vae_model.decoder(latent_vec_hat)
        pred_PH = pred_PH.detach().numpy() + param_nominal
        # get the robot position error
        robot = get_PH_from_param(pred_PH,robot,unit='radians')
        T_pred = robot.fwd(q)
        p_error = np.linalg.norm(T_pred.p - data_T[i][:3])
        p_error_all.append(p_error)
    return p_error_all

def test_fwd_accuracy(model, interp_funcs, train_q, data_q, data_T,robot,param_nominal,q_index=np.array([1,2])):
    
    p_error_all = []
    ori_error_all = []
    for i,q in enumerate(data_q):
        # first, interpolate q2q3 to get latent vector
        # q2q3 = np.array([q[1],q[2]])
        q_input = np.array(q[q_index])
        latent_vec_pred = np.array([interp_func(q_input) for interp_func in interp_funcs]).T
        if np.isnan(latent_vec_pred).any():
            raise ValueError('Interpolation failed')
            
        latent_vec_pred = torch.tensor(latent_vec_pred[0], dtype=torch.float32)
        # then, predict delta PH using the latent vector and the decoder  
        pred_PH = model.decoder(latent_vec_pred)
        pred_PH = pred_PH.detach().numpy() + param_nominal
        # get the robot position error
        robot = get_PH_from_param(pred_PH,robot,unit='radians')
        T_pred = robot.fwd(q)
        p_error = np.linalg.norm(T_pred.p - data_T[i][:3])
        k,theta = R2rot(T_pred.R@q2R(data_T[i][3:]).T)
        ori_error = np.degrees(np.linalg.norm(k*theta))
        p_error_all.append(p_error)
        ori_error_all.append(np.abs(ori_error))
    return p_error_all,ori_error_all

def latent_space_analysis(inputs_q2q3, data_delta_PH, training_q, training_T, testing_q, testing_T,robot,param_nominal,robot_type):

    # vae_model_dir = 'trainLATENT_AE_R1_latent6_2411121051/'
    vae_model_name = 'trainLATENT_AE_R1_latent12_2411121110/'
    vae_model_dir = 'PH_NN_results/'+vae_model_name

    # tensorize the data
    data_delta_PH = torch.tensor(data_delta_PH, dtype=torch.float32)
    inputs_q2q3_tensor = torch.tensor(inputs_q2q3, dtype=torch.float32)

    # read meta data
    with open(vae_model_dir+'meta_data.yaml') as file:
        vae_meta_data = yaml.full_load(file)
    # read vae model
    # Create an instance of the neural network
    if vae_meta_data['Variational']:
        vae_model = VariationalAutoEncoder(vae_meta_data['data_size'], vae_meta_data['latent_size'], vae_meta_data['hidden_sizes'], mu=vae_meta_data['mu'], sigma=vae_meta_data['sigma'])
    else:
        vae_model = AutoEncoder(vae_meta_data['data_size'], vae_meta_data['latent_size'], vae_meta_data['hidden_sizes'])
    vae_model.load_state_dict(torch.load(vae_model_dir+'best_testing_model.pt',weights_only=True))
    vae_model.eval()

    # fourier model
    fourier_model = FourierNetwork(2, 33)
    fourier_model.eval()

    # NN model
    nn_model_name = 'train_200_200_200_lr0.02_2409171041/'
    nn_model_dir = 'PH_NN_results/'+nn_model_name
    # read meta data
    with open(nn_model_dir+'meta_data.yaml') as file:
        nn_meta_data = yaml.full_load(file)
    nn_model = NeuralNetwork(nn_meta_data['input_size'], nn_meta_data['output_size'], nn_meta_data['hidden_sizes'])
    nn_model.load_state_dict(torch.load(nn_model_dir+'best_testing_model.pt',weights_only=True))
    nn_model.eval()
    
    input_size = inputs_q2q3.shape[1]
    latent_size = vae_meta_data['latent_size']

    # forward pass
    vae_model.eval()
    latent_vec = vae_model.encoder(data_delta_PH)
    latent_vec_cpu = latent_vec.detach().numpy()
    print(latent_vec_cpu.shape)

    # fourier latent vectors
    fourier_latent_vec = fourier_model.forward_features(inputs_q2q3_tensor)
    fourier_latent_vec_cpu = fourier_latent_vec.detach().numpy()
    # stack 1 at every latent vectors
    # fourier_latent_vec_cpu = np.hstack((fourier_latent_vec_cpu,np.ones((fourier_latent_vec_cpu.shape[0],1))))
    print(fourier_latent_vec_cpu.shape)
    # NN latent vectors
    nn_latent_vec = nn_model.forward_features(inputs_q2q3_tensor)
    nn_latent_vec_cpu = nn_latent_vec.detach().numpy()

    # svd analysis of the latent space
    _, s_vae_latent, _ = np.linalg.svd(latent_vec_cpu.T, full_matrices=True)
    _, s_fourier_latent, _ = np.linalg.svd(fourier_latent_vec_cpu.T, full_matrices=True)
    plt.plot(np.log10(s_vae_latent/np.max(s_vae_latent)), '-o', label='AE latent')
    plt.plot(np.log10(s_fourier_latent/np.max(s_fourier_latent)), '-o', label='Fourier latent')
    latent_combine = np.hstack((latent_vec_cpu/np.max(s_vae_latent),fourier_latent_vec_cpu/np.max(s_fourier_latent)))
    _, s_combine, _ = np.linalg.svd(latent_combine.T, full_matrices=True)
    plt.plot(np.log10(s_combine/np.max(s_combine)), '-o', label='Combined latent')
    _, s_nn_latent, _ = np.linalg.svd(nn_latent_vec_cpu.T, full_matrices=True)
    # plt.plot(np.log10(s_nn_latent), '-o', label='NN latent')
    plt.legend(fontsize=14)
    plt.ylabel('Normalized Singular value (log10)', fontsize=16, fontweight='bold')
    plt.xlabel('Singular value index', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Normalized Singular value analysis of latent space', fontsize=18, fontweight='bold')
    plt.grid()
    plt.show()

    # plot the latent space w.r.t q2q3
    plot_row = 2
    plot_col = 6
    fig, axs = plt.subplots(plot_row, plot_col)
    # set figure size
    # fig.set_size_inches(18.5, 10.5)
    for i in range(plot_row*plot_col):
        axs[i//plot_col,i%plot_col].scatter(np.degrees(inputs_q2q3[:,0]),np.degrees(inputs_q2q3[:,1]),c=latent_vec_cpu[:,i])
        axs[i//plot_col,i%plot_col].tick_params(axis='both', which='major', labelsize=12)
        axs[i//plot_col,i%plot_col].set_title('Latent '+str(i+1),fontsize=13, fontweight='bold')
        if i>=(plot_row-1)*plot_col:
            axs[i//plot_col,i%plot_col].set_xlabel('q2',fontsize=12, fontweight='bold')
        else:
            # remove xticks
            axs[i//plot_col,i%plot_col].set_xticks([])
        if i%plot_col==0:
            axs[i//plot_col,i%plot_col].set_ylabel('q3',fontsize=12, fontweight='bold')
    # show the color bar and adjust the color bar size and position
    cbar = fig.colorbar(axs[0,0].collections[0], ax=axs, orientation='horizontal', anchor=(0.5, 2), fraction=0.1, shrink=0.5)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Latent value', fontsize=14, fontweight='bold')
    # figure title
    fig.suptitle('Autoencoder latent space vs q2,q3',fontsize=20, fontweight='bold')
    plt.show()

def trained_model_test(inputs_q2q3, data_delta_PH, training_q, training_T, testing_q, testing_T,robot,param_nominal,robot_type):

    # data preprocessing
    N_per_cluster = 7
    q_index = np.arange(1,3)
    data_delta_PH = torch.tensor(data_delta_PH, dtype=torch.float32)
    inputs_q2q3_tensor = torch.tensor(inputs_q2q3, dtype=torch.float32)
    # augmented inputs
    inputs_qall = []
    data_delta_PH_qall = []
    for i,q2q3 in enumerate(inputs_q2q3):
        nearest_q_index = np.argsort(np.linalg.norm(training_q[:,1:3] - q2q3, ord=2, axis=1))
        inputs_qall.extend(training_q[nearest_q_index[:N_per_cluster]][:,q_index])
        data_delta_PH_qall.extend(np.tile(data_delta_PH[i],(N_per_cluster,1)))
    inputs_qall = np.array(inputs_qall)
    data_delta_PH_qall = np.array(data_delta_PH_qall)
    inputs_qall_tensor = torch.tensor(inputs_qall, dtype=torch.float32)
    data_delta_PH_qall_tensor = torch.tensor(data_delta_PH_qall, dtype=torch.float32)

    # AE_model_dir = "trainLATENT_VAE_R1_latent6_2411121154/"
    # AE_model_dir = 'trainLATENT_AE_R1_latent6_2411121051/'
    # AE_model_dir = 'trainLATENT_AE_R2_latent6_2411201610/'
    # AE_model_dir = 'trainLATENT_AE_R1_latent6_weighted_2503091944/'
    AE_model_dir = 'trainLATENT_AE_R2_latent6_weighted_2503092025/'

    data_dir = 'PH_NN_results/'+AE_model_dir
    # read meta data
    with open(data_dir+'meta_data.yaml') as file:
        vae_meta_data = yaml.full_load(file)
    # read vae model
    # Create an instance of the neural network
    if vae_meta_data['Variational']:
        vae_model = VariationalAutoEncoder(vae_meta_data['data_size'], vae_meta_data['latent_size'], vae_meta_data['hidden_sizes'], mu=vae_meta_data['mu'], sigma=vae_meta_data['sigma'])
    else:
        vae_model = AutoEncoder(vae_meta_data['data_size'], vae_meta_data['latent_size'], vae_meta_data['hidden_sizes'])
    vae_model.load_state_dict(torch.load(data_dir+'best_testing_model.pt',weights_only=True))
    vae_model.eval()

    latent_size = vae_meta_data['latent_size']

    # get linear interpolation functions
    latent_vec = vae_model.encoder(data_delta_PH_qall_tensor)
    latent_vec_cpu = latent_vec.detach().numpy()
    interp_funcs = []
    print("Get linear interpolation functions")
    for latent_i in range(latent_size):
        interp_funcs.append(LinearNDInterpNearestExtrap(inputs_qall, latent_vec_cpu[:,latent_i]))
    print("Interpolation functions done")
    # get data accuracy
    training_T_error,training_ori_error = test_fwd_accuracy(vae_model, interp_funcs, inputs_qall, training_q, training_T,robot,param_nominal,q_index)
    testing_T_error,testing_ori_error = test_fwd_accuracy(vae_model, interp_funcs, inputs_qall, testing_q, testing_T,robot,param_nominal,q_index)
    # print training and testing error, mean, max
    print(f'Training error: mean={np.mean(training_T_error):.4f}, max={np.max(training_T_error):.4f}')
    print(f'Testing error: mean={np.mean(testing_T_error):.4f}, max={np.max(testing_T_error):.4f}')

    print(f'Max testing error: {round(np.max(testing_T_error),2):.2f}')
    print(f'Mean testing error: {round(np.mean(testing_T_error),2):.2f}')
    print(f'Std testing error: {round(np.std(testing_T_error),2):.2f}')
    print(f'Max testing ori error: {round(np.max(testing_ori_error),2):.2f}')
    print(f'Mean testing ori error: {round(np.mean(testing_ori_error),2):.2f}')
    print(f'Std testing ori error: {round(np.std(testing_ori_error),2):.2f}')

def train_interp(inputs_q2q3, data_delta_PH, training_q, training_T, testing_q, testing_T,robot,param_nominal,robot_type):

    print("Train interpolation model")

    # data preprocessing
    data_delta_PH = torch.tensor(data_delta_PH, dtype=torch.float32)
    inputs_q2q3_tensor = torch.tensor(inputs_q2q3, dtype=torch.float32)

    # AE_model_dir = "trainLATENT_AE_R1_latent12_2411121110/"
    AE_model_dir = "trainLATENT_VAE_R1_latent6_2411121154/"
    data_dir = 'PH_NN_results/'+AE_model_dir
    # read meta data
    with open(data_dir+'meta_data.yaml') as file:
        vae_meta_data = yaml.full_load(file)
    # read vae model
    # Create an instance of the neural network
    if vae_meta_data['Variational']:
        vae_model = VariationalAutoEncoder(vae_meta_data['data_size'], vae_meta_data['latent_size'], vae_meta_data['hidden_sizes'], mu=vae_meta_data['mu'], sigma=vae_meta_data['sigma'])
    else:
        vae_model = AutoEncoder(vae_meta_data['data_size'], vae_meta_data['latent_size'], vae_meta_data['hidden_sizes'])
    vae_model.load_state_dict(torch.load(data_dir+'best_testing_model.pt',weights_only=True))
    vae_model.eval()


    input_size = inputs_q2q3.shape[1]
    latent_size = vae_meta_data['latent_size']
    # model parameters
    hidden_sizes = [200,200,200]
    # model
    model = NeuralNetwork(input_size, latent_size, hidden_sizes)
    # print the model architecture
    print("Interpolator:", model)

    # loss function
    loss_fn = nn.MSELoss()
    # Define the learning rate
    learning_rate = 0.003
    # Define the number of epochs
    num_epochs = 50000
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # get save folder path
    formatted_string = datetime.datetime.now().strftime("%Y%m%d%H%M")
    formatted_string = formatted_string[2:]
    folder_path = 'PH_NN_results/trainINTERP_'
    folder_path += robot_type+'_'
    folder_path += formatted_string+'/'

    # save a training parameters meta yaml file to folder_path
    meta_data = {'AE_data': AE_model_dir, 'hidden_sizes': hidden_sizes, 'learning_rate': learning_rate, 'num_epochs': num_epochs}
    meta_data['robot_type'] = robot_type
    if not os.path.exists(folder_path):
        pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
    with open(folder_path+'meta_data.yaml', 'w') as file:
        documents = yaml.dump(meta_data, file)

    # Training loop
    loss_all = []
    training_mean_error_all = []
    testing_mean_error_all = []
    training_max_error_all = []
    testing_max_error_all = []
    training_std_error_all = []
    testing_std_error_all = []
    data_sample_epoches = []
    best_loss = 1e10
    best_training_error = 1e10
    best_testing_error = 1e10

    training_start_time = time.time()
    training_t_epoch = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # get ground truth from ae
        latent_vec = vae_model.encoder(data_delta_PH)

        # Forward pass
        model.train()
        latent_vec_hat = model(inputs_q2q3_tensor)
        loss = loss_fn(latent_vec_hat, latent_vec)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get testing data loss
        # test_outputs = model(test_inputs_q2q3)
        # test_loss = loss_fn(test_outputs, test_targets_delta_PH)

        # Print the loss for every 10 epochs
        print_loss = False
        print_error = False
        if epoch==0:
            print_loss = True
            print_error = True
        elif epoch<1001:
            if (epoch+1) % 10 == 0:
                print_loss = True
            if (epoch+1) % 100 == 0:
                print_error = True
        else:
            if (epoch+1) % 100 == 0:
                print_loss = True
            if (epoch+1) % 500 == 0:
                print_error = True

        model.eval()
        if best_loss > loss.item():
            best_loss = loss.item()
            torch.save(model.state_dict(), folder_path+'best_lost_model.pt')
        loss_all.append(loss.item())
        np.save(folder_path+'loss_all.npy',np.array(loss_all)) # save the loss
        if print_loss:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        if print_error:
            training_T_error = test_fwd_interp_accuracy(model, vae_model, inputs_q2q3, training_q, training_T,robot,param_nominal)
            testing_T_error = test_fwd_interp_accuracy(model, vae_model, inputs_q2q3, testing_q, testing_T,robot,param_nominal)
            # print training and testing error, mean, max
            print(f'Training error: mean={np.mean(training_T_error):.4f}, max={np.max(training_T_error):.4f}')
            print(f'Testing error: mean={np.mean(testing_T_error):.4f}, max={np.max(testing_T_error):.4f}')
            training_mean_error_all.append(np.mean(training_T_error))
            testing_mean_error_all.append(np.mean(testing_T_error))
            training_max_error_all.append(np.max(training_T_error))
            testing_max_error_all.append(np.max(testing_T_error))
            training_std_error_all.append(np.std(training_T_error))
            testing_std_error_all.append(np.std(testing_T_error))
            data_sample_epoches.append(epoch)
            # save the model
            if best_training_error > np.max(training_T_error):
                best_training_error = np.max(training_T_error)
                torch.save(model.state_dict(), folder_path+'best_training_model.pt')
            if best_testing_error > np.max(testing_T_error):
                best_testing_error = np.max(testing_T_error)
                torch.save(model.state_dict(), folder_path+'best_testing_model.pt')
            np.save(folder_path+'training_mean_error_all.npy',np.array(training_mean_error_all))
            np.save(folder_path+'testing_mean_error_all.npy',np.array(testing_mean_error_all))
            np.save(folder_path+'training_max_error_all.npy',np.array(training_max_error_all))
            np.save(folder_path+'testing_max_error_all.npy',np.array(testing_max_error_all))
            np.save(folder_path+'training_std_error_all.npy',np.array(training_std_error_all))
            np.save(folder_path+'testing_std_error_all.npy',np.array(testing_std_error_all))
            np.save(folder_path+'data_sample_epoches.npy',np.array(data_sample_epoches))

        # training time for each epoch
        epoch_end_time = time.time()
        training_t_epoch.append(epoch_end_time-epoch_start_time)
        # print(f'Mean epoch time: {np.mean(training_t_epoch):.5f}, Total time: {epoch_end_time-training_start_time:.2f}')

    print('Training time:',time.time()-training_start_time)  

def train(inputs_q2q3, data_delta_PH, training_q, training_T, testing_q, testing_T,robot,param_nominal,robot_type):

    # data preprocessing
    data_delta_PH = torch.tensor(data_delta_PH, dtype=torch.float32)
    inputs_q2q3_tensor = torch.tensor(inputs_q2q3, dtype=torch.float32)
    

    # Define the input size, hidden size, and output size
    latent_size = 6 # 2 6 12
    hidden_sizes = [200,200,200]
    data_size = 33
    mu = 0
    sigma = 0.001
    loss_kl_weight = 0.1
    Variational = False

    # Create an instance of the neural network
    if Variational:
        model = VariationalAutoEncoder(data_size, latent_size, hidden_sizes, mu=mu, sigma=sigma)
    else:
        model = AutoEncoder(data_size, latent_size, hidden_sizes)

    # Print the model architecture
    print("Encoder:", model.encoder)
    print("Decoder:", model.decoder)
    # Define the loss function
    loss_mse_fn = nn.MSELoss()
    weighted = True
    if weighted:
        loss_mse_fn = WeightedMSELoss()
        # weights = torch.tensor([1]*33, dtype=torch.float32)
        weights_P = 1
        weights_H = 180/np.pi*10
        weights = torch.tensor(np.append(np.ones(21)*weights_P,np.ones(12)*weights_H), dtype=torch.float32)
    # Define the learning rate
    learning_rate = 0.003
    # Define the number of epochs
    num_epochs = 50000
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # get save folder path
    formatted_string = datetime.datetime.now().strftime("%Y%m%d%H%M")
    formatted_string = formatted_string[2:]
    if Variational:
        folder_path = 'PH_NN_results/trainLATENT_VAE_'
    else:
        folder_path = 'PH_NN_results/trainLATENT_AE_'
    folder_path += robot_type+'_'
    folder_path += 'latent'+str(latent_size)+'_'
    if weighted:
        folder_path += 'weighted_'
    folder_path += formatted_string+'/'

    # save a training parameters meta yaml file to folder_path
    meta_data = {'data_size': data_size, 'latent_size': latent_size, 'hidden_sizes': hidden_sizes, 'learning_rate': learning_rate, 'num_epochs': num_epochs}
    meta_data['mu'] = mu
    meta_data['sigma'] = sigma
    meta_data['loss_kl_weight'] = loss_kl_weight
    meta_data['Variational'] = Variational
    meta_data['robot_type'] = robot_type
    meta_data['weighted'] = weighted
    if weighted:
        meta_data['weights_P'] = weights_P
        meta_data['weights_H'] = weights_H
    if not os.path.exists(folder_path):
        pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
    with open(folder_path+'meta_data.yaml', 'w') as file:
        documents = yaml.dump(meta_data, file)

    # Training loop
    loss_all = []
    training_mean_error_all = []
    testing_mean_error_all = []
    training_max_error_all = []
    testing_max_error_all = []
    training_std_error_all = []
    testing_std_error_all = []
    data_sample_epoches = []
    best_loss = 1e10
    best_training_error = 1e10
    best_testing_error = 1e10

    training_start_time = time.time()
    training_t_epoch = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Forward pass
        model.train()
        outputs = model(data_delta_PH)
        if weighted:
            loss = loss_mse_fn(outputs, data_delta_PH, weights)
        else:
            loss = loss_mse_fn(outputs, data_delta_PH)
        if Variational:
            loss = loss*(1-loss_kl_weight) + model.encoder.kl*loss_kl_weight

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get testing data loss
        # test_outputs = model(test_inputs_q2q3)
        # test_loss = loss_fn(test_outputs, test_targets_delta_PH)

        # Print the loss for every 10 epochs
        print_loss = False
        print_error = False
        if epoch==0:
            print_loss = True
            print_error = True
        elif epoch<1001:
            if (epoch+1) % 10 == 0:
                print_loss = True
            if (epoch+1) % 100 == 0:
                print_error = True
        else:
            if (epoch+1) % 100 == 0:
                print_loss = True
            if (epoch+1) % 100 == 0:
                print_error = True

        model.eval()
        if best_loss > loss.item():
            best_loss = loss.item()
            torch.save(model.state_dict(), folder_path+'best_lost_model.pt')
        loss_all.append(loss.item())
        np.save(folder_path+'loss_all.npy',np.array(loss_all)) # save the loss
        if print_loss:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        if print_error:
            latent_vec = model.encoder(data_delta_PH)
            latent_vec_cpu = latent_vec.detach().numpy()
            interp_funcs = []
            for latent_i in range(latent_size):
                interp_funcs.append(LinearNDInterpNearestExtrap(inputs_q2q3, latent_vec_cpu[:,latent_i]))
            training_T_error,training_ori_error = test_fwd_accuracy(model, interp_funcs, inputs_q2q3, training_q, training_T,robot,param_nominal)
            testing_T_error,testing_ori_error = test_fwd_accuracy(model, interp_funcs, inputs_q2q3, testing_q, testing_T,robot,param_nominal)
            # print training and testing error, mean, max
            print(f'Training error: mean={np.mean(training_T_error):.4f}, max={np.max(training_T_error):.4f}')
            print(f'Testing error: mean={np.mean(testing_T_error):.4f}, max={np.max(testing_T_error):.4f}')
            training_mean_error_all.append(np.mean(training_T_error))
            testing_mean_error_all.append(np.mean(testing_T_error))
            training_max_error_all.append(np.max(training_T_error))
            testing_max_error_all.append(np.max(testing_T_error))
            training_std_error_all.append(np.std(training_T_error))
            testing_std_error_all.append(np.std(testing_T_error))
            data_sample_epoches.append(epoch)
            # save the model
            if best_training_error > np.max(training_T_error):
                best_training_error = np.max(training_T_error)
                torch.save(model.state_dict(), folder_path+'best_training_model.pt')
            if best_testing_error > np.max(testing_T_error):
                best_testing_error = np.max(testing_T_error)
                mean_testing_error = np.mean(testing_T_error)
                std_testing_error = np.std(testing_T_error)
                best_testing_ori_error = np.max(testing_ori_error)
                mean_testing_ori_error = np.mean(testing_ori_error)
                std_testing_ori_error = np.std(testing_ori_error)
                torch.save(model.state_dict(), folder_path+'best_testing_model.pt')
            np.save(folder_path+'training_mean_error_all.npy',np.array(training_mean_error_all))
            np.save(folder_path+'testing_mean_error_all.npy',np.array(testing_mean_error_all))
            np.save(folder_path+'training_max_error_all.npy',np.array(training_max_error_all))
            np.save(folder_path+'testing_max_error_all.npy',np.array(testing_max_error_all))
            np.save(folder_path+'training_std_error_all.npy',np.array(training_std_error_all))
            np.save(folder_path+'testing_std_error_all.npy',np.array(testing_std_error_all))
            np.save(folder_path+'data_sample_epoches.npy',np.array(data_sample_epoches))
            print("Current best:")
            print(f'mean testing error: {mean_testing_error:.2f}')
            print(f'std testing error: {std_testing_error:.2f}')
            print(f'max testing error: {best_testing_error:.2f}')
            print(f'mean testing ori error: {mean_testing_ori_error:.2f}')
            print(f'std testing ori error: {std_testing_ori_error:.2f}')
            print(f'max testing ori error: {best_testing_ori_error:.2f}')
            print("=========================")

        # training time for each epoch
        epoch_end_time = time.time()
        training_t_epoch.append(epoch_end_time-epoch_start_time)
        # print(f'Mean epoch time: {np.mean(training_t_epoch):.5f}, Total time: {epoch_end_time-training_start_time:.2f}')

    print('Training time:',time.time()-training_start_time)            

Rx=np.array([1,0,0])
Ry=np.array([0,1,0])
Rz=np.array([0,0,1])


config_dir='../config/'
\

# robot_type = 'R1'
####
robot_type = 'R2'

if robot_type == 'R1':
    ph_dataset_date='0801'
    test_dataset_date='0801'
    robot_marker_dir=config_dir+'MA2010_marker_config/'
    tool_marker_dir=config_dir+'weldgun_marker_config/'
    robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',\
                        tool_file_path=config_dir+'torch.csv',d=15,\
                        #  tool_file_path='',d=0,\
                        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
                        base_marker_config_file=robot_marker_dir+'MA2010_'+ph_dataset_date+'_marker_config.yaml',\
                        tool_marker_config_file=tool_marker_dir+'weldgun_'+ph_dataset_date+'_marker_config.yaml')
    nom_P=np.array([[0,0,0],[150,0,0],[0,0,760],\
                   [1082,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
    nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                   [-1,0,0],[0,-1,0],[-1,0,0]]).T
elif robot_type == 'R2':
    ph_dataset_date='0804'
    test_dataset_date='0804'
    robot_marker_dir=config_dir+'MA1440_marker_config/'
    tool_marker_dir=config_dir+'mti_marker_config/'
    robot=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',\
                        tool_file_path=config_dir+'mti.csv',\
                        pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
                        base_marker_config_file=robot_marker_dir+'MA1440_'+ph_dataset_date+'_marker_config.yaml',\
                        tool_marker_config_file=tool_marker_dir+'mti_'+ph_dataset_date+'_marker_config.yaml')
    nom_P=np.array([[0,0,0],[155,0,0],[0,0,614],\
                   [640,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
    nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                   [-1,0,0],[0,-1,0],[-1,0,0]]).T

# T_base_basemarker = robot.T_base_basemarker
# T_basemarker_base = T_base_basemarker.inv()
robot.P_nominal=deepcopy(robot.robot.P)
robot.H_nominal=deepcopy(robot.robot.H)
robot.P_nominal=robot.P_nominal.T
robot.H_nominal=robot.H_nominal.T
robot = get_H_param_axis(robot) # get the axis to parametrize H
param_nominal = np.array(np.reshape(robot.robot.P.T,-1).tolist()+[0]*12)

#### using rigid body
use_toolmaker=True
T_base_basemarker = robot.T_base_basemarker
T_basemarker_base = T_base_basemarker.inv()

if use_toolmaker:
    robot.robot.R_tool = robot.T_toolmarker_flange.R
    robot.robot.p_tool = robot.T_toolmarker_flange.p
    robot.T_tool_toolmarker = Transform(np.eye(3),[0,0,0])
    
    # robot.robot.R_tool = np.eye(3)
    # robot.robot.p_tool = np.zeros(3)
    # robot.T_tool_toolmarker = robot.T_toolmarker_flange.inv()

PH_data_dir='PH_grad_data/test'+ph_dataset_date+'_'+robot_type+'/train_data_'
# test_data_dir='kinematic_raw_data/test'+test_dataset_date+'_aftercalib/'
test_data_dir='kinematic_raw_data/test'+test_dataset_date+'_'+robot_type+'/'

print(PH_data_dir)
print(test_data_dir)

use_raw=False
test_robot_q = np.loadtxt(test_data_dir+'robot_q_align.csv',delimiter=',')
test_mocap_T = np.loadtxt(test_data_dir+'mocap_T_align.csv',delimiter=',')

train_robot_q = np.loadtxt(PH_data_dir+'robot_q_align.csv',delimiter=',')
train_mocap_T = np.loadtxt(PH_data_dir+'mocap_T_align.csv',delimiter=',')

# split_index = len(train_robot_q)
# test_robot_q = np.vstack((train_robot_q,test_robot_q))
# test_mocap_T = np.vstack((train_mocap_T,test_mocap_T))

calib_file_name = 'calib_PH_q_ana.pickle'
with open(PH_data_dir+calib_file_name,'rb') as file:
    PH_q=pickle.load(file)

# ph_param_fbf=PH_Param(nom_P,nom_H)
# ph_param_fbf.fit(PH_q,method='FBF')

# get theta phi
train_q=[]
param_PH_q = []
for qkey in PH_q.keys():
    # NN data input: q2 q3
    train_q.append(np.array(qkey))
    # NN output: P H
    this_H = PH_q[qkey]['H']
    param_H = []
    for i,h in enumerate(this_H.T):
        theta_sol = subproblem2(nom_H[:,i], h, robot.param_k2[i], robot.param_k1[i])
        theta_sol = theta_sol[0] if theta_sol[0][0]<np.pi/2 and theta_sol[0][0]>-np.pi/2 else theta_sol[1]
        param_H.extend(theta_sol[::-1])
    param_PH = np.array(np.reshape(PH_q[qkey]['P'].T,-1).tolist()+param_H)
    param_PH_q.append(param_PH-param_nominal) # relative to nominal, predict the difference

train_q = np.array(train_q)
param_PH_q = np.array(param_PH_q)

# draw the data q (6 of them) vs index 
# in a subplot
# fig, axs = plt.subplots(2, 3)
# for i in range(6):
#     axs[i//3,i%3].plot(np.degrees(train_robot_q[:,i]))
#     axs[i//3,i%3].plot(np.degrees(test_robot_q[:,i]))
#     axs[i//3,i%3].set_title('q'+str(i+1))
# plt.show()

## NN input: training q, 2x1
## NN output: training param_PH, 33x1
## train the NN
# train(np.array(train_q),np.array(param_PH_q),train_robot_q,train_mocap_T,test_robot_q,test_mocap_T,robot,param_nominal,robot_type)
# train_interp(np.array(train_q),np.array(param_PH_q),train_robot_q,train_mocap_T,test_robot_q,test_mocap_T,robot,param_nominal,robot_type)
# latent_space_analysis(np.array(train_q),np.array(param_PH_q),train_robot_q,train_mocap_T,test_robot_q,test_mocap_T,robot,param_nominal,robot_type)
trained_model_test(np.array(train_q),np.array(param_PH_q),train_robot_q,train_mocap_T,test_robot_q,test_mocap_T,robot,param_nominal,robot_type)
