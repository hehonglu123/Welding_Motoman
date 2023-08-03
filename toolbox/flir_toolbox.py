import numpy as np
##############################################FLIR COUNTS TO TEMPERATURE CONVERSION##############################################
####FROM https://flir.custhelp.com/app/answers/detail/a_id/3321/~/the-measurement-formula


def counts2temp(data_counts,R,B,F,J1,J0,Emiss):
    # reflected energy
    TRefl = 18

    # atmospheric attenuation
    TAtmC = 20
    TAtm = TAtmC + 273.15
    Tau = 0.99 #transmission

    # external optics
    TExtOptics = 20
    TransmissionExtOptics = 1
  
    K1 = 1 / (Tau * Emiss * TransmissionExtOptics)
        
    # Pseudo radiance of the reflected environment
    r1 = ((1-Emiss)/Emiss) * (R/(np.exp(B/TRefl)-F))
    # Pseudo radiance of the atmosphere
    r2 = ((1 - Tau)/(Emiss * Tau)) * (R/(np.exp(B/TAtm)-F)) 
    # Pseudo radiance of the external optics
    r3 = ((1-TransmissionExtOptics) / (Emiss * Tau * TransmissionExtOptics)) * (R/(np.exp(B/TExtOptics)-F))
            
    K2 = r1 + r2 + r3
    
    data_obj_signal = (data_counts - J0)/J1
    data_temp = (B / np.log(R/((K1 * data_obj_signal) - K2) + F)) -273.15
    
    return data_temp


def counts2temp_4learning(data_counts,R,B,F,J1,J0):
    Emiss=data_counts[-1]
    data_counts=data_counts[:-1]
    # reflected energy
    TRefl = 18

    # atmospheric attenuation
    TAtmC = 20
    TAtm = TAtmC + 273.15
    Tau = 0.99 #transmission

    # external optics
    TExtOptics = 20
    TransmissionExtOptics = 1
  
    K1 = 1 / (Tau * Emiss * TransmissionExtOptics)
        
    # Pseudo radiance of the reflected environment
    r1 = ((1-Emiss)/Emiss) * (R/(np.exp(B/TRefl)-F))
    # Pseudo radiance of the atmosphere
    r2 = ((1 - Tau)/(Emiss * Tau)) * (R/(np.exp(B/TAtm)-F)) 
    # Pseudo radiance of the external optics
    r3 = ((1-TransmissionExtOptics) / (Emiss * Tau * TransmissionExtOptics)) * (R/(np.exp(B/TExtOptics)-F))
            
    K2 = r1 + r2 + r3
    
    data_obj_signal = (data_counts - J0)/J1
    data_temp = (B / np.log(R/((K1 * data_obj_signal) - K2) + F)) -273.15
    
    return data_temp