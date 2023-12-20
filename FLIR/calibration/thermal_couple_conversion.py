# Extracted ITS-90 coefficients for a K-Type thermocouple from the image provided by the user.
# These coefficients are for three different temperature ranges.
import numpy as np

def voltage_to_temperature(voltage):
    """ voltage in microvolt
    Convert voltage to temperature for a K-type thermocouple using ITS-90 coefficients.
    The function selects the appropriate set of coefficients based on the voltage.
    """
    coefficients = {
        'subzero': [  # -200°C to 0°C
            0.0000000e+00,
            2.5173462e-02,
            -1.1662878e-06,
            -1.0833638e-09,
            -8.9773540e-13,
            -3.7342377e-16,
            -8.6632643e-20,
            -1.0450598e-23,
            -5.1920577e-29
        ],
        'midrange': [  # 0°C to 500°C
            0.0000000e+00,
            2.508355e-02,
            7.860106e-08,
            -2.503131e-10,
            8.315270e-14,
            -1.228034e-17,
            9.804036e-22,
            -4.413030e-26,
            1.057734e-30,
            -1.052755e-35
        ],
        'highrange': [  # 500°C to 1372°C
            -1.318058e+02,
            4.830222e-02,
            -1.646031e-06,
            5.464731e-11,
            -9.650715e-16,
            8.802193e-21,
            -3.110810e-26
        ]
    }
    if voltage < 0:
        # Subzero range
        coeffs = coefficients['subzero']
        temp_range = 'subzero'
    elif voltage < 20644:
        # Midrange
        coeffs = coefficients['midrange']
        temp_range = 'midrange'
    else:
        # High range
        coeffs = coefficients['highrange']
        temp_range = 'highrange'


    temperature = 0.0
    # Calculate temperature using the polynomial equation for the given voltage range
    for i, coeff in enumerate(coeffs):
        # print(i,coeff)
        temperature += coeff * (voltage ** i)
    
    # temperature+=magnetic_coeff[0]*np.exp(magnetic_coeff[1]*(voltage-magnetic_coeff[2])**2)
    return temperature

# # Example voltage reading
# voltage_example = 40000.0  # Example in microvolt
# temperature = voltage_to_temperature(voltage_example)
# print(temperature)
