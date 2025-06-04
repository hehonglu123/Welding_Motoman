import numpy as np

def celcius_to_clicks(temperature, R=21106.77, B=1501.0, F=1.0, O=0.0):
    """
    Convert temperature in Celsius to FLIR raw signal (clicks).
    temperature: Temperature in Celsius
    R: Calibration constant for FLIR camera
    B: Calibration constant for FLIR camera
    F: Calibration constant for FLIR camera
    O: Calibration constant for FLIR camera
    """
    T_kelvin = temperature + 273.15  # Convert °C to K
    clicks = R / (np.exp(B / T_kelvin) - F) + O
    return clicks

def raw_to_temperature_with_emissivity(raw, R=21106.77, B=1501.0, F=1.0, O=0.0,
                                       emissivity=0.05, T_reflected=298.15):
    """
    Convert FLIR raw signal to true object temperature with emissivity correction.
    raw: Raw thermal reading from FLIR (clicks)
    emissivity: Estimated surface emissivity (0.04–0.1 for bare Al)
    T_reflected: Reflected ambient temperature in Kelvin (e.g., 25°C = 298.15 K)
    """
    # Apparent temperature from raw data (assuming emissivity = 1.0)
    T_apparent = B / np.log(R / (raw - O) + F)

    # Compensate for low emissivity
    numerator = (T_apparent**4 - (1 - emissivity) * T_reflected**4)
    T_corrected = (numerator / emissivity) ** 0.25

    return T_corrected - 273.15  # Convert K → °C


print(celcius_to_clicks(25))  # Example conversion from 25°C to clicks
print(raw_to_temperature_with_emissivity(10000, emissivity=0.5))