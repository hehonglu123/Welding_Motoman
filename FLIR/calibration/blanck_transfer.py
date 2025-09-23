import numpy as np

polyfit_coefficients_temp = np.load('../../FLIR/calibration/ER4043_IR_calibration.npy')  # Example coefficients

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

def calibration_with_ml(raw):
    """
    Convert raw FLIR data to temperature using a polynomial fit.
    raw: Raw thermal reading from FLIR (clicks)
    """
    return np.poly1d(polyfit_coefficients_temp)(raw)


if __name__ == "__main__":
    # Example usage
    # print(celcius_to_clicks(25))  # Convert 25°C to clicks
    # print(raw_to_temperature_with_emissivity(10000, emissivity=0.5))  # Example conversion with emissivity
    # # Load coefficients from a file and use them for calibration
    
    # print(calibration_with_ml(np.arange(8800,20000,1000)))  # Example conversion using ML calibration

    # tick_test = np.arange(8800, 17000, 10)
    # tick_test = np.array([16109.148, 16109.148 ,15692.236, 15692.238, 14837.291, 15405.782, 15405.782, 15995.254, 14923.462, 16032.09 ])
    # draw_relation = calibration_with_ml(tick_test)
    # print(tick_test[np.argmax(draw_relation)], draw_relation[np.argmax(draw_relation)])
    # from matplotlib import pyplot as plt
    # plt.plot(tick_test, draw_relation)
    # plt.xlabel('Raw Signal (Clicks)')
    # plt.ylabel('Temperature (°C)')
    # plt.title('Calibration Curve')
    # plt.grid()
    # plt.show()

    raw_counts = np.array([8000, 10000, 14000, 18000, 22000, 26000])
    T = raw_to_temperature_with_emissivity(
        raw_counts,
        R=21106.77, B=1501.0, F=1.0, O=0.0,
        emissivity=0.75,       # oxidized stainless surface
        T_reflected=298.15     # ~25 °C lab ambient
    )
    print("Raw Counts:", raw_counts)
    print("Temperatures (°C):", T)