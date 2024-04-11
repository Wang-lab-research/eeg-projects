import numpy as np


# Create surrogate data
def create_surrogate_data(original_data):
    """
    Generate surrogate data by randomizing phases of the original data's FFT.
    :param original_data: The original data array.
    :return: Surrogate data with randomized phases.
    """
    # Step 1: Compute the FFT of the original data
    fft_result = np.fft.fft(original_data)

    # Step 2: Randomize the phases
    # Extract amplitudes
    amplitudes = np.abs(fft_result)

    # Generate random phases
    random_phases = np.exp(2j * np.pi * np.random.random(size=original_data.shape))

    # Combine amplitudes with random phases
    modified_fft_result = amplitudes * random_phases

    # Step 3: Perform the Inverse FFT
    surrogate_data = np.fft.ifft(modified_fft_result)

    # Return the real part of the surrogate data
    return np.real(surrogate_data)
