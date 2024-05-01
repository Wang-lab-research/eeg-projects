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


# Create AAFT surrogate
def create_aaft_surrogate(original_data):  
    # Step 1: Generate white noise with the same variance as the original data  
    white_noise = np.random.normal(0, np.std(original_data), len(original_data))  
  
    # Step 2: Sort the white noise in the order of the sorted original data  
    surrogate_data = np.sort(white_noise)[np.argsort(np.argsort(original_data))]  
  
    # Step 3: Do a Fourier transform on the surrogate data  
    fft_surrogate = np.fft.fft(surrogate_data)  
  
    # Step 4: Generate phase randomized surrogate data  
    phase_randomized_surrogate = np.abs(fft_surrogate) * np.exp(1j * 2 * np.pi * np.random.random(len(original_data)))  
  
    # Step 5: Do an inverse Fourier transform on the phase randomized surrogate  
    inverse_fft = np.fft.ifft(phase_randomized_surrogate)  
  
    # Step 6: Sort the inverse FFT in the order of the sorted surrogate data from step 2  
    final_surrogate = np.sort(np.real(inverse_fft))[np.argsort(np.argsort(surrogate_data))]  
  
    return final_surrogate  
