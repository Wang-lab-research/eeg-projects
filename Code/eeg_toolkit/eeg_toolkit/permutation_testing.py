import numpy as np


# Create surrogate data
def create_surrogate_data(data_to_permute, sfreq, tmin, tmax):
    """
    Generate surrogate data by randomizing phases of the original data's FFT.
    :param data_to_permute: The original data array.
    :return: Surrogate data with randomized phases.
    """
    # Set random seed
    np.random.seed(42)

    # Step 1: Shift the data at least one half-cycle (based on theta),
    # at most 14 half-cycles (so as not to completely overlap with original data)
    theta_range = [*range(4, 9)]  # 4-8Hz
    theta_mid_freq = np.mean(theta_range)
    full_cycle = sfreq / theta_mid_freq
    half_cycle = full_cycle / 2

    # determine how many cycles to shift
    num_ts = (tmax - tmin) * sfreq
    num_cycles = np.floor(num_ts / half_cycle)

    rand_int = np.random.randint(1, num_cycles)
    shift = int(rand_int * half_cycle)

    data_to_permute = np.concatenate((data_to_permute[shift:], data_to_permute[:shift]))

    print(f"len(data_to_permute) = {len(data_to_permute)}")

    # Step 2: Compute the FFT of the original data
    fft_result = np.fft.fft(data_to_permute)

    # Step 3: Randomize the phases
    # Extract amplitudes
    amplitudes = np.abs(fft_result)

    # Generate random phases
    random_phases = np.exp(2j * np.pi * np.random.random(size=data_to_permute.shape))

    # Combine amplitudes with random phases
    modified_fft_result = amplitudes * random_phases

    # Step 4: Perform the Inverse FFT
    surrogate_data = np.fft.ifft(modified_fft_result)

    # Return the real part of the surrogate data
    return np.real(surrogate_data)
