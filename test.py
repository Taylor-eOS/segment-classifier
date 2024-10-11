import numpy as np

def test():
    # Create a small matrix manually (6 frames, 3 MFCC features)
    mfcc = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18]
    ])  # Shape: (6, 3)

    # N_MFCC is 3 in this example
    N_MFCC = 3
    print(mfcc)
    print("")

    # Reshape the matrix to group every 3 frames
    mfcc_reshaped = mfcc.reshape(-1, 3, N_MFCC)  # Shape: (2, 3, 3)
    print(mfcc_reshaped)
    print("")

    # Average along the second axis (groups of 3 frames)
    mfcc_avg = mfcc_reshaped.mean(axis=1)        # Shape: (2, 3)

    return mfcc_avg

# Run the test
print(test())

