import numpy as np

ZERO_THRESHOLD = 1e-6


def get_all_measurements():
    """
    Gets all II, HI, IH, VI, IV measurements for tomography.

    In the future, this function can be extended to call IBM Qiskit code to
    actually fetch simulator data.

    Returns:
        tuple[NDArray, NDArray, NDArray, NDArray, NDArray]: II, HI, IH, VI, IV measurements, in that order
    """
    II = np.array([0, 0, 0, 0])
    IH = np.array([0, 0, 0, 0])
    IH = np.array([0, 0, 0, 0])
    VI = np.array([0, 0, 0, 0])
    IV = np.array([0, 0, 0, 0])
    return II, IH, IH, VI, IV


def get_II():
    return np.array([0, 0, 0, 0])


def get_IH():
    return np.array([0, 0, 0, 0])


def get_HI():
    return np.array([0, 0, 0, 0])


def get_IV():
    return np.array([0, 0, 0, 0])


def get_VI():
    return np.array([0, 0, 0, 0])


def get_ICH():
    return np.array([0, 0, 0, 0])


def get_ICV():
    return np.array([0, 0, 0, 0])


def get_CHI():
    return np.array([0, 0, 0, 0])


def get_CVI():
    return np.array([0, 0, 0, 0])


def general_tomography():
    """
    Performs general tomography for two qubits.

    Returns:
        NDArray[float64]: Reconstructed two qubit state
    """
    ii = get_II()

    # clamp low measurements to 0
    ii[ii < ZERO_THRESHOLD] = 0

    ii = get_II()

    # clamp low measurements to 0
    ii[ii < ZERO_THRESHOLD] = 0

    if np.count_nonzero(ii) > 2:
        raise Exception("Incorrect number of nonzero measurements")

    # reconstructed values
    r_v = np.zeros((4, 2))

    if np.count_nonzero(ii) == 1:
        loc = 0
        while ii[loc] == 0:
            loc += 1
        r_v[loc][0] = 1

    elif np.count_nonzero(ii) == 2:
        nz_indices = np.nonzero(ii)

        r = (1 << np.arange(2))[:, None]
        if np.count_nonzero((np.bitwise_xor(nz_indices[0], nz_indices[1]) & r) != 0) == 1:  # if in tensor form
            if ii[0] == 0:
                if ii[1] == 0:  # ii^T = [ 0 0 x x ]
                    hi = get_HI()
                    vi = get_VI()

                    # calculate x_3
                    r_v[2][0] = np.sqrt(ii[2])

                    # calculate x_4
                    r_v[3][0] = (hi[2] - hi[3]) / (2 * r_v[2][0])
                    r_v[3][1] = (vi[3] - vi[2]) / (2 * r_v[2][0])
                else:  # ii^T = [ 0 x 0 x ]
                    ih = get_IH()
                    iv = get_IV()

                    # calculate x_2
                    r_v[1][0] = np.sqrt(ii[1])

                    # calculate x_4
                    r_v[3][0] = (ih[1] - ih[3]) / (2 * r_v[1][0])
                    r_v[3][1] = (iv[3] - iv[1]) / (2 * r_v[1][0])
            else:
                if ii[1] == 0:  # ii^T = [ x 0 x 0 ]
                    ih = get_IH()
                    iv = get_IV()

                    # calculate x_1
                    r_v[0][0] = np.sqrt(ii[0])

                    # calculate x_3
                    r_v[2][0] = (ih[0] - ih[2]) / (2 * r_v[0][0])
                    r_v[2][1] = (iv[2] - iv[0]) / (2 * r_v[0][0])
                else:  # ii^T = [ x x 0 0 ]
                    hi = get_HI()
                    vi = get_VI()

                    # calculate x_1
                    r_v[0][0] = np.sqrt(ii[0])

                    # calculate x_2
                    r_v[1][0] = (hi[0] - hi[1]) / (2 * r_v[0][0])
                    r_v[1][1] = (vi[1] - vi[0]) / (2 * r_v[0][0])
        else:  # not in tensor form
            cvi = get_CVI()
            chi = get_CHI()
            if ii[0] == 0:  # ii^T = [ 0 x x 0 ]
                # calculate x_2
                r_v[1][0] = np.sqrt(ii[1])

                # calculate x_4
                r_v[3][0] = (chi[1] - chi[3]) / (2 * r_v[1][0])
                r_v[3][1] = (cvi[3] - cvi[1]) / (2 * r_v[1][0])
            else:  # ii^T = [ x 0 0 x ]
                # calculate x_1
                r_v[0][0] = np.sqrt(ii[0])

                # calculate x_3
                r_v[2][0] = (chi[0] - chi[2]) / (2 * r_v[0][0])
                r_v[2][1] = (cvi[2] - cvi[0]) / (2 * r_v[0][0])

    else:
        ii, hi, ih, vi, iv = get_all_measurements()
        # special case (if x_1 == 0)
        if ii[0] == 0:
            # calculate x_2
            r_v[1][0] = np.sqrt(ii[1])

            # calculate x_4
            r_v[3][0] = (hi[1] - hi[3]) / (2 * r_v[1][0])
            r_v[3][1] = (vi[3] - vi[1]) / (2 * r_v[1][0])

            # calculate x_3
            r_v[2][0] = (-1 * r_v[3][0] * (ih[2] - ih[3])) / (2 * (r_v[3][0] * r_v[3][0] - r_v[3][1] * r_v[3][1])) + (
                        r_v[3][1] * (iv[3] - iv[2])) / (2 * (r_v[3][0] * r_v[3][0] - r_v[3][1] * r_v[3][1]))
            r_v[2][1] = (r_v[3][1] * (ih[2] - ih[3])) / (2 * (r_v[3][0] * r_v[3][0] - r_v[3][1] * r_v[3][1])) + (
                        -1 * r_v[3][0] * (iv[3] - iv[2])) / (2 * (r_v[3][0] * r_v[3][0] - r_v[3][1] * r_v[3][1]))

        # special case (if x_3 == 0)
        elif ii[2] == 0:
            # calculate x_1
            r_v[0][0] = np.sqrt(ii[0])

            # calculate x_2
            r_v[1][0] = (ih[0] - ih[1]) / (2 * r_v[0][0])
            r_v[1][1] = (iv[1] - iv[0]) / (2 * r_v[0][0])

            # calculate x_4
            r_v[3][0] = (-1 * r_v[1][0] * (hi[1] - hi[3])) / (2 * (r_v[1][0] * r_v[1][0] - r_v[1][1] * r_v[1][1])) + (
                        r_v[1][1] * (vi[3] - vi[1])) / (2 * (r_v[1][0] * r_v[1][0] - r_v[1][1] * r_v[1][1]))
            r_v[3][1] = (r_v[1][1] * (hi[1] - hi[3])) / (2 * (r_v[1][0] * r_v[1][0] - r_v[1][1] * r_v[1][1])) + (
                        -1 * r_v[1][0] * (vi[3] - vi[1])) / (2 * (r_v[1][0] * r_v[1][0] - r_v[1][1] * r_v[1][1]))

        # general case (0 zero measurements, or x_2 == 0 or x_4 == 0)
        else:
            # calculate x_1
            r_v[0][0] = np.sqrt(ii[0])

            # calculate x_3
            r_v[2][0] = (hi[0] - hi[2]) / (2 * r_v[0][0])
            r_v[2][1] = (vi[2] - vi[0]) / (2 * r_v[0][0])

            # calculate x_2
            r_v[1][0] = (ih[0] - ih[1]) / (2 * r_v[0][0])
            r_v[1][1] = (iv[1] - iv[0]) / (2 * r_v[0][0])

            # calculate x_4
            r_v[3][0] = (-1 * r_v[2][0] * (ih[2] - ih[3])) / (2 * (r_v[2][0] * r_v[2][0] - r_v[2][1] * r_v[2][1])) + (
                        r_v[2][1] * (iv[3] - iv[2])) / (2 * (r_v[2][0] * r_v[2][0] - r_v[2][1] * r_v[2][1]))
            r_v[3][1] = (r_v[2][1] * (ih[2] - ih[3])) / (2 * (r_v[2][0] * r_v[2][0] - r_v[2][1] * r_v[2][1])) + (
                        -1 * r_v[2][0] * (iv[3] - iv[2])) / (2 * (r_v[2][0] * r_v[2][0] - r_v[2][1] * r_v[2][1]))
    return r_v
