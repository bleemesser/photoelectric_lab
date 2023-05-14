import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
from tqdm import tqdm
import hashlib
import pickle
import os


def s_statistic(x, average_kes, frequencies, stdevs, offset=0):
    return (
        np.sum((average_kes - (x[0] * frequencies + x[1])) ** 2 / stdevs**2) - offset
    )


def find_zero_in_direction(
    x, direction, average_kes, frequencies, stdevs, target_value=0, max_iters=20000
):
    # we'll offset the s-statistic by the target value so that we can find the closest zero to the target value in the given direction
    # we'll use gradient descent to find the closest zero in the given direction

    # start at the given point
    p = x
    # since the values of x[0] is in the order of 10^-34, we'll use a step size of 10^-35
    step_size_x = 1e-34
    step_size_y = 1e-19
    iters = 0
    # iterate until we find a point where the s-statistic is within 1e-12 of the target value
    s = s_statistic(p, average_kes, frequencies, stdevs, target_value)

    while (
        abs(s) > 1e-12
        and iters < max_iters
    ):
        # print("Step size:", step_size, "Point:", p, "S-Statistic:", s_statistic(p, average_kes, frequencies, stdevs, target_value))
        # take a step in the given direction
        p[0] += step_size_x * direction[0]
        p[1] += step_size_y * direction[1]
        s = s_statistic(p, average_kes, frequencies, stdevs, target_value)
        # if we've gone too far, take a step in the opposite direction and reduce the step size
        # print(f"S={s}, target={1e-12}, step_size_x={step_size_x}, step_size_y={step_size_y}, p={p}")
        if s > 0:
            p[0] -= step_size_x * direction[0]
            p[1] -= step_size_y * direction[1]
            step_size_x /= 2
            step_size_y /= 2
        else:
            # increase the step size if we're going in the right direction but are still too far away
            step_size_x *= 1.1
            step_size_y *= 1.1
        iters += 1
    if iters == max_iters:
        return p, False
    #     return [None, None]
    return p, True


def main():
    # READ DATA
    # check if data.csv exists
    if not os.path.exists("data.csv"):
        print("data.csv not found")
        return
    print("Reading data.csv...")
    data = pd.read_csv("data.csv")
    datahash = hashlib.md5(data.to_string().encode("utf-8")).hexdigest()
    # check if Wavelength column exists
    if "Wavelength" not in data.columns:
        print("Wavelength column not found")
        return
    wavelengths = data["Wavelength"]

    # convert wavelengths (nm)) to frequencies (Hz)
    frequencies = 299792458 / (wavelengths * 10**-9)

    # check if any voltage columns exist, simultaneously remove all non-voltage columns
    data = data.filter(like="Voltage")
    if len(data.columns) == 0:
        print("No voltage columns found")
        return

    # convert all voltages (V) to energies (J) by multiplying by Qe
    data = data * -1.602176634e-19
    data.columns = data.columns.str.replace("Voltage", "KE")

    # get average KE across all trials
    average_kes = data.filter(like="KE").mean(axis=1)

    # get standard deviation of KE across all trials
    std = data.filter(like="KE").std(axis=1)

    # create new dataframe with frequency, average KE, and standard deviation
    df = pd.DataFrame({"Frequency": frequencies, "KE": average_kes, "Stdev": std})

    # perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df["Frequency"], df["KE"]
    )
    print(
        "\nLinear Regression: y=",
        slope,
        "x+",
        intercept,
        ", r=",
        r_value,
        ", p=",
        p_value,
        ", std_err=",
        std_err,
    )

    # perform s-statistic optimization
    actual_planck = (
        6.62607015 * 10**-34
    )  # actual value for planck's constant, we'll use this as a guess
    # we'll use the work function found by linear regression as a guess for the intercept
    res = minimize(
        s_statistic,
        [actual_planck, intercept],
        args=(average_kes, frequencies, std),
        method="Nelder-Mead",
        tol=1e-12,
    )
    slope_s = res.x[0]  # our found value for planck's constant
    intercept_s = res.x[1]  # our found value for the work function
    min_s = res.fun  # the minimum value of the s-statistic
    print("\nS-Statistic: y=", slope_s, "x+", intercept_s, ", s=", min_s)

    # find the error in the slope and intercept by finding the points where the s-statistic is 1 above the minimum
    # this will be the 0.68 confidence interval
    # to do this, we'll use the bisection method to find the zeroes of the s-statistic minus s_min - 1
    print("\nFinding 0.68 confidence interval...")
    # we need to rotate around the minimum point in steps of 2pi/num_points
    NUM_POINTS = 2000
    MAX_ITERS = 1000
    points = []
    failed_points = []
    USE_CACHE = True
    if not os.path.exists("cache"):
        os.mkdir("cache")
    if os.path.exists(f"cache/{datahash}{NUM_POINTS}{MAX_ITERS}.pkl") and USE_CACHE == True:
        print("Cached points found matching data values, skipping calculation")
        with open(f"cache/{datahash}{NUM_POINTS}{MAX_ITERS}.pkl", "rb") as f:
            [points, failed_points] = pickle.load(f)
    else:
        print("No cached points found matching data values, calculating...")
        # create progress bar
        pbar = tqdm(total=NUM_POINTS)
        for i in range(NUM_POINTS):
            # we want to find the closest zero to the point (slope_s, intercept_s), so we'll use that as a guess and then bisect
            # start at the minimum point and do gradient descent to find the closest zero in each direction
            direction = [
                np.cos(2 * np.pi * i / NUM_POINTS),
                np.sin(2 * np.pi * i / NUM_POINTS),
            ]
            if (
                abs(direction[0]) < 1e-10
            ):  # we had some weird issues with the direction being exactly 0, so we'll just set it to 0 if it's close enough
                direction[0] = 0
            if abs(direction[1]) < 1e-10:
                direction[1] = 0
            p, success = find_zero_in_direction(
                [slope_s, intercept_s],
                direction,
                average_kes,
                frequencies,
                std,
                min_s + 1,
                MAX_ITERS,
            )
            pbar.update(1)
            if success:
                points.append(p)
            else:
                failed_points.append(p)
                # print("\nNumber:", i, "Point found:", p)

        # convert points to numpy array
        points = np.array(points)
        failed_points = np.array(failed_points)
        pickle.dump([points, failed_points], open(f"cache/{datahash}{NUM_POINTS}{MAX_ITERS}.pkl", "wb"))
        print("Saved points to cache")

    # find the largest distance between the minimum point and the points on the 0.68 confidence interval
    max_dist = 0
    for point in points:
        dist = np.sqrt((point[0] - slope_s) ** 2 + (point[1] - intercept_s) ** 2)
        if dist > max_dist:
            max_dist = dist
    print("\nMax distance from minimum point:", max_dist)

    # PLOT DATA
    # plot 2d graph of frequency vs average KE
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.errorbar(
        frequencies,
        average_kes,
        yerr=std,
        fmt="o",
        markersize=2,
        capsize=2,
        label="Data",
    )
    plt.plot(
        frequencies,
        slope * frequencies + intercept,
        label=f"Linear Regression: y={slope:.2e}x+{intercept:.2e}",
        color="grey",
    )
    plt.plot(
        frequencies,
        slope_s * frequencies + intercept_s,
        label=f"S-Statistic: y={slope_s:.2e}x+{intercept_s:.2e}",
        color="darkblue",
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Average Kinetic Energy (J)")
    plt.title("Frequency vs Average Kinetic Energy")
    plt.legend()

    # plot 3d graph of s-statistic vs planck's constant vs work function
    ax = plt.subplot(1, 2, 2, projection="3d")
    # create a meshgrid of planck's constant and work function values
    # x is planck's constant, z is work function
    xdelta = 0.006 * slope_s
    ydelta = 0.006 * intercept_s
    x = np.linspace(slope_s - xdelta, slope_s + xdelta, 100)
    z = np.linspace(intercept_s - ydelta, intercept_s + ydelta, 100)
    x, z = np.meshgrid(x, z)
    y = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            y[i][j] = s_statistic([x[i][j], z[i][j]], average_kes, frequencies, std)
    # plot the surface
    ax.plot_surface(x, z, y, cmap="winter", alpha=0.35)
    # plot the minimum point
    ax.scatter(slope_s, intercept_s, min_s, c="black", label="Minimum Point", s=10)
    # plot the 0.68 confidence interval
    ax.scatter(
        points[:, 0],
        points[:, 1],
        min_s + 1,
        c="r",
        label="0.68 Confidence Interval",
        s=1,
    )
    # print(failed_points)
    ax.scatter(failed_points[:, 0], failed_points[:, 1], [s_statistic(p, average_kes, frequencies, std) for p in failed_points], c="grey", s=1)
    ax.set_xlabel("Slope (Planck's constant) (m^2 kg / s) unit: 10^-34")
    ax.set_ylabel("Intercept (Work Function) (J) unit: 10^-19")
    ax.set_zlabel("S-Statistic")
    ax.set_title("S-Statistic vs Slope and Intercept")
    ax.legend()

    plt.savefig("KE_vs_Freq.png", dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.show()


if __name__ == "__main__":
    main()
