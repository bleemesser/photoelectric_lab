import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize

# SPREADSHEET REQUIREMENTS:
# - Must be a csv file
# - Must have a column named 'Wavelength' with the wavelength of the light in nm
# - Must have columns named 'Voltage 1', 'Voltage 2', etc. with the voltages in volts. The number of columns does not matter, but there must be at least one. The values should be negative.

# READ DATA
data = pd.read_csv('csv/data.csv')
wavelengths = data['Wavelength']
frequencies = 299792458 / wavelengths
data = data.filter(like='Voltage')
# multiply all voltages by 1.602176634e-19 to get energy in joules, multiply by 10**-9 to compensate for wavelength in nm
data = data * -1.602176634e-19 *10**-9
# in column names, replace 'Voltage' with 'KE'
data.columns = data.columns.str.replace('Voltage', 'KE')
# get average KE across all trials
average_voltages = data.filter(like='KE').mean(axis=1)
# get standard deviation of voltage
std = data.filter(like='KE').std(axis=1)
# create new dataframe with frequency, average KE, and standard deviation
df = pd.DataFrame({'Frequency': frequencies, 'KE': average_voltages, 'Stdev': std})

# FIT MODELS
# Linear Regression:
slope, intercept, r_value, p_value, std_err = stats.linregress(df['Frequency'], df['KE'])

# S-Statistic:
def s_statistic(x):
    return np.sum((df['KE'] - (x[0] * df['Frequency'] + x[1])) ** 2 / df['Stdev'] ** 2)

planck = 6.67607015e-34 # guess for planck's constant
res = minimize(s_statistic, [planck, intercept], method='Nelder-Mead', tol=1e-12)
slope_s = res.x[0] # our found value for planck's constant
intercept_s = res.x[1] # our found value for the work function
min_s = res.fun # the minimum value of the s-statistic


# plot data
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.errorbar(df['Frequency']/1000, df['KE'], yerr=df['Stdev'], fmt='o', markersize=3.5, label='Data', capsize=4, elinewidth=1.5)
plt.plot(df['Frequency']/1000, slope * df['Frequency'] + intercept, label=f'Linear Regression: y={slope:.2e}x+{intercept:.2e}') # thus far seems basically identical to the s-statistic
plt.plot(df['Frequency']/1000, slope_s * df['Frequency'] + intercept_s, label=f'S-Statistic: y={slope_s:.2e}x+{intercept_s:.2e}, s={min_s:.2e}')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Kinetic Energy (J)')
plt.title('Kinetic Energy vs Frequency')
plt.legend()
ax = plt.subplot(1, 2, 2, projection='3d')
x = np.linspace(slope_s - 0.1 * slope_s, slope_s + 0.1 * slope_s, 100)
z = np.linspace(intercept_s - 0.1 * intercept_s, intercept_s + 0.1 * intercept_s, 100)
x, z = np.meshgrid(x, z)
y = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        y[i][j] = s_statistic([x[i][j], z[i][j]])
# create a new colormap that is the same as the default one, but with the first color removed
viridis = matplotlib.colormaps['viridis']
newcolors = viridis(np.linspace(0, 1, 256))
newcolors = newcolors[1:, :]
newcmp = ListedColormap(newcolors)
# plot the surface
ax.plot_surface(x, z, y, cmap=newcmp, alpha=0.75)
# add the minimum point as a dark black dot
ax.scatter(slope_s, intercept_s, min_s, c='black', s=10, label=f'Minimum: y={slope_s:.2e}x+{intercept_s:.2e}, s={min_s:.2e}', zorder=10)
ax.legend()
# add labels
ax.set_title('S-Statistic vs Slope and Intercept')
ax.set_xlabel('Slope (x10^-34))')
ax.set_ylabel('Intercept (x10^-28)')
ax.set_zlabel('S-Statistic')

# save the figure as a high quality png
plt.savefig('plots/KE_vs_Freq.png', dpi=300, bbox_inches='tight')
plt.show()




