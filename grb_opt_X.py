# import all of the packages i need

import numpy as np
import pandas as pd
from scipy import interpolate
from adjustText import adjust_text
import matplotlib.pyplot as plt, seaborn as sns, astropy.units as u
from matplotlib import ticker
from astropy.io import votable, ascii
from astropy.coordinates import SkyCoord
#from gdpyc.core import GasMap, DustMap
from scipy import interpolate, stats, optimize
#from src.xrt import XRT_lightcurve, get_photonIndex, get_temporalIndex, get_columnDensity
from scipy import interpolate, integrate
from bs4 import BeautifulSoup as bs
from astropy.coordinates import SkyCoord
from astropy.table import Table



#from src.xrt import XRT_lightcurve, get_photonIndex, get_temporalIndex, get_columnDensity
#from asymmetric_uncertainty import a_u



columns = ['GRB', 'B_o', 'B_o_uncert', 'epoch_start', 'epoch_end', 'A_V', 'A_V_uncert', 'host_model', 'reduced_chi2', 'probability']

GRBs = pd.read_excel("trimmed_data.xlsx", header=None, names=columns)
GRBs = GRBs.ffill()
#print(GRBs)


GRBs['epoch_start'] = pd.to_numeric(GRBs['epoch_start'], errors='coerce')
GRBs['epoch_end'] = pd.to_numeric(GRBs['epoch_end'], errors='coerce')

GRBs['time_start'] = 10**(GRBs['epoch_start'].astype(float))
GRBs['time_end'] = 10**(GRBs['epoch_end'].astype(float))
GRBs['time_mid'] = ((GRBs['time_start'].astype(float)) + (GRBs['time_end'].astype(float))) / 2


swift = pd.read_html("https://swift.gsfc.nasa.gov/archive/grb_table/fullview/",
                     attrs={"class":"grbtable"}, encoding='ISO-8859-1').pop()

swift.columns = [col[0] for col in swift.columns]

GRBs['GRB'] = GRBs['GRB'].astype(str).str.strip()
swift['GRB'] = swift['GRB'].astype(str).str.strip()


GRBs = GRBs.merge(swift[['GRB', 'Trigger Number']], on='GRB', how='left')

GRBs.to_csv('GRBs.csv')

#print(GRBs)

#add trigger number column to GRBs from swift

#convert epoch_start and epoch_end values to times (10^epoch value)

#create plots of the 'B_o' with 'B_o_uncert' as uncertainties as functions of time for every GRB that has at least 3 B_o values, plot against weighted average and calculate reduced chi squared value

red_chi_sq_values = []

results = []


for name, group in GRBs.groupby('GRB'):
		if len(group) < 3:
			continue

		weights = 1 / (group['B_o_uncert']**2)
		weighted_avg = np.sum(group['B_o'] * weights) / np.sum(weights)
		
		chi_sq = np.sum(((group['B_o'] - weighted_avg) / group['B_o_uncert'])**2)
		dof = len(group) - 1
		red_chi_sq = chi_sq / dof

		red_chi_sq_values.append(red_chi_sq)


		#can adjust my range for sure, I just sort of arbitrarily chose 0-3

		if 0 <= red_chi_sq <= 3:
			category = "within range"

		else:
			category = "outlier"


		results.append({'GRB': name, 'red_chi_sq': red_chi_sq, 'category': category})

df_results = pd.DataFrame(results)

df_results.to_csv('redchisq_results.csv')

#print(df_results['category'].value_counts())

#print(df_results)

#now that I've sort my grbs, I want to go through the outliers and see whether it is just one or two data points that are messing up the red chi sq or if they really do seem to be evolving with time


#plotting each grb w/ at least 3 B_o data points (commented out for when I don't need it lol)
'''

		plt.figure(figsize=(8, 5))
		plt.errorbar(group['time_mid'], group['B_o'], yerr=group['B_o_uncert'], 
                 fmt='o', label=f'Data (Trigger: {group["Trigger Number"].iloc[0]})')
		plt.axhline(weighted_avg, color='darkorange', linestyle='--', label=f'Weighted Avg: {weighted_avg:.2f}')
		plt.xscale('log')
		plt.xlabel('Time (s)')
		plt.ylabel('$B_o$')
		plt.title(f'GRB {name} | $\chi^2_\\nu$: {red_chi_sq:.2f}')
		plt.legend()
		plt.grid(True)
		plt.savefig(f'{name}_redchisq.pdf')

'''


'''

#plot dist of red_chi_sq values!

plt.figure(figsize=(10, 6))
plt.hist(red_chi_sq_values, bins=25, color='purple', alpha=0.8)

#plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Ideal $\chi^2_\\nu = 1$')

plt.title('Distribution of Reduced $\chi^2$ for GRB $B_o$ Fits', fontsize=14)
plt.xlabel('Reduced $\chi^2$', fontsize=12)
plt.ylabel('Number of GRBs', fontsize=12)

plt.savefig('red_chi_sq_dist.pdf')

#print(f"Mean Reduced Chi-Squared: {np.mean(red_chi_sq_values):.2f}")
#print(f"Median Reduced Chi-Squared: {np.median(red_chi_sq_values):.2f}")

'''



#pull the swift data that I need, see if I can just pull the xrt_density_pc file instead of downloading EVERYTHING lmao


#plot my opt data as a function of time again, do my red chi squared plots relative to a weighted average, and separate those with a good red_chisq from those with a bad value (plt some histograms and look at the outliers) 

#then go through and plot with the X-ray data in time as well

#see if I can temporally match my opt and X-ray data points as well
