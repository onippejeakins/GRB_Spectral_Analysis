import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

GRBs = pd.read_csv("GRBs.csv", index_col=0)
redchisq_results = pd.read_csv("redchisq_results.csv", index_col=0)



#-------------------PLOTTING WITHIN RANGE VS OUTLIER REDCHISQ------------------------

near_ideal = redchisq_results[redchisq_results['category'] == "within range"]
outliers = redchisq_results[redchisq_results['category'] == "outlier"]


'''
print(f"Total GRBs analyzed: {len(redchisq_results)}")
print(f"Number within [0, 3]: {len(near_ideal)}")
print(f"Number of outliers: {len(outliers)}")

# Print the names outliers vs within range
print("\n--- Outlier GRBs ---")
print(outliers[['GRB', 'red_chi_sq']].to_string(index=False))

print("\n--- Within Range GRBs ---")
print(near_ideal[['GRB', 'red_chi_sq']].to_string(index=False))
'''


'''
plt.figure(figsize=(10, 6))

plt.hist(near_ideal['red_chi_sq'], bins=5, alpha=0.7, color='forestgreen', label='Near Ideal (0-3)')


plt.hist(outliers['red_chi_sq'], bins=20, alpha=0.7, color='firebrick', label='Outliers (>3)')

plt.axvline(1.0, color='black', linestyle='--', label='Ideal $\chi^2_\\nu = 1$')
plt.xlabel('Reduced $\chi^2$')
plt.ylabel('Count')
plt.title('Categorized Distribution of Reduced $\chi^2$')
plt.xscale('log')
plt.legend()
#plt.show()
plt.savefig('split_redchisq_hist.pdf')
'''





#-------------------------------BUILDING GRB SPECTRA---------------------------------


def spectrum(F, v, vm, vc, p):
	sm = 2.2 - (0.52*p)
	sc = 1.6 - (0.38*p)
	flux = F*(((v/vm)**(-sm/3))+((v/vm)**(-sm*(1-p)*0.5)))**(-1/sm)*(1+((v/vc)**(sc/2)))**(-1/sc)
	return flux


def sharp_break(F_norm, v, vm, vc, p):
	conditions = [v <= vm, (v > vm) & (v <= vc), v>vc]
	f1 = lambda v: F_norm * (v/vm)**(1/3)
	f2 = lambda v: F_norm * (v/vm)**((1-p)/2)
	f3 = lambda v: (F_norm * (vc/vm)**((1-p)/2) * (v/vc)**(-p/2))

	return np.piecewise(v, conditions, [f1, f2, f3])



#defining my frequency arrays to step in log space (blegh)
log_start = 13.5    
log_stop = 19   
step_size = 0.1      # The constant difference in log space

exponents = np.arange(log_start, log_stop+step_size, step_size)


vc_values = 10**exponents

#v_values = 10**exponents


v_values = np.geomspace(1e10, 1e20, 500) #just to check vm, can revert back to above after checks


#print(len(vc_values))

F_0 = v_values[0]

v_m = 1E11




# setup!


p = 2.25
vc_list = [1e15,5e15, 1e16, 5e16, 1e17, 5e17, 1e18, 5e18, 1e19]
colors = plt.cm.viridis(np.linspace(0, 1, len(vc_values)))






#------------------------------PLOT EVOLUTION---------------------------------------


bands = {
    "Band 1 (X-ray)": (7.254e16, 2.418e18),
    "Band 2 (Opt)": (2.998e14, 7.495e14)
}


'''
plt.figure(figsize=(8, 6))

for i, v_c in enumerate(vc_values):
		flux_smooth = spectrum(F_0, v_values, v_m, v_c)
		plt.plot(v_values, flux_smooth, alpha=0.6, color=colors[i])
		flux_sharp = sharp_break(F_0, v_values, v_m, v_c, p)
		plt.plot(v_values, flux_sharp, '--', alpha=0.6, color=colors[i])

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Frequency $(\nu)$ [Hz]', fontsize=14)
plt.ylabel('Flux (F)', fontsize=14)
plt.title(rf'Smooth vs. Sharp Spectral Models for Changing $\nu_c$ ($p={p}$)', fontsize=16)
plt.axvspan(bands["Band 1 (X-ray)"][0], bands["Band 1 (X-ray)"][1], color='red', alpha=0.1, label='X-ray Band Location')
plt.axvspan(bands["Band 2 (Opt)"][0], bands["Band 2 (Opt)"][1], color='blue', alpha=0.1, label='Opt Band Location')
#plt.axvline(x=7.254E16, color='r', linestyle='--', label='Swift XRT X-ray range')
#plt.axvline(x=2.418E18, color='r', linestyle='--')
#plt.axvline(x=7.495E14, color='b', linestyle='--', label='Optical range')
#plt.axvline(x=2.998E14, color='b', linestyle='--')
#plt.legend(loc='lower left')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
#plt.show()
plt.savefig('smooth_vs_sharp_spectra_poster.pdf')
'''


'''
plt.figure(figsize=(10, 6))

for i, v_c in enumerate(vc_values):
    # sharp break model
    flux_sharp = sharp_break(F_0, v_values, v_m, v_c, p)
    plt.plot(v_values, flux_sharp, '-', color=colors[i], label=f'Sharp $v_c$={v_c:.0e}')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Frequency $(\nu)$ [Hz]')
plt.ylabel('Flux (F)')
plt.title(rf'Sharp Spectral Model for Changing $\nu_c$ ($p={p}$)')
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.show()
#plt.savefig('sharp_spectra.pdf')
'''






#-----------------------------INDIVIDUALLY-----------------------------------

'''
bands = {
    "Band 1 (X-ray)": (7.254e16, 2.418e18),
    "Band 2 (Opt)": (2.998e14, 7.495e14)
}




for v_c in vc_values:
	flux_values_smooth = []
	flux_values_sharp = []
	for v in v_values:
		flux_values_smooth.append(spectrum(F_0, v, v_m, v_c))
		flux_values_sharp.append(sharp_break(F_0, v, v_m, v_c, p))
		#print(flux_values)
	plt.figure(figsize=(10,6))
	plt.title(rf'Smooth vs. Sharp Spectral Model ($p={p}$, $\nu_c$ = {v_c:.3e})')
	plt.plot(v_values, flux_values_smooth, color='navy', linewidth=2, label='Smooth')
	plt.plot(v_values, flux_values_sharp, color='orange', linewidth=2, label='Sharp')
	plt.axvspan(bands["Band 1 (X-ray)"][0], bands["Band 1 (X-ray)"][1], color='red', alpha=0.1, label='X-ray Band Location')
	plt.axvspan(bands["Band 2 (Opt)"][0], bands["Band 2 (Opt)"][1], color='blue', alpha=0.1, label='Opt Band Location')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'Frequency ($\nu$) [Hz]')
	plt.ylabel('Flux (F)')
	plt.grid(True, which="both", ls="-", alpha=0.2)
	plt.legend()
	plt.show()
	#plt.savefig(f'{v_c:.3e}_sharp_vs_smooth.pdf')


'''



#----------------------------CALCULATING B_OX-----------------------------------


#ok might need to talk this out first, but I'm trying to calculate B_ox given these different
#cooling break locations, right? because the slope will differ, and I'll try to match the observations
#to whichever calculated value works? 


# B_ox = (log(F_o/Fx))/(log(v_o, v_x)), so I need to pick the logarithmic midpoint of the optical (about 650nm) and the X-ray (1 keV) bands

#and then use those values in my flux calculations while cycling through the different v_c values

B_ox_smooth = []
B_ox_sharp = []

B_ox_sm = []
B_ox_sh = []


for i, v_c in enumerate(vc_values):
		flux_smooth_opt = spectrum(F_0, (4.61E14), v_m, v_c, 2.25)
		flux_smooth_x = spectrum(F_0, (2.418E17), v_m, v_c, 2.25)
		flux_sharp_opt = sharp_break(F_0, (4.61E14), v_m, v_c, 2.25)
		flux_sharp_x = sharp_break(F_0, (2.418E17), v_m, v_c, 2.25)
		B_ox_sm.append((np.log10(flux_smooth_opt/flux_smooth_x))/(np.log10(4.61E14/2.418E17)))
		B_ox_sh.append((np.log10(flux_sharp_opt/flux_sharp_x))/(np.log10(4.61E14/2.418E17)))



B_ox_sm_p2 = []
B_ox_sh_p2 = []


for i, v_c in enumerate(vc_values):
		flux_smooth_opt = spectrum(F_0, (4.61E14), v_m, v_c, 2.2)
		flux_smooth_x = spectrum(F_0, (2.418E17), v_m, v_c, 2.2)
		flux_sharp_opt = sharp_break(F_0, (4.61E14), v_m, v_c, 2.2)
		flux_sharp_x = sharp_break(F_0, (2.418E17), v_m, v_c, 2.2)
		B_ox_sm_p2.append((np.log10(flux_smooth_opt/flux_smooth_x))/(np.log10(4.61E14/2.418E17)))
		B_ox_sh_p2.append((np.log10(flux_sharp_opt/flux_sharp_x))/(np.log10(4.61E14/2.418E17)))



B_ox_sm_p3 = []
B_ox_sh_p3 = []


for i, v_c in enumerate(vc_values):
		flux_smooth_opt = spectrum(F_0, (4.61E14), v_m, v_c, 2.3)
		flux_smooth_x = spectrum(F_0, (2.418E17), v_m, v_c, 2.3)
		flux_sharp_opt = sharp_break(F_0, (4.61E14), v_m, v_c, 2.3)
		flux_sharp_x = sharp_break(F_0, (2.418E17), v_m, v_c, 2.3)
		B_ox_sm_p3.append((np.log10(flux_smooth_opt/flux_smooth_x))/(np.log10(4.61E14/2.418E17)))
		B_ox_sh_p3.append((np.log10(flux_sharp_opt/flux_sharp_x))/(np.log10(4.61E14/2.418E17)))



'''
plt.plot(vc_values, B_ox_smooth, color='navy', label=r"Smooth $\beta_{ox}$")
plt.plot(vc_values, B_ox_sharp, color='orange', label=r"Sharp $\beta_{ox}$")
plt.axvspan(bands["Band 1 (X-ray)"][0], bands["Band 1 (X-ray)"][1], color='red', alpha=0.1, label='X-ray Band Location')
plt.axvspan(bands["Band 2 (Opt)"][0], bands["Band 2 (Opt)"][1], color='blue', alpha=0.1, label='Opt Band Location')
plt.xscale('log')
plt.xlabel(r'Cooling Break Location $(\nu_c)$ [Hz]')
plt.ylabel(r'$\beta_{ox}$')
plt.legend()
plt.show()
#plt.savefig('b_ox_smooth_vs_sharp.pdf')
'''



#---------------------------------SLOPES-------------------------------------

def get_slope(v_range, flux_values):
    log_v = np.log10(v_range)
    log_f = np.log10(flux_values)
    slope, _ = np.polyfit(log_v, log_f, 1)
    return slope

# Bands (Hz)
bands = {
    "Band 1 (X-ray)": (7.254e16, 2.418e18),
    "Band 2 (Opt)": (2.998e14, 7.495e14)
}


v_fit1 = np.geomspace(bands["Band 1 (X-ray)"][0], bands["Band 1 (X-ray)"][1], 20)
v_fit2 = np.geomspace(bands["Band 2 (Opt)"][0], bands["Band 2 (Opt)"][1], 20)

#print(f"{'vc':<9} | {'Smooth B1':<10} | {'Sharp B1':<10} | {'Smooth B2':<10} | {'Sharp B2':<10}")
#print("-" * 65)


'''
for vc in vc_values:

    s_b1 = get_slope(v_fit1, spectrum(1.0, v_fit1, v_m, vc))
    sh_b1 = get_slope(v_fit1, sharp_break(1.0, v_fit1, v_m, vc)) 

    s_b2 = get_slope(v_fit2, spectrum(1.0, v_fit2, v_m, vc))
    sh_b2 = get_slope(v_fit2, sharp_break(1.0, v_fit2, v_m, vc))
'''



    #print(f"{vc:.1e} | {s_b1: .4f}   | {sh_b1: .4f}   | {s_b2: .4f}   | {sh_b2: .4f}")


v_m = 1e11
p = 2.25
vc_range = np.logspace(13.5, 18.3, 100) 

bands = {
    "Band 1 (X-ray)": (7.254e16, 2.418e18),
    "Band 2 (Opt)": (2.998e14, 7.495e14)
}



#results = {"B1_sm": [], "B1_sh": [], "B2_sm": [], "B2_sh": []}


results = {"B1_sm": [], "B1_sm_p2": [], "B1_sm_p3": [], "B1_sh": [], "B1_sh_p2": [], "B1_sh_p3": [], "B2_sm": [], "B2_sm_p2": [], "B2_sm_p3": [], "B2_sh": [], "B2_sh_p2": [], "B2_sh_p3": []}





v_fit1 = np.geomspace(bands["Band 1 (X-ray)"][0], bands["Band 1 (X-ray)"][1], 50)
v_fit2 = np.geomspace(bands["Band 2 (Opt)"][0], bands["Band 2 (Opt)"][1], 50)




'''
for vc in vc_values:
    results["B1_sm"].append(get_slope(v_fit1, spectrum(1.0, v_fit1, v_m, vc)))
    results["B1_sh"].append(get_slope(v_fit1, sharp_break(1.0, v_fit1, v_m, vc)))
    results["B2_sm"].append(get_slope(v_fit2, spectrum(1.0, v_fit2, v_m, vc)))
    results["B2_sh"].append(get_slope(v_fit2, sharp_break(1.0, v_fit2, v_m, vc)))
'''


for vc in vc_values:
		results["B1_sm"].append(get_slope(v_fit1, spectrum(1.0, v_fit1, v_m, vc, 2.25)))
		results["B1_sm_p2"].append(get_slope(v_fit1, spectrum(1.0, v_fit1, v_m, vc, 2.2)))
		results["B1_sm_p3"].append(get_slope(v_fit1, spectrum(1.0, v_fit1, v_m, vc, 2.3)))
		results["B1_sh"].append(get_slope(v_fit1, sharp_break(1.0, v_fit1, v_m, vc, 2.25)))
		results["B1_sh_p2"].append(get_slope(v_fit1, sharp_break(1.0, v_fit1, v_m, vc, 2.2)))
		results["B1_sh_p3"].append(get_slope(v_fit1, sharp_break(1.0, v_fit1, v_m, vc, 2.3)))
		results["B2_sm"].append(get_slope(v_fit2, spectrum(1.0, v_fit2, v_m, vc, 2.25)))
		results["B2_sm_p2"].append(get_slope(v_fit2, spectrum(1.0, v_fit2, v_m, vc, 2.2)))
		results["B2_sm_p3"].append(get_slope(v_fit2, spectrum(1.0, v_fit2, v_m, vc, 2.3)))
		results["B2_sh"].append(get_slope(v_fit2, sharp_break(1.0, v_fit2, v_m, vc, 2.25)))
		results["B2_sh_p2"].append(get_slope(v_fit2, sharp_break(1.0, v_fit2, v_m, vc, 2.2)))
		results["B2_sh_p3"].append(get_slope(v_fit2, sharp_break(1.0, v_fit2, v_m, vc, 2.3)))


plt.figure(figsize=(10, 6))


'''
plt.plot(vc_values, results["B1_sm"], 'r-', label=r'$\beta_X$ (Smooth)')
plt.plot(vc_values, results["B1_sh"], 'r--', alpha=0.4, label=r'$\beta_X$ (Sharp)')
plt.plot(vc_values, results["B2_sm"], 'b-', label=r'$\beta_O$ (Smooth)')
plt.plot(vc_values, results["B2_sh"], 'b--', alpha=0.4, label=r'$\beta_O$ (Sharp)')
#plt.plot(vc_values, B_ox_smooth, color='purple', linestyle='-', label=r'$\beta_{OX}$ (Smooth)')
#plt.plot(vc_values, B_ox_sharp, color='purple', linestyle= '--', alpha=0.4, label=r'$\beta_{OX}$ (Sharp)')
'''

plt.plot(vc_values, results["B2_sm"], color='dodgerblue', linestyle='-', label=r'$\beta_O$, p=2.25 (Smooth)')
plt.fill_between(vc_values, results['B2_sm_p2'], results['B2_sm_p3'], color='dodgerblue', alpha=0.1)




plt.plot(vc_values, B_ox_sm, color='purple', linestyle='-', label=r'$\beta_{OX}$, p=2.25 (Smooth)')
plt.fill_between(vc_values, B_ox_sm_p2, B_ox_sm_p3, color='purple', alpha=0.1)





plt.plot(vc_values, results["B1_sm"], color='crimson', linestyle='-', label=r'$\beta_X$, p=2.25 (Smooth)')
plt.fill_between(vc_values, results['B1_sm_p2'], results['B1_sm_p3'], color='deeppink', alpha=0.1)

#plt.plot(vc_values, results["B1_sm_p2"], 'r-', alpha=0.5)
#plt.plot(vc_values, results["B1_sm_p3"], 'r-', alpha=0.5)

#plt.plot(vc_values, results["B1_sh"], color='crimson', linestyle='--', alpha=0.8, label=r'$\beta_X$, p=2.25 (Sharp)')
#plt.fill_between(vc_values, results['B1_sh_p2'], results['B1_sh_p3'], color='deeppink', hatch='/', alpha=0.1)

#plt.plot(vc_values, results["B1_sh_p2"], 'r--', alpha=0.4)
#plt.plot(vc_values, results["B1_sh_p3"], 'r--', alpha=0.4)
i
#plt.plot(vc_values, B_ox_sharp, color='purple', linestyle= '--', alpha=0.4, label=r'$\beta_{OX}$ (Sharp)')


#plt.plot(vc_values, results["B2_sm_p2"], 'b-', alpha=0.5)
#plt.plot(vc_values, results["B2_sm_p3"], 'b-', alpha=0.5)

#plt.plot(vc_values, results["B2_sh"], color='dodgerblue', linestyle='--', alpha=0.8, label=r'$\beta_O$, p=2.25 (Sharp)')
#plt.fill_between(vc_values, results['B2_sh_p2'], results['B2_sh_p3'], color='dodgerblue', hatch='/', alpha=0.1)

#plt.plot(vc_values, results["B2_sh_p2"], 'b--', alpha=0.4)
#plt.plot(vc_values, results["B2_sh_p3"], 'b--', alpha=0.4)




plt.hlines(-0.71, xmin=(10**16.1), xmax=(10**17.1), color='mediumblue', linestyle='--', label=r'Average Observed $\beta_o$')
plt.plot((3.2E16), -0.71, marker='*', color='mediumblue', ms=13)
plt.hlines(-0.78, xmin=(5.5E16), xmax=(2.5E17), color='rebeccapurple', linestyle='--', label=r'Average Observed $\beta_{ox}$')
plt.plot((1.1E17), -0.78, marker='*', color='rebeccapurple', ms=13)
plt.hlines(-0.91, xmin=(1.1E17), xmax=(3.5E17), color='brown', linestyle='--', label=r'Average Observed $\beta_x$')
plt.plot((2.0E17), -0.91, marker='*', color='brown', ms=13)




plt.axvspan(bands["Band 2 (Opt)"][0], bands["Band 2 (Opt)"][1], color='blue', alpha=0.1, label='Opt Band')
plt.axvspan(bands["Band 1 (X-ray)"][0], bands["Band 1 (X-ray)"][1], color='red', alpha=0.1, label='X-ray Band')




plt.xscale('log')
plt.xlim(10**13.5, 10**19)
plt.ylabel('Observed Spectral Index', fontsize=14)
plt.xlabel(r'Cooling Break Location $\nu_c$ position [Hz]', fontsize=14)
#plt.title(r'Slope Evolution -- How $\nu_c$ position changes the observed fit')
plt.legend(loc='upper left', fontsize=9.5, ncol=1)
plt.grid(True, which="both", alpha=0.2)
#plt.show()
plt.savefig('spectral_indices_stars_poster.pdf')





#------------------------------O vs X------------------------------------

'''
plt.figure(figsize=(8, 6))
plt.plot(results["B2_sm"], results["B1_sm"], color='purple', label='Smooth')
plt.plot(results["B2_sh"], results["B1_sh"], color='purple', linestyle='--', alpha=0.5, label='Sharp')
plt.xlabel(r'$\beta_O$')
plt.ylabel(r'$\beta_X$')
#plt.title("Optical vs X-ray Indices")
plt.legend()
#plt.show()
#plt.savefig('OvsX.pdf')



#-----------------------------OX vs X------------------------------------


plt.figure(figsize=(8, 6))
plt.plot(B_ox_smooth, results["B1_sm"], color='red', label='Smooth')
plt.plot(B_ox_sharp, results["B1_sh"], color='red', linestyle='--', alpha=0.5, label='Sharp')
plt.xlabel(r'$\beta_{OX}$')
plt.ylabel(r'$\beta_X$')
#plt.title("Broadband (OX) vs Local X-ray Index")
plt.legend()
#plt.show()
#plt.savefig('OXvsX.pdf')




#-----------------------------OX vs O------------------------------------


plt.figure(figsize=(8, 6))
plt.plot(B_ox_smooth, results["B2_sm"], color='b', label='Smooth')
plt.plot(B_ox_sharp, results["B2_sh"], color='b', linestyle='--', alpha=0.5, label='Sharp')
plt.xlabel(r'$\beta_{OX}$')
plt.ylabel(r'$\beta_X$')
#plt.title("Broadband (OX) vs Local X-ray Index")
plt.legend()
#plt.show()
#plt.savefig('OXvsO.pdf')
'''





'''
# Convert vc_values to log scale for a better color transition
log_vc = np.log10(vc_values) 

def create_index_plot(x_data, y_data, x_label, y_label, title):
    plt.figure(figsize=(9, 6))
    
    # Use scatter to apply the colormap to the points
    # 'c' sets the values for the color map, 'cmap' chooses the colors (e.g., 'viridis', 'plasma', 'coolwarm')
    sc = plt.scatter(x_data, y_data, c=log_vc, cmap='viridis', s=20, edgecolor='none')
    
    # Add a faint line to show the trajectory/direction of evolution
    plt.plot(x_data, y_data, 'k-', alpha=0.2)

    # Add the color bar
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$\log_{10}(\nu_c)$', fontsize=12)

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

# 1. Optical vs. X-ray
create_index_plot(results["B2_sm"], results["B1_sm"], 
                  r'$\beta_O$ (Optical)', r'$\beta_X$ (X-ray)', 
                  'Index-Index: Optical vs X-ray')

# 2. Broadband (OX) vs. X-ray
create_index_plot(B_ox_smooth, results["B1_sm"], 
                  r'$\beta_{OX}$', r'$\beta_X$ (X-ray)', 
                  'Index-Index: OX vs X-ray')
'''





#-----------------------------------------SPECTRA & FITS--------------------------------------------


def get_linear_fit(v_band, flux_vals):
    log_v = np.log10(v_band)
    log_f = np.log10(flux_vals)
    m, b = np.polyfit(log_v, log_f, 1)
    return 10**(m * log_v + b), m


v_m = 1e11
p = 2.25
vc_list = [1e15, 5e15, 1e16, 5e16, 1e17, 5e17, 1e18, 5e18, 1e19]
v_axis = np.geomspace(1e13, 1e20, 1000)

bands = {
    "X-ray": np.geomspace(7.254e16, 2.418e18, 20),
    "Opt": np.geomspace(2.998e14, 7.495e14, 20)
}
'''
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(vc_list)))

for i, vc in enumerate(vc_list):
    c = colors[i]
    
    f_smooth_full = spectrum(1.0, v_axis, v_m, vc)
    f_sharp_full = sharp_break(1.0, v_axis, v_m, vc)
    
    plt.loglog(v_axis, f_smooth_full, color=c, alpha=0.15, lw=1)
    plt.loglog(v_axis, f_sharp_full, color=c, alpha=0.1, ls='--', lw=1)

    for name, v_band in bands.items():
        # smooth
        flux_sm = spectrum(1.0, v_band, v_m, vc)
        fit_sm, m_sm = get_linear_fit(v_band, flux_sm)
        plt.loglog(v_band, fit_sm, color=c, lw=2.5, solid_capstyle='round')
        
        # sharp
        flux_sh = sharp_break(1.0, v_band, v_m, vc)
        fit_sh, m_sh = get_linear_fit(v_band, flux_sh)
        plt.loglog(v_band, fit_sh, color=c, lw=1.5, ls=':', alpha=0.8)



from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='gray', lw=2),
                Line2D([0], [0], color='gray', lw=1.5, ls=':'),
                Line2D([0], [0], color='red', alpha=0.2, lw=8),
                Line2D([0], [0], color='blue', alpha=0.2, lw=8)]

plt.legend(custom_lines, ['Smooth Fit', 'Sharp Fit', 'X-ray Band', 'Opt Band'], loc='lower left')



plt.axvspan(7.254e16, 2.418e18, color='red', alpha=0.05)
plt.axvspan(2.998e14, 7.495e14, color='blue', alpha=0.05)




plt.title(f"Spectral Slopes for p={p} across multiple " + r"$\nu_c$")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Flux")
plt.ylim(1e-6, 2)
plt.xlim(1e13, 1e20)
plt.grid(True, which="both", alpha=0.2)
plt.tight_layout()
#plt.show()
'''

def get_linear_fit(v_band, flux_vals):
    log_v = np.log10(v_band)
    log_f = np.log10(flux_vals)
    m, b = np.polyfit(log_v, log_f, 1)
    return 10**(m * log_v + b), m

# Setup
v_m = 1e11
p = 2.25
vc_list = [1e15, 5e15, 1e16, 5e16, 1e17, 5e17, 1e18, 5e18, 1e19]
colors = plt.cm.viridis(np.linspace(0, 1, len(vc_list)))

# Define the zoomed-in x-axes for the plots
v_ranges = {
    "Band 1 (X-ray)": np.geomspace(10**(16.5), 10**(18.5), 500),
    "Band 2 (Opt)": np.geomspace(10**(14.3), 1e15, 500)
}

# specific fitting windows
fit_windows = {
    "Band 1 (X-ray)": np.geomspace(7.254e16, 2.418e18, 50),
    "Band 2 (Opt)": np.geomspace(2.998e14, 7.495e14, 50)
}
'''
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, (name, v_axis) in enumerate(v_ranges.items()):
    ax = axes[i]
    v_fit = fit_windows[name]
    
    for j, vc in enumerate(vc_list):
        c = colors[j]
        
        # Plot full models (faint background)
        ax.loglog(v_axis, spectrum(1.0, v_axis, v_m, vc), color=c, alpha=0.1, lw=1)
        ax.loglog(v_axis, sharp_break(1.0, v_axis, v_m, vc), color=c, alpha=0.1, ls='--', lw=1)
        
        # Calculate and plot smooth fit
        fit_sm, m_sm = get_linear_fit(v_fit, spectrum(1.0, v_fit, v_m, vc))
        ax.loglog(v_fit, fit_sm, color=c, lw=2.5, label=f'v_c={v_c:.1e} (B={m_sm:.2f})')
        
        # Calculate and plot sharp fit
        fit_sh, m_sh = get_linear_fit(v_fit, sharp_break(1.0, v_fit, v_m, vc))
        ax.loglog(v_fit, fit_sh, color=c, lw=1.5, ls=':', alpha=0.8)

    ax.set_title(f"Zoom: {name}", fontsize=14)
    ax.set_xlabel("Frequency (Hz)")
    ax.grid(True, which="both", alpha=0.1)
    
    if i == 0:
        ax.set_ylabel("Flux")
        ax.legend(fontsize='x-small', loc='lower left', ncol=2)

plt.tight_layout()
#plt.show()
'''








#------------------------------ANIMATION?------------------------------------

'''
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(10, 6))

line_full_sm, = ax.loglog([], [], color='black', alpha=0.15, lw=1, label='Full Smooth Model')
line_full_sh, = ax.loglog([], [], color='black', alpha=0.1, ls='--', lw=1, label='Full Sharp Model')
vline = ax.axvline(0, color='orange', ls='--', alpha=0.5)

band_lines = {}
for name in bands.keys():
    color = 'crimson' if name == "X-ray" else 'royalblue'
    line_sm, = ax.loglog([], [], color=color, lw=3, label=f'{name} Smooth')
    line_sh, = ax.loglog([], [], color=color, lw=1.5, ls=':', alpha=0.7, label=f'{name} Sharp')
    band_lines[name] = {'sm': line_sm, 'sh': line_sh}

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Flux")
ax.set_ylim(1e-7, 2)
ax.set_xlim(min(v_full), max(v_full)) 
ax.grid(True, which="both", alpha=0.1)
legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()

def update(vc):
    f_sm_full = spectrum(1.0, v_full, v_m, vc)
    f_sh_full = sharp_break(1.0, v_full, v_m, vc)
    line_full_sm.set_data(v_full, f_sm_full)
    line_full_sh.set_data(v_full, f_sh_full)

    vline.set_xdata([vc])
    
    for name, v_band in bands.items():
        # Smooth fit
        f_sm_band = spectrum(1.0, v_band, v_m, vc)
        fit_sm, m_sm = get_linear_fit(v_band, f_sm_band)
        band_lines[name]['sm'].set_data(v_band, fit_sm)
        band_lines[name]['sm'].set_label(f'{name} Smooth (β={m_sm:.2f})')
        
        # Sharp fit
        f_sh_band = sharp_break(1.0, v_band, v_m, vc)
        fit_sh, m_sh = get_linear_fit(v_band, f_sh_band)
        band_lines[name]['sh'].set_data(v_band, fit_sh)
        band_lines[name]['sh'].set_label(f'{name} Sharp (β={m_sh:.2f})')

    ax.set_title(f"Spectrum & Fits: $\\nu_c$ = {vc:.3e} Hz")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    return [line_full_sm, line_full_sh, vline] + \
           [l for b in band_lines.values() for l in b.values()]

ani = FuncAnimation(fig, update, frames=vc_values, interval=100, blit=False)

plt.show()


'''




