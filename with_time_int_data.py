import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import scipy.stats

GRBs = pd.read_csv("GRBs.csv", index_col=0)
redchisq_results = pd.read_csv("redchisq_results.csv", index_col=0)


data_spec_ind = pd.read_excel("data.xlsx")

data_spec_ind = data_spec_ind.drop(columns=['Bopt2', 'Uncert2'])
#print(data_spec_ind)

#grb = data_spec_ind.iloc[0]
#print(grb)
#print(len(data_spec_ind['GRB']))




#-----------------------------PLOT HISTOGRAMS OF INDICES-----------------------------

mu_o, std_o = norm.fit(data_spec_ind['Bopt'])
mu_x, std_x = norm.fit(data_spec_ind['Bx'])
mu_ox, std_ox = norm.fit(data_spec_ind['Box'])


#xmin, xmax = plt.xlim()
x = np.linspace(0, 2.00, 100)


result_ks = scipy.stats.ks_2samp(data_spec_ind['Bopt'], data_spec_ind['Bx'])

#print(result_ks)



plt.hist(data_spec_ind['Uncert'], bins=8, color='darkmagenta', alpha=0.6, label=r'$\beta_o$ +/- Uncert')
plt.hist(-1*data_spec_ind['Uncert'], bins=8, color='darkmagenta', alpha=0.6)
plt.hist(data_spec_ind['Bx_'], bins=8, color='cornflowerblue', alpha=0.7, label=r'$\beta_x$ +/- Uncert')
plt.hist(data_spec_ind['Bx^'], bins=8, color='cornflowerblue', alpha=0.7)
plt.hist(data_spec_ind['Box_'], bins=4, color='lightpink', alpha=0.7, label=r'$\beta_{ox}$ +/- Uncert')
plt.hist(data_spec_ind['Box^'], bins=4, color='lightpink', alpha=0.7)
plt.xlabel('Spectral Index Errors of GRB sample')
plt.ylabel('Normalized Counts')
plt.legend()
plt.show()


'''
plt.hist(data_spec_ind['Bopt'], bins=15, density=True, color='darkmagenta', alpha=0.7, label=r'$\beta_o$')
plt.hist(data_spec_ind['Bx'], bins=15, density=True, color='cornflowerblue', alpha=0.7, label=r'$\beta_x$')
plt.hist(data_spec_ind['Box'], bins=15, density=True, color='lightpink', alpha=0.7, label=r'$\beta_{ox}$')

plt.plot(x, norm.pdf(x, mu_o, std_o), color='darkmagenta', linewidth=2)
plt.plot(x, norm.pdf(x, mu_x, std_x), color='royalblue', linewidth=2)
plt.plot(x, norm.pdf(x, mu_ox, std_ox), color='hotpink', linewidth=2)


plt.xlabel('Spectral Indices of GRB sample')
plt.ylabel('Normalized Counts')
plt.legend()
#plt.show()
plt.savefig('spectral_indices_hist.pdf')
'''






mu_xo, std_xo = norm.fit(data_spec_ind['Bx'] - data_spec_ind['Bopt'])
mu_oxo, std_oxo = norm.fit(data_spec_ind['Box'] - data_spec_ind['Bopt'])
mu_xox, std_xox = norm.fit(data_spec_ind['Bx'] - data_spec_ind['Box'])

x = np.linspace(-1.0, 1.25, 100)


'''
plt.hist((data_spec_ind['Bx'] - data_spec_ind['Bopt']), bins=15, density=True, color='darkmagenta', alpha=0.7, label=r'$\beta_x - \beta_o$')
plt.hist((data_spec_ind['Box'] - data_spec_ind['Bopt']), bins=15, density=True, color='cornflowerblue', alpha=0.7, label=r'$\beta_{ox} - \beta_o$')
plt.hist((data_spec_ind['Bx'] - data_spec_ind['Box']), bins=15, density=True, color='lightpink',alpha=0.7, label=r'$\beta_x - \beta_{ox}$')

plt.plot(x, norm.pdf(x, mu_xo, std_xo), color='darkmagenta', linewidth=2)
plt.plot(x, norm.pdf(x, mu_oxo, std_oxo), color='royalblue', linewidth=2)
plt.plot(x, norm.pdf(x, mu_xox, std_xox), color='hotpink', linewidth=2)

plt.xlabel('Optical -- X-ray Spectral Index Differences of GRB Sample')
plt.ylabel('Normalized Counts')
plt.legend()
#plt.show()
plt.savefig('spectral_indices_differences.pdf')

'''

mu_o_u, std_o_u = np.nanmean(data_spec_ind['Uncert']), np.nanstd(data_spec_ind['Uncert'])
mu_x_u_, std_x_u_ = norm.fit(data_spec_ind['Bx_'])
mu_ox_u_, std_ox_u_ = norm.fit(data_spec_ind['Box_'])
mu_x_u, std_x_u = norm.fit(data_spec_ind['Bx^'])
mu_ox_u, std_ox_u = norm.fit(data_spec_ind['Box^'])


print('        Average                   Standard Deviation')
print('Opt Uncert: ', mu_o_u, '            ', std_o_u)
print('- Uncert X-ray: ', mu_x_u_, '            ', std_x_u_)
print('+ Uncert X-ray: ', mu_x_u, '            ', std_x_u)
print('- Uncert OX: ', mu_ox_u_, '            ', std_ox_u_)
print('+ Uncert Ox: ', mu_ox_u, '            ', std_ox_u)


print("Opt: ", mu_o, std_o)
print("X: ", mu_x, std_x)
print("OX: ", mu_ox, std_ox)

#print('X-O: ',mu_xo, '            ', std_xo)
#print('OX-O: ', mu_oxo, '            ', std_oxo)
#print('X-OX: ', mu_xox, '            ', std_xox)



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
log_stop = 18.3   
step_size = 0.1      # the constant difference in log space

exponents = np.arange(log_start, log_stop+step_size, step_size)


vc_values = 10**exponents


v_values = np.geomspace(1e10, 1e20, 500) #just to check vm, can revert back to above after checks


F_0 = v_values[0]

v_m = 1E11




# setup!


#vc_list = [1e15,5e15, 1e16, 5e16, 1e17, 5e17, 1e18, 5e18, 1e19]
colors = plt.cm.viridis(np.linspace(0, 1, len(vc_values)))






#----------------------------CALCULATING B_OX-----------------------------------


#ok might need to talk this out first, but I'm trying to calculate B_ox given these different
#cooling break locations, right? because the slope will differ, and I'll try to match the observations
#to whichever calculated value works? 


# B_ox = (log(F_o/Fx))/(log(v_o, v_x)), so I need to pick the logarithmic midpoint of the optical (about 650nm) and the X-ray (1 keV) bands

#and then use those values in my flux calculations while cycling through the different v_c values

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
plt.axvspan(bands["Band 2 (Opt/IR)"][0], bands["Band 2 (Opt/IR)"][1], color='blue', alpha=0.1, label='Opt/IR Band Location')
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

    #print(f"{vc:.1e} | {s_b1: .4f}   | {sh_b1: .4f}   | {s_b2: .4f}   | {sh_b2: .4f}")
'''

vc_range = np.logspace(13.5, 18.3, 100) 

bands = {
    "Band 1 (X-ray)": (7.254e16, 2.418e18),
    "Band 2 (Opt)": (2.998e14, 7.495e14)
}

results = {"B1_sm": [], "B1_sm_p2": [], "B1_sm_p3": [], "B1_sh": [], "B1_sh_p2": [], "B1_sh_p3": [], "B2_sm": [], "B2_sm_p2": [], "B2_sm_p3": [], "B2_sh": [], "B2_sh_p2": [], "B2_sh_p3": []}


v_fit1 = np.geomspace(bands["Band 1 (X-ray)"][0], bands["Band 1 (X-ray)"][1], 50)
v_fit2 = np.geomspace(bands["Band 2 (Opt)"][0], bands["Band 2 (Opt)"][1], 50)

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


#grb = data_spec_ind.iloc[0]

'''
for i in range(0, len(data_spec_ind['GRB'])):
	grb = data_spec_ind.iloc[[i]]
	plt.figure(figsize=(10, 6))
	plt.plot(vc_values, results["B1_sm"], 'r-', label=r'$\beta_X$, p=2.25 (Smooth)')
	plt.plot(vc_values, results["B1_sm_p2"], 'r-', alpha=0.5, label=r'$\beta_X$, p=2.2 (Smooth)')
	plt.plot(vc_values, results["B1_sm_p3"], 'r-', alpha=0.5, label=r'$\beta_X$, p=2.3 (Smooth)')
	plt.plot(vc_values, results["B1_sh"], 'r--', alpha=0.8, label=r'$\beta_X$, p=2.25 (Sharp)')
	plt.plot(vc_values, results["B1_sh_p2"], 'r--', alpha=0.4, label=r'$\beta_X$, p=2.2 (Sharp)')
	plt.plot(vc_values, results["B1_sh_p3"], 'r--', alpha=0.4, label=r'$\beta_X$, p=2.3 (Sharp)')
	plt.plot(vc_values, results["B2_sm"], 'b-', label=r'$\beta_O$, p=2.25 (Smooth)')
	plt.plot(vc_values, results["B2_sm_p2"], 'b-', alpha=0.5, label=r'$\beta_O$, p=2.2 (Smooth)')
	plt.plot(vc_values, results["B2_sm_p3"], 'b-', alpha=0.5, label=r'$\beta_O$, p=2.3 (Smooth)')
	plt.plot(vc_values, results["B2_sh"], 'b--', alpha=0.8, label=r'$\beta_O$, p=2.25 (Sharp)')
	plt.plot(vc_values, results["B2_sh_p2"], 'b--', alpha=0.4, label=r'$\beta_O$, p=2.2 (Sharp)')
	plt.plot(vc_values, results["B2_sh_p3"], 'b--', alpha=0.4, label=r'$\beta_O$, p=2.3 (Sharp)')
	plt.plot(vc_values, B_ox_sm, color='purple', linestyle='-', label=r'$\beta_{OX}$, p=2.25 (Smooth)')
	plt.plot(vc_values, B_ox_sm_p2, color='purple', linestyle='-', alpha=0.5, label=r'$\beta_{OX}$, p=2.25 (Smooth)')
	plt.plot(vc_values, B_ox_sm_p3, color='purple', linestyle='-', alpha=0.5, label=r'$\beta_{OX}$, p=2.25 (Smooth)')
	plt.plot(vc_values, B_ox_sh, color='purple', linestyle= '--', alpha=0.8, label=r'$\beta_{OX}$, p=2.25 (Sharp)')
	plt.plot(vc_values, B_ox_sh_p2, color='purple', linestyle= '--', alpha=0.4, label=r'$\beta_{OX}$, p=2.2 (Sharp)')
	plt.plot(vc_values, B_ox_sh_p3, color='purple', linestyle= '--', alpha=0.4, label=r'$\beta_{OX}$, p=2.3 (Sharp)')

	

	plt.axvspan(bands["Band 1 (X-ray)"][0], bands["Band 1 (X-ray)"][1], color='red', alpha=0.1, label='X-ray Band')
	plt.axvspan(bands["Band 2 (Opt)"][0], bands["Band 2 (Opt)"][1], color='blue', alpha=0.1, label='Opt Band')

	name = grb['GRB'].to_string()
	print(name)


	plt.scatter(10**16, (-1*grb['Bopt']), color='navy', marker='*', label=f"{name} opt index")
	plt.scatter(10**16, (-1*grb['Bx']), color='maroon', marker='*', label=f"{name} X-ray index")

	plt.xscale('log')
	plt.ylabel('Observed Spectral Index')
	plt.xlabel(r'Cooling Break Location $\nu_c$ position [Hz]')
	#plt.title(r'Slope Evolution -- How $\nu_c$ position changes the observed fit')
	plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize='small', ncol=1)
	plt.tight_layout()
	plt.grid(True, which="both", alpha=0.2)
	#plt.show()
	plt.savefig(f"{name}_spec_ind.pdf")
'''





### note that b1 corresponds to x-rays and b2 corresponds to optical



#------------------------------O vs X------------------------------------


'''
for i in range(0, len(data_spec_ind['GRB'])):
	grb = data_spec_ind.iloc[[i]]
	name = grb['GRB'].to_string(index=False)
	#print(name)
	plt.figure(figsize=(8, 6))
	plt.plot((results["B2_sm"]), (results["B1_sm"]), color='purple', label='Smooth')
	plt.plot((results["B2_sm_p2"]), (results["B1_sm_p2"]), color='purple', alpha=0.5)
	plt.plot((results["B2_sm_p3"]), (results["B1_sm_p3"]), color='purple', alpha=0.5)

	plt.plot((results["B2_sh"]), (results["B1_sh"]), color='purple', linestyle='--', alpha=0.8, label='Sharp')
	plt.plot((results["B2_sh_p2"]), (results["B1_sh_p2"]), color='purple', linestyle='--', alpha=0.4)
	plt.plot((results["B2_sh_p3"]), (results["B1_sh_p3"]), color='purple', linestyle='--', alpha=0.4)

	plt.errorbar((-1*grb['Bopt']), (-1*grb['Bx']), xerr=grb['Uncert'], yerr=(abs(grb['Bx_']), grb['Bx^']), marker='*', color = 'gold', label=f"{name}")

	plt.xlabel(r'$\beta_O$')
	plt.ylabel(r'$\beta_X$')
	#plt.title("Optical vs X-ray Indices")
	plt.legend()
	#plt.show()
	plt.savefig(f'{name}_OvsX.pdf')
'''





'''
plt.figure(figsize=(8, 6))
plt.plot((results["B2_sm"]), (results["B1_sm"]), color='purple', label='Smooth')
plt.plot((results["B2_sm_p2"]), (results["B1_sm_p2"]), color='purple', alpha=0.5)
plt.plot((results["B2_sm_p3"]), (results["B1_sm_p3"]), color='purple', alpha=0.5)

plt.plot((results["B2_sh"]), (results["B1_sh"]), color='purple', linestyle='--', alpha=0.8, label='Sharp')
plt.plot((results["B2_sh_p2"]), (results["B1_sh_p2"]), color='purple', linestyle='--', alpha=0.4)
plt.plot((results["B2_sh_p3"]), (results["B1_sh_p3"]), color='purple', linestyle='--', alpha=0.4)

#plt.errorbar((-1*data_spec_ind['Bopt']), (-1*data_spec_ind['Bx']), xerr=data_spec_ind['Uncert'], yerr=(abs(data_spec_ind['Bx_']), data_spec_ind['Bx^']), marker='*', linestyle='none', color = 'mediumturquoise')

#plt.scatter((-1*data_spec_ind['Bopt']), (-1*data_spec_ind['Bx']), marker='*', color = 'mediumturquoise')

plt.errorbar((-1*mu_o),(-1*mu_x), xerr=std_o, yerr=std_x, marker='*', color='mediumturquoise', label=r'Average GRB $\beta_o$ vs $\beta_x$')


plt.xlabel(r'$\beta_O$')
plt.ylabel(r'$\beta_X$')
plt.legend()
plt.show()
#plt.savefig(f'OvsX_no_error.pdf')
'''






#-----------------------------OX vs X------------------------------------

'''
plt.figure(figsize=(8, 6))
plt.plot(B_ox_sm, results["B1_sm"], color='red', label='Smooth')
plt.plot(B_ox_sh, results["B1_sh"], color='red', linestyle='--', alpha=0.5, label='Sharp')
plt.xlabel(r'$\beta_{OX}$')
plt.ylabel(r'$\beta_X$')

plt.errorbar((-1*mu_ox),(-1*mu_x), xerr=std_ox, yerr=std_x, marker='*', color='mediumturquoise', label=r'Average GRB $\beta_{ox}$ vs $\beta_x$')


plt.legend()
plt.show()
#plt.savefig('OXvsX.pdf')




#-----------------------------OX vs O------------------------------------


plt.figure(figsize=(8, 6))
plt.plot(B_ox_sm, results["B2_sm"], color='b', label='Smooth')
plt.plot(B_ox_sh, results["B2_sh"], color='b', linestyle='--', alpha=0.5, label='Sharp')
plt.xlabel(r'$\beta_{OX}$')
plt.ylabel(r'$\beta_X$')

plt.errorbar((-1*mu_ox),(-1*mu_o), xerr=std_ox, yerr=std_o, marker='*', color='mediumturquoise', label=r'Average GRB $\beta_{ox}$ vs $\beta_o$')


plt.legend()
plt.show()
#plt.savefig('OXvsO.pdf')

'''








#-----------------------------------------SPECTRA & FITS--------------------------------------------

'''
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




