from adjustText import adjust_text
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns, astropy.units as u
from matplotlib import ticker
from astropy.io import votable, ascii
from astropy.coordinates import SkyCoord
#from gdpyc.core import GasMap, DustMap
from scipy import interpolate, stats, optimize
#from src.xrt import XRT_lightcurve, get_photonIndex, get_temporalIndex, get_columnDensity
from scipy import interpolate, integrate
import requests
from bs4 import BeautifulSoup as bs
from astropy.coordinates import SkyCoord
from astropy.table import Table

def effective_wavelength(filter_response, show_plot=False): # pass a dataframe with columns Wavelength (in Ang), Transmission (in %)
    filter_response.sort_values(by="Wavelength",inplace=True)

    vega_spec = pd.read_table("http://svo2.cab.inta-csic.es/svo/theory/fps3/morefiles/vega.dat",
                              delimiter=" ",header=None,names=["Wavelength","Flux"])
    vega_func = interpolate.interp1d(vega_spec["Wavelength"],vega_spec["Flux"],
                                     bounds_error=False,fill_value=0)
    λ = filter_response["Wavelength"]
    T = filter_response["Transmission"]
    Vg = vega_func(λ)

    numerator = integrate.trapz(y=λ*T*Vg, x=λ) # ∫ λ T(λ) Vg(λ) dλ
    denominator = integrate.trapz(y=T*Vg, x=λ) # ∫ T(λ) Vg(λ) dλ
    
    lambda_eff = numerator/denominator
    
    if show_plot:
        fig,ax = plt.subplots()
        ax.plot(vega_spec.Wavelength,vega_spec.Flux,"b",label="Vega spectrum")
        ax.set_yscale("log")
        ax.set_xlim(filter_response["Wavelength"].min(), filter_response["Wavelength"].max())
        ax.set_ylabel("Flux [erg/cm$^2$/s/Ang]")
        ax.set_xlabel(r"$\lambda$ [Ang]")
        ax2 = ax.twinx()
        ax2.plot(λ,T,color="dimgrey",linestyle=":",label="Filter response")
        ax2.set_ylabel("Transmission [%]")
        ax2.axvline(lambda_eff,color="g",linestyle="--",label=r"$\lambda_\mathrm{eff}$")
        ax2.set_ylim(0,100)
        fig.legend(loc="upper right")
        plt.show()

    return lambda_eff

def lightcurve(grb, band="optical", xlimits=False, ylimits=False, **kwargs):
    """Intended for use in an environment where DataFrames called `xrt_data` and `new_optical`
    already exist and contain afterglow optical/X-ray flux data over time, respectively. This
    function uses that information to plot the light curve of the burst in a band of the user's
    choice."""

    global xrt_data
    global new_optical
    if band == "xray":
        subset = xrt_data.loc[xrt_data["GRB"]==grb]
        neg_err = [0.4*flux.value if np.isinf(flux.minus) else flux.minus for flux in subset["SpecFlux"]]
        pos_err = [0.4*flux.value if np.isinf(flux.plus) else flux.plus for flux in subset["SpecFlux"]]
        plt.errorbar(subset.Time,[flux.value for flux in subset.SpecFlux],
                     xerr=np.array(subset.Tneg,subset.Tpos).T, yerr=np.array((neg_err,pos_err)),
                     linestyle="", capthick=0, **kwargs)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(f"Swift XRT Lightcurve for GRB {grb}")
        plt.xlabel("Time since trigger [s]")
        plt.ylabel("Flux (~1 keV) [Jy]")
        plt.grid(linestyle="--")
        if xlimits:
            plt.xlim(xlimits)
        if ylimits:
            plt.ylim(ylimits)
    else:
        if band == "optical":
            subset = new_optical.loc[(new_optical["GRB"]==grb) & (new_optical["λ_eff"]>=3000) & (new_optical["λ_eff"]<=8000)]
        elif band == "UV":
            subset = new_optical.loc[(new_optical["GRB"]==grb) & (new_optical["λ_eff"]<3000)]
        elif band == "IR":
            subset = new_optical.loc[(new_optical["GRB"]==grb) & (new_optical["λ_eff"]>8000)]
        else:
            print("Unrecognized band selected")
            return None
        
        fig,ax = plt.subplots()
        neg_err = [0.4*flux.value if np.isinf(flux.minus) else flux.minus for flux in subset["Flux (Jy)"]]
        pos_err = [0.4*flux.value if np.isinf(flux.plus) else flux.plus for flux in subset["Flux (Jy)"]]
        ax.errorbar(subset["Time (s)"],[flux.value for flux in subset["Flux (Jy)"]], marker=".", linestyle="", capthick=0,
                    yerr=np.array((neg_err, [point.plus for point in subset["Flux (Jy)"]])),
                    uplims=[np.isinf(point.minus) for point in subset["Flux (Jy)"]],
                    lolims=[np.isinf(point.plus) for point in subset["Flux (Jy)"]], **kwargs)
        ax.grid(linestyle="--")
        ax.set(xscale="log",yscale="log",xlabel="Time (s)",ylabel=f"Flux ({band}) [Jy]")
        if xlimits:
            ax.set(xlim=xlimits)
        if ylimits:
            ax.set(ylim=ylimits)
    plt.show()

def simulate_spectrum(idx):
    """Intended for use in an environment where a DataFrame called `results` already
    exists and contains matched optical/X-ray data points. This function uses that
    information to plot the broadband "spectrum" at a selected point."""

    global results
    inter_freqs = np.linspace(results.loc[idx,"nu_o"],0.3/4.135667696e-18,100)
    xray_freqs = np.linspace(0.3,10,100)/4.135667696e-18
    
    ox_spec = inter_freqs**(-results.loc[idx,"B_ox"].value)
    ox_spec *= (results.loc[idx,"F_o"]/ox_spec[0])
    x_spec = xray_freqs**(-results.loc[idx,"B_x"])
    x_spec *= (ox_spec[-1]/x_spec[0])
    plt.plot(inter_freqs,ox_spec,label=r"$\beta_{ox}=%f$"%(-results.loc[idx,"B_ox"]))
    plt.plot(xray_freqs,x_spec,label=r"$\beta_x=%f$"%(-results.loc[idx,"B_x"]))
    plt.vlines([10**np.mean((np.log10(0.3),np.log10(10))) / 4.135667696e-18],0,
                10**np.log10((x_spec.max()+x_spec.min())/2),"k",linestyle="--")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    #plt.gca().set_yticklabels([])
    plt.show()

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
from astropy.coordinates import SkyCoord
from astropy.table import Table


import matplotlib.pyplot as plt
import warnings
from math import erf

np_erf = np.vectorize(erf, doc="Vectorized error function at x.")  # prevents need for SciPy dependency

class a_u:
    """
    Class for representing and handling propagation of asymmetric uncertainties assuming a pseudo-Gaussian
    probability distribution where the plus and minus errors in each direction of the nominal value are like
    modified 1-sigma standard deviations.

    Parameters
    ----------
    numeric : nominal
        the nominal value of the represented quantity
    numeric : pos_err
        the plus error on the value
    numeric : neg_err
        the minus error on the value

    Attributes
    ----------
    numeric : value
        the nominal value of the represented quantity
    numeric : plus
        the positive error on the value
    numeric : plus
        the negative error on the value
    """
    
    def __init__(self, nominal, pos_err=0, neg_err=0):
        if isinstance(nominal,str):
            stripped = nominal.replace(" ","")
            if "±" in stripped:
                self.value = float(stripped.split("±")[0])
                self.plus = self.minus = float(nominal.split("±")[1])
            elif "+" in stripped and "-" in stripped:
                self.value = float(stripped.split("(")[0])
                err_str = stripped.split("(")[1].replace(")","")
                self.plus = float(err_str.split(",")[0][1:])
                self.minus = float(err_str.split(",")[1][1:])
            else:
                raise ValueError("Failed to parse string, likely due to improper formatting.")
        else:
            self.value = float(nominal)
            self.plus = np.abs(float(pos_err))
            self.minus = np.abs(float(neg_err))
        self.maximum = self.value+self.plus
        self.minimum = self.value-self.minus
        self.sign = 1 if self.value >= 0 else -1
        self.is_symmetric = np.isclose(self.plus, self.minus)
    
    def __str__(self):
        return f"{self}"
    
    def __format__(self, fmt_spec):
        if np.isclose(self.plus, self.minus):
            return f"{format(self.value, fmt_spec)} ± {format(self.plus, fmt_spec)}"
        else:
            return "%s (+%s, -%s)" % tuple(format(x, fmt_spec) for x in self.items())
    
    def _repr_latex_(self, float_fmt="f"):
        if np.isclose(self.plus, self.minus):
            return f"${format(self.value, float_fmt)} \pm {format(self.plus, float_fmt)}$"
        else:
            return "$%s^{+%s}_{-%s}$" % tuple(format(x, float_fmt) for x in self.items())
    
    def pdf(self, x):
        """
        Computes and returns the values of the probability distribution function for the specified input.
        """
        x_arr = np.asanyarray(x).astype(float)
        pdf_arr =  np.piecewise(x_arr, [x_arr<self.value, x_arr>=self.value],
                                [lambda x: np.sqrt(2)/np.sqrt(np.pi)/(self.plus+self.minus) * np.exp(-1*(x-self.value)**2 / (2*self.minus**2)),
                                 lambda x: np.sqrt(2)/np.sqrt(np.pi)/(self.plus+self.minus) * np.exp(-1*(x-self.value)**2 / (2*self.plus**2))])
        return pdf_arr.item() if np.isscalar(x) else pdf_arr
    
    def cdf(self, x):
        """
        Computes and returns the values of the cumulative distribution function for the specified input.
        """
        x_arr = np.asanyarray(x).astype(float)
        cdf_arr = np.piecewise(x_arr, [x_arr<self.value, x_arr>=self.value],
                               [lambda x: 0.5*(1 + np_erf((x-self.value)/(self.minus*np.sqrt(2)))),
                                lambda x: 0.5*(1 + np_erf((x-self.value)/(self.plus*np.sqrt(2))))])
        return cdf_arr.item() if np.isscalar(x) else cdf_arr
    
    def pdfplot(self,num_sigma=5,discretization=100,**kwargs):
        """
        Plots the associated PDF over the specified number of sigma, using 2*`discretization` points.
        `**kwargs` are passed on to `matplotlib` for configuration of the resulting plot.
        """
        neg_x = np.linspace(self.value-(num_sigma*self.minus), self.value, discretization)
        pos_x = np.linspace(self.value, self.value+(num_sigma*self.plus), discretization)
        x = np.array(list(neg_x)+list(pos_x))
        pdf = self.pdf(x)
        plt.plot(x,pdf,**kwargs)
        plt.show()
        
    def cdfplot(self,num_sigma=5,discretization=100,**kwargs):
        """
        Plots the associated CDF over the specified number of sigma, using 2*`discretization` points.
        `**kwargs` are passed on to `matplotlib` for configuration of the resulting plot.
        """
        neg_x = np.linspace(self.value-(num_sigma*self.minus), self.value, discretization)
        pos_x = np.linspace(self.value, self.value+(num_sigma*self.plus), discretization)
        x = np.array(list(neg_x)+list(pos_x))
        cdf = self.cdf(x)
        plt.plot(x,cdf,**kwargs)
        plt.show()
        
    def add_error(self, delta, method="quadrature", inplace=False):
        """
        Adds `delta` to an instance's existing error. Possible `method`s are `quadrature`, `straight`, or `split`.
        If `inplace` is `True`, the existing object's errors are modified in place. If it is `False`, a new instance is returned.
        """
        if method=="quadrature":
            new_pos = np.sqrt(self.plus**2 + delta**2)
            new_neg = np.sqrt(self.minus**2 + delta**2)
        elif method=="straight":
            new_pos = self.plus + delta
            new_neg = self.minus + delta
        elif method=="split":
            new_pos = self.plus + delta/2
            new_neg = self.minus + delta/2
        else:
            raise ValueError
        if inplace:
            self.plus = new_pos
            self.minus = new_neg
        else:
            return a_u(self.value,new_pos,new_neg)
    
    def items(self):
        """
        Returns a tuple of `(value,plus,minus)`.
        """
        return (self.value,self.plus,self.minus)
    
    def __int__(self):
        return int(self.value)
    
    def __float__(self):
        return float(self.value)
    
    def __neg__(self):
        return a_u(-self.value,self.minus,self.plus)
        
    def __add__(self,other):
        if isinstance(other,type(self)):
            pass
        else:
            other = a_u(other,0,0)
        result = self.value + other.value
        pos = np.sqrt(self.plus**2 + other.plus**2)
        neg = np.sqrt(self.minus**2 + other.minus**2)
        #print("added",self,"+",other,"=",a_u(result,pos,neg))
        return a_u(result,pos,neg)
    
    def __radd__(self,other):
        if isinstance(other,type(self)):
            pass
        else:
            other = a_u(other,0,0)
        result = self.value + other.value
        pos = np.sqrt(self.plus**2 + other.plus**2)
        neg = np.sqrt(self.minus**2 + other.minus**2)
        #print("added",other,"+",self,"=",a_u(result,pos,neg))
        return a_u(result,pos,neg)
    
    def __sub__(self,other):
        if isinstance(other,type(self)):
            pass
        else:
            other = a_u(other,0,0)
        result = self.value - other.value
        pos = np.sqrt(self.plus**2 + other.minus**2)
        neg = np.sqrt(self.minus**2 + other.plus**2)
        #print("subtracted",other,"from",self,"=",a_u(result,pos,neg))
        return a_u(result,pos,neg)
    
    def __rsub__(self,other):
        if isinstance(other,type(self)):
            pass
        else:
            other = a_u(other,0,0)
        result = other.value - self.value
        pos = np.sqrt(self.minus**2 + other.plus**2)
        neg = np.sqrt(self.plus**2 + other.minus**2)
        #print("subtracted",self,"from",other,"=",a_u(result,pos,neg))
        return a_u(result,pos,neg)
    
    def __mul__(self,other):
        if isinstance(other,type(self)):
            pass
        else:
            other = a_u(other,0,0)
        
        result = self.value * other.value
        pos = np.sqrt((self.plus/self.value)**2 + (other.plus/other.value)**2) * np.abs(result)
        neg = np.sqrt((self.minus/self.value)**2 + (other.minus/other.value)**2) * np.abs(result)
        #print("multiplied",self,"by",other,"=",a_u(result,pos,neg))
        return a_u(result,pos,neg)
    
    def __rmul__(self,other):
        if isinstance(other,type(self)):
            pass
        else:
            other = a_u(other,0,0)
        
        result = self.value * other.value
        pos = np.sqrt((self.plus/self.value)**2 + (other.plus/other.value)**2) * np.abs(result)
        neg = np.sqrt((self.minus/self.value)**2 + (other.minus/other.value)**2) * np.abs(result)
        #print("multiplied",other,"by",self,"=",a_u(result,pos,neg))        
        return a_u(result,pos,neg)
    
    def __truediv__(self,other): # self divided by something
        if isinstance(other,type(self)):
            pass
        else:
            other = a_u(other,0,0)
        result = self.value / other.value
        pos = np.sqrt((self.plus/self.value)**2 + (other.minus/other.value)**2) * np.abs(result)
        neg = np.sqrt((self.minus/self.value)**2 + (other.plus/other.value)**2) * np.abs(result)
        #print("divided",self,"by",other,"=",a_u(result,pos,neg))        
        return a_u(result,pos,neg)
    
    def __rtruediv__(self,other): # something divided by self
        if isinstance(other,type(self)):
            pass
        else:
            other = a_u(other,0,0)
        result = other.value / self.value
        pos = np.sqrt((other.plus/other.value)**2 + (self.minus/self.value)**2) * np.abs(result)
        neg = np.sqrt((other.minus/other.value)**2 + (self.plus/self.value)**2) * np.abs(result)
        #print("divided",other,"by",self,"=",a_u(result,pos,neg))        
        return a_u(result,pos,neg)
    
    def __pow__(self,other): # self to the something power
        if isinstance(other,type(self)):
            pass
        else:
            other = a_u(other,0,0)
        result = self.value**other.value
        pos = np.abs(result)*np.sqrt((self.plus*other.value/self.value)**2 + (other.plus*np.log(self.value))**2)
        neg = np.abs(result)*np.sqrt((self.minus*other.value/self.value)**2 + (other.minus*np.log(self.value))**2)
        #print("raised",self,"to",other,"=",a_u(result,pos,neg))        
        return a_u(result,pos,neg)
    
    def __rpow__(self,other): # something to the self power
        if isinstance(other,type(self)):
            pass
        else:
            other = a_u(other,0,0)
        result = other.value**self.value
        pos = np.abs(result)*np.sqrt((other.plus*self.value/other.value)**2 + (self.plus*np.log(other.value))**2)
        neg = np.abs(result)*np.sqrt((other.minus*self.value/other.value)**2 + (self.minus*np.log(other.value))**2)
        #print("raised",other,"to",self,"=",a_u(result,pos,neg))                
        return a_u(result,pos,neg)
    
    def log10(self):
        result = np.log10(self.value)
        pos = self.plus/(self.value*np.log(10))
        neg = self.minus/(self.value*np.log(10))
        #print("logged",self,"=",a_u(result,pos,neg))
        return a_u(result,pos,neg)
    
    def log(self):
        result = np.log(self.value)
        pos = self.plus/self.value
        neg = self.minus/self.value
        return a_u(result,pos,neg)

    def exp(self):
        return np.exp(1)**self

    def sqrt(self):
        return self**0.5
    
    def __eq__(self,other):
        if isinstance(other,type(self)):
            pass
        else:
            other = a_u(other,0,0)
        return self.value == other.value and self.plus == other.plus and self.minus == other.minus
    
    def __gt__(self,other):
        if isinstance(other,type(self)):
            pass
        else:
            other = a_u(other,0,0)
        return self.value > other.value
    
    def __lt__(self,other):
        if isinstance(other,type(self)):
            pass
        else:
            other = a_u(other,0,0)
        return self.value < other.value
    
    def __lshift__(self,other): # overloaded <<; definitively less than
        if isinstance(other,type(self)):
            pass
        else:
            other = a_u(other,0,0)
        return self.maximum < other.minimum
    
    def __rshift__(self,other): # overloaded >>; definitively greater than
        if isinstance(other,type(self)):
            pass
        else:
            other = a_u(other,0,0)
        return self.minimum > other.maximum
    
    def __le__(self,other):
        if isinstance(other,type(self)):
            pass
        else:
            other = a_u(other,0,0)
        return self.value <= other.value
    
    def __ge__(self,other):
        if isinstance(other,type(self)):
            pass
        else:
            other = a_u(other,0,0)
        return self.value >= other.value
    
    def conjugate(self):
        return self.value
    
    def __isfinite__(self):
        return all(np.isfinite(self.items()))
    
    def isna(self):
        """
        `pandas`-style NaN checker. Returns True if value is NaN or None, and False if neither.        
        """
        return (np.isnan(self.value) or (self.value is None))
    
    def notna(self):
        """
        Inverse of `isna()`. Returns True if value is neither NaN nor None, and False if it is.
        """
        return not (np.isnan(self.value) or (self.value is None))

class AsymmetricUncertainty(a_u): # alias for legacy namespace support
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn("class AsymmetricUncertainty has been renamed. Use a_u instead.",
                      DeprecationWarning, stacklevel=2)

class UncertaintyArray(list):
    """
    Class for representing an array of `a_u` objects.
    Mostly provides utilities for slicing and accessing various attributes.
    """
    def refresh(self):
        for i in range(len(self)):
            try:
                self[i].value
                self[i].plus
                self[i].minus
            except AttributeError:
                self[i] = a_u(self[i],0,0)
                
        self.as_numpy = np.array(self.as_list)
        self.flattened = self.as_numpy.flatten()
        self.shape = self.as_numpy.shape
        self.ndim = self.as_numpy.ndim

        self.minus = [v.minus for v in self.as_list]
        self.plus = [v.plus for v in self.as_list]
        self.values = [v.value for v in self.as_list]
    
    def __init__(self,array=[]):

        self.as_list = list(array)
        self.refresh()

    def __len__(self):
        return len(self.as_list)

    def __iter__(self):
        return iter(self.as_list)

    def __getitem__(self, key):
        return self.as_list[key]

    def __setitem__(self, key, val):
        self.as_list[key] = val

    def __str__(self):
        return str([str(entry) for entry in self.as_list])

    def __repr__(self):
        return str(self)

    def __contains__(self, item):
        return item in self.as_list

    def append(self,entry):
        self.as_list.append(entry)
        self.refresh()
    
    def pdf(self,x):
        return np.sum([entry.pdf(x) for entry in self], axis=0)

def pos_errors(array):
    """
    Stand-alone function to return an array of the positive errors of an array of `a_u` objects.
    Functional equivalent to `UncertaintyArray(array).plus`.
    """
    return [v.plus for v in array]

def neg_errors(array):
    """
    Stand-alone function to return an array of the negative errors of an array of `a_u` objects.
    Functional equivalent to `UncertaintyArray(array).minus`.
    """
    return [v.minus for v in array]

#grb_list = pd.read_table("https://www.swift.ac.uk/xrt_curves/grb.list",
                         #sep=" |\t",header=None,engine="python",
                         #names=["_","GRB","Trigger Number"]).drop("_",axis=1)



'''
def XRT_lightcurve(burst_id,lookuptable=grb_list):
    """
    Function for retrieving X-ray observations (flux values) for a given gamma-ray burst.
    
    Author: Caden Gobat, George Washington University

    Parameters
    ----------
    burst_id : string
        GRB ID/name in the form YYMMDDx
    lookuptable : pandas DataFrame
        the reference table to get the Trigger Number

    Returns
    -------
    fluxdata : pandas DataFrame
        table containing the time series fluxes in the XRT 0.3-10 keV band.
        Columns are Time, Tpos, Tneg, Flux, Fluxpos, and Fluxneg.

    Raises
    ------
    IndexError
        if the table could not be retrieved
    
    """
    
    trigger = lookuptable.loc[lookuptable["GRB"] == burst_id, "Trigger Number"]
    print(trigger)
    
    lightcurveURL = f"https://www.swift.ac.uk/xrt_curves/{int(trigger):0>8}/flux_incbad.qdp"
    fluxdata = pd.DataFrame(columns=['Time', 'Time_perr', 'Time_nerr', 'Flux', 'Flux_perr', 'Flux_nerr'])
    i = 0
    while True:
        try:
            current = Table.read(lightcurveURL,format="ascii.qdp",table_id=i,names=["Time","Flux"]).to_pandas()
            fluxdata = pd.concat([fluxdata,current], ignore_index=True)
            i += 1
        except:
            if i>0:
                break
            else:
                raise IndexError
    fluxdata.columns = ["Time","Tpos","Tneg","Flux","Fluxpos","Fluxneg"]
    fluxdata = fluxdata.apply(pd.to_numeric)
    fluxdata["GRB"] = [burst_id]*len(fluxdata)
    #print("Retrieved",burst_id)
    return fluxdata


def get_columnDensity(burst_id,lookuptable=grb_list):
    """
    Function for retrieving the neutral hydrogen column density in the direction of a given gamma-ray burst.
    
    Author: Caden Gobat, George Washington University

    Parameters
    ----------
    burst_id : string
        GRB ID/name in the form YYMMDDx
    lookuptable : pandas DataFrame
        the reference table to get the Trigger Number

    Returns
    -------
    N_H : a_u
        value of the neutral hydrogen column density in the form (value (pos_err, neg_err)) [units of cm^-2]

    Raises
    ------
    
    """
    try:
        trigger = int(lookuptable.loc[lookuptable["GRB"] == burst_id, "Trigger Number"])
    except ValueError:
        trigger = int(grb_list.loc[grb_list["GRB"] == burst_id, "Trigger Number"])
    spectrumURL = f"https://www.swift.ac.uk/xrt_spectra/{trigger:0>8}/"
    
    page = requests.get(spectrumURL)
    soup = bs(page.content,"html.parser")
    assert page.status_code != 404, "404 Error."
    assert len(soup.findAll("table",{"summary":"Model fitted to interval0 data"}))>0, "No tables found."

    tables = {}
    for section in soup.findAll("div",{"class":"fitres"}):
        mode = section.find("h3").text[:2]
        current = section.find("table",{"summary":"Model fitted to interval0 data"})
        if current is not None:
            tables[mode] = current
    if "PC" in tables.keys():
        table = tables["PC"] # use PC mode info if available
        used_mode = "PC"
    else:
        table = tables["WT"] # if not, use WT mode and flag as such
        used_mode = "WT"

    for entry in table.children:
        label = entry.find("th")
        if isinstance(label,int):
            pass
        elif label.text == "NH (intrinsic)":
            NH_str = entry.find("td").text
            
    mult = NH_str.index("×")
    power = NH_str[mult+4:mult+6]
    vals = NH_str[:mult-1]+NH_str[mult+6:]
    nominal, plus, minus = vals.split()[0], vals[vals.index("+")+1:vals.index(",")], vals[vals.index("-")+1:vals.index(")")]
    N_H = a_u(float(nominal)*10**int(power), float(plus)*10**int(power), float(minus)*10**int(power))
    
    return N_H, used_mode

def get_photonIndex(burst_id,lookuptable=grb_list):
    """
    Function for retrieving the X-ray spectral index for a given gamma-ray burst.
    
    Author: Caden Gobat, George Washington University

    Parameters
    ----------
    burst_id : string
        GRB ID/name in the form YYMMDDx
    lookuptable : pandas DataFrame
        the reference table to get the Trigger Number

    Returns
    -------
    gamma : a_u
        value of the spectral index in the form (value (pos_err, neg_err))

    Raises
    ------
    
    """
    try:
        trigger = int(lookuptable.loc[lookuptable["GRB"] == burst_id, "Trigger Number"])
    except ValueError:
        trigger = int(grb_list.loc[grb_list["GRB"] == burst_id, "Trigger Number"])
    spectrumURL = f"https://www.swift.ac.uk/xrt_spectra/{trigger:0>8}/"
    
    page = requests.get(spectrumURL)
    soup = bs(page.content,"html.parser")
    assert page.status_code != 404, "404 Error."
    assert len(soup.findAll("table",{"summary":"Model fitted to interval0 data"}))>0, "No tables found."

    tables = {}
    for section in soup.findAll("div",{"class":"fitres"}):
        mode = section.find("h3").text[:2]
        current = section.find("table",{"summary":"Model fitted to interval0 data"})
        if current is not None:
            tables[mode] = current
    if "PC" in tables.keys():
        table = tables["PC"] # use PC mode info if available
        used_mode = "PC"
    else:
        table = tables["WT"] # if not, use WT mode and flag as such
        used_mode = "WT"

    for entry in table.children:
        label = entry.find("th")
        if isinstance(label,int):
            pass
        elif label.text == "Photon index":
            photon_index = entry.find("td").text
            
    (Gamma, Gammapos, Gammaneg) = (float(num) for num in "".join([char for char in photon_index if char not in "[]()+-,"]).split())
    gamma = a_u(Gamma, Gammapos, Gammaneg)
    return gamma, used_mode

def get_temporalIndex(burst_id,query_time,lookuptable=grb_list):
    """
    Function for retrieving the temporal index (power-law slope) of a gamma-ray burst at a given time.
    
    Author: Caden Gobat, George Washington University

    Parameters
    ----------
    burst_id : string
        GRB ID/name in the form YYMMDDx
    query_time : numeric
        time (in seconds) at which to retrieve the temporal index
    lookuptable : pandas DataFrame
        the reference table to get the Trigger Number

    Returns
    -------
    alpha : a_u
        value of the temporal index at the specified time in the form (value (pos_err, neg_err)) [units of cm^-2]

    Raises
    ------
    
    """
    trigger = lookuptable.loc[lookuptable["GRB"] == burst_id, "Trigger Number"]
    livecatURL = f"https://www.swift.ac.uk/xrt_live_cat/{int(trigger):0>8}/"
    
    livecat_tables = pd.read_html(livecatURL)
    slopes_table = livecat_tables[2]
    assert len(slopes_table.columns)==2
    
    breaktimes = [slopes_table.iloc[i,1][:-2] for i in slopes_table.index if "T" in slopes_table.iloc[i,0]]
    for i,time in enumerate(breaktimes):
        if "×" in time:
            vals,power = time.split('×')
            power = power.split()[0][2:]
            nominal, plus, minus = vals.split()[0], vals[vals.index("+")+1:vals.index(",")], vals[vals.index("-")+1:vals.index(")")]
            time = a_u(float(nominal), float(plus), float(minus)) * 10**int(power)
        else:
            time = a_u(time)
        breaktimes[i] = time
    breaktimes += [np.inf]
    
    alphas = [a_u(slopes_table.iloc[i,1]) for i in slopes_table.index if "α" in slopes_table.iloc[i,0]]
    
    assert len(breaktimes)+len(alphas) == len(slopes_table.index)+1, "Mismatch error in parsing table rows"
    
    for i,time in enumerate(breaktimes):
        if query_time < time:
            alpha = alphas[i]
            break
        
    return alpha

'''


new_optical = pd.read_excel("grb_data.xlsx")

for col in ["GRB","TriggerNumber","Observatory","Instrument","Source","E(B-V)"]:
    for i in new_optical.index:
        if pd.isna(new_optical.loc[i,col]):
            new_optical.loc[i,col] = new_optical.loc[i-1,col] # deal with merged Excel cells
new_optical["Magnitude"] = pd.to_numeric(new_optical["Magnitude"], errors="coerce")
new_optical.dropna(subset=["Magnitude"],inplace=True)




UVOT_conversions = {'V': -0.01, 'B': -0.13, 'U': 1.02, 'UVW1': 1.51, 'UVM2': 1.69, 'UVW2': 1.73, 'White': 0.8}

for i in new_optical.loc[new_optical["Observatory"]=="Swift"].index:
    filt = new_optical.loc[i,"Filter"]
    if filt in UVOT_conversions.keys() or pd.isna(filt):
        pass
    elif filt.upper() in UVOT_conversions.keys():
        new_optical.loc[i,"Filter"] = filt.upper()
    elif 'wh' in filt.lower():
        new_optical.loc[i,"Filter"] = "White"
    elif filt.lower().endswith("_fc"):
        new_optical.loc[i,"Filter"] = filt[:-3].title()
    elif 'm2' in filt.lower():
        new_optical.loc[i,"Filter"] = "UVM2"
    elif 'w1' in filt.lower():
        new_optical.loc[i,"Filter"] = "UVW1"
    elif 'w2' in filt.lower():
        new_optical.loc[i,"Filter"] = "UVW2"
    
    new_optical.loc[i,"Magnitude"] += UVOT_conversions[new_optical.loc[i,"Filter"]]
    
new_optical["mag_w_err"] = [a_u(mag,err,err) if isinstance(err,(float,int)) else a_u(mag,np.inf,0) for mag,err in new_optical[["Magnitude","Mag error"]].values]
    


RbTable = pd.read_csv("Rb.csv") # Table 6 from Schlafly & Finkbeiner (2011)
RbTable.drop([37,55,61,73],axis=0,inplace=True) # smoothing

Rb = interpolate.interp1d(RbTable["lambda_eff"],RbTable["R_b"],fill_value="extrapolate") # function that takes a wavelength [Ang] and returns the corresponding R_b value


new_optical["Extinction"] = Rb(new_optical["λ_eff"])*new_optical["E(B-V)"]

new_optical["Flux (Jy)"] = 3631*10**(-(new_optical["mag_w_err"]-new_optical["Extinction"])/2.5) # AB mag = 0 at f_nu = 3631 Jy

#print(new_optical)


GRBs = pd.read_excel("grb_data.xlsx")


grb_list = pd.read_table("https://www.swift.ac.uk/xrt_curves/grb.list",
                         sep=" |\t",header=None,engine="python",
                         names=["_","GRB","Trigger Number"]).drop("_",axis=1)

grb_list.drop([1723], inplace = True)


def XRT_lightcurve(burst_id,lookuptable=grb_list):
    """
    Function for retrieving X-ray observations (flux values) for a given gamma-ray burst.
    
    Author: Caden Gobat, George Washington University

    Parameters
    ----------
    burst_id : string
        GRB ID/name in the form YYMMDDx
    lookuptable : pandas DataFrame
        the reference table to get the Trigger Number

    Returns
    -------
    fluxdata : pandas DataFrame
        table containing the time series fluxes in the XRT 0.3-10 keV band.
        Columns are Time, Tpos, Tneg, Flux, Fluxpos, and Fluxneg.

    Raises
    ------
    IndexError
        if the table could not be retrieved
    
    """
    trigger = lookuptable.loc[lookuptable["GRB"] == burst_id, "Trigger Number"]
    #print(trigger)
    
    
    lightcurveURL = f"https://www.swift.ac.uk/xrt_curves/{int(trigger):0>8}/flux_incbad.qdp"
    fluxdata = pd.DataFrame(columns=['Time', 'Time_perr', 'Time_nerr', 'Flux', 'Flux_perr', 'Flux_nerr'])
    i = 0
    while True:
        try:
            current = Table.read(lightcurveURL,format="ascii.qdp",table_id=i,names=["Time","Flux"]).to_pandas()
            fluxdata = pd.concat([fluxdata,current], ignore_index=True)
            i += 1
        except:
            if i>0:
                break
            else:
                raise IndexError
    fluxdata.columns = ["Time","Tpos","Tneg","Flux","Fluxpos","Fluxneg"]
    fluxdata = fluxdata.apply(pd.to_numeric)
    fluxdata["GRB"] = [burst_id]*len(fluxdata)
    #print("Retrieved",burst_id)
    return fluxdata

def get_columnDensity(burst_id,lookuptable=grb_list):
    """
    Function for retrieving the neutral hydrogen column density in the direction of a given gamma-ray burst.
    
    Author: Caden Gobat, George Washington University

    Parameters
    ----------
    burst_id : string
        GRB ID/name in the form YYMMDDx
    lookuptable : pandas DataFrame
        the reference table to get the Trigger Number

    Returns
    -------
    N_H : a_u
        value of the neutral hydrogen column density in the form (value (pos_err, neg_err)) [units of cm^-2]

    Raises
    ------
    
    """
    try:
        trigger = int(lookuptable.loc[lookuptable["GRB"] == burst_id, "Trigger Number"])
    except ValueError:
        trigger = int(grb_list.loc[grb_list["GRB"] == burst_id, "Trigger Number"])
    spectrumURL = f"https://www.swift.ac.uk/xrt_spectra/{trigger:0>8}/"
    
    page = requests.get(spectrumURL)
    soup = bs(page.content,"html.parser")
    assert page.status_code != 404, "404 Error."
    assert len(soup.findAll("table",{"summary":"Model fitted to interval0 data"}))>0, "No tables found."

    tables = {}
    for section in soup.findAll("div",{"class":"fitres"}):
        mode = section.find("h3").text[:2]
        current = section.find("table",{"summary":"Model fitted to interval0 data"})
        if current is not None:
            tables[mode] = current
    if "PC" in tables.keys():
        table = tables["PC"] # use PC mode info if available
        used_mode = "PC"
    else:
        table = tables["WT"] # if not, use WT mode and flag as such
        used_mode = "WT"

    for entry in table.children:
        label = entry.find("th")
        if isinstance(label,int):
            pass
        elif label.text == "NH (intrinsic)":
            NH_str = entry.find("td").text
            
    mult = NH_str.index("×")
    power = NH_str[mult+4:mult+6]
    vals = NH_str[:mult-1]+NH_str[mult+6:]
    nominal, plus, minus = vals.split()[0], vals[vals.index("+")+1:vals.index(",")], vals[vals.index("-")+1:vals.index(")")]
    N_H = a_u(float(nominal)*10**int(power), float(plus)*10**int(power), float(minus)*10**int(power))
    
    return N_H, used_mode

def get_photonIndex(burst_id,lookuptable=grb_list):
    """
    Function for retrieving the X-ray spectral index for a given gamma-ray burst.
    
    Author: Caden Gobat, George Washington University

    Parameters
    ----------
    burst_id : string
        GRB ID/name in the form YYMMDDx
    lookuptable : pandas DataFrame
        the reference table to get the Trigger Number

    Returns
    -------
    gamma : a_u
        value of the spectral index in the form (value (pos_err, neg_err))

    Raises
    ------
    
    """
    try:
        trigger = int(lookuptable.loc[lookuptable["GRB"] == burst_id, "Trigger Number"])
    except ValueError:
        trigger = int(grb_list.loc[grb_list["GRB"] == burst_id, "Trigger Number"])
    spectrumURL = f"https://www.swift.ac.uk/xrt_spectra/{trigger:0>8}/"
    
    page = requests.get(spectrumURL)
    soup = bs(page.content,"html.parser")
    assert page.status_code != 404, "404 Error."
    assert len(soup.findAll("table",{"summary":"Model fitted to interval0 data"}))>0, "No tables found."

    tables = {}
    for section in soup.findAll("div",{"class":"fitres"}):
        mode = section.find("h3").text[:2]
        current = section.find("table",{"summary":"Model fitted to interval0 data"})
        if current is not None:
            tables[mode] = current
    if "PC" in tables.keys():
        table = tables["PC"] # use PC mode info if available
        used_mode = "PC"
    else:
        table = tables["WT"] # if not, use WT mode and flag as such
        used_mode = "WT"

    for entry in table.children:
        label = entry.find("th")
        if isinstance(label,int):
            pass
        elif label.text == "Photon index":
            photon_index = entry.find("td").text
            
    (Gamma, Gammapos, Gammaneg) = (float(num) for num in "".join([char for char in photon_index if char not in "[]()+-,"]).split())
    gamma = a_u(Gamma, Gammapos, Gammaneg)
    return gamma, used_mode

def get_temporalIndex(burst_id,query_time,lookuptable=grb_list):
    """
    Function for retrieving the temporal index (power-law slope) of a gamma-ray burst at a given time.
    
    Author: Caden Gobat, George Washington University

    Parameters
    ----------
    burst_id : string
        GRB ID/name in the form YYMMDDx
    query_time : numeric
        time (in seconds) at which to retrieve the temporal index
    lookuptable : pandas DataFrame
        the reference table to get the Trigger Number

    Returns
    -------
    alpha : a_u
        value of the temporal index at the specified time in the form (value (pos_err, neg_err)) [units of cm^-2]

    Raises
    ------
    
    """
    trigger = lookuptable.loc[lookuptable["GRB"] == burst_id, "Trigger Number"]
    livecatURL = f"https://www.swift.ac.uk/xrt_live_cat/{int(trigger):0>8}/"
    
    livecat_tables = pd.read_html(livecatURL)
    slopes_table = livecat_tables[2]
    assert len(slopes_table.columns)==2
    
    breaktimes = [slopes_table.iloc[i,1][:-2] for i in slopes_table.index if "T" in slopes_table.iloc[i,0]]
    for i,time in enumerate(breaktimes):
        if "×" in time:
            vals,power = time.split('×')
            power = power.split()[0][2:]
            nominal, plus, minus = vals.split()[0], vals[vals.index("+")+1:vals.index(",")], vals[vals.index("-")+1:vals.index(")")]
            time = a_u(float(nominal), float(plus), float(minus)) * 10**int(power)
        else:
            time = a_u(time)
        breaktimes[i] = time
    breaktimes += [np.inf]
    
    alphas = [a_u(slopes_table.iloc[i,1]) for i in slopes_table.index if "α" in slopes_table.iloc[i,0]]
    
    assert len(breaktimes)+len(alphas) == len(slopes_table.index)+1, "Mismatch error in parsing table rows"
    
    for i,time in enumerate(breaktimes):
        if query_time < time:
            alpha = alphas[i]
            break
        
    return alpha


def effective_wavelength(filter_response, show_plot=False): # pass a dataframe with columns Wavelength (in Ang), Transmission (in %)
    filter_response.sort_values(by="Wavelength",inplace=True)

    vega_spec = pd.read_table("http://svo2.cab.inta-csic.es/svo/theory/fps3/morefiles/vega.dat",
                              delimiter=" ",header=None,names=["Wavelength","Flux"])
    vega_func = interpolate.interp1d(vega_spec["Wavelength"],vega_spec["Flux"],
                                     bounds_error=False,fill_value=0)
    λ = filter_response["Wavelength"]
    T = filter_response["Transmission"]
    Vg = vega_func(λ)

    numerator = integrate.trapz(y=λ*T*Vg, x=λ) # ∫ λ T(λ) Vg(λ) dλ
    denominator = integrate.trapz(y=T*Vg, x=λ) # ∫ T(λ) Vg(λ) dλ
    
    lambda_eff = numerator/denominator
    
    if show_plot:
        fig,ax = plt.subplots()
        ax.plot(vega_spec.Wavelength,vega_spec.Flux,"b",label="Vega spectrum")
        ax.set_yscale("log")
        ax.set_xlim(filter_response["Wavelength"].min(), filter_response["Wavelength"].max())
        ax.set_ylabel("Flux [erg/cm$^2$/s/Ang]")
        ax.set_xlabel(r"$\lambda$ [Ang]")
        ax2 = ax.twinx()
        ax2.plot(λ,T,color="dimgrey",linestyle=":",label="Filter response")
        ax2.set_ylabel("Transmission [%]")
        ax2.axvline(lambda_eff,color="g",linestyle="--",label=r"$\lambda_\mathrm{eff}$")
        ax2.set_ylim(0,100)
        fig.legend(loc="upper right")
        plt.show()

    return lambda_eff

def lightcurve(grb, band="optical", xlimits=False, ylimits=False, **kwargs):
    """Intended for use in an environment where DataFrames called `xrt_data` and `new_optical`
    already exist and contain afterglow optical/X-ray flux data over time, respectively. This
    function uses that information to plot the light curve of the burst in a band of the user's
    choice."""

    global xrt_data
    global new_optical
    if band == "xray":
        subset = xrt_data.loc[xrt_data["GRB"]==grb]
        neg_err = [0.4*flux.value if np.isinf(flux.minus) else flux.minus for flux in subset["SpecFlux"]]
        pos_err = [0.4*flux.value if np.isinf(flux.plus) else flux.plus for flux in subset["SpecFlux"]]
        plt.errorbar(subset.Time,[flux.value for flux in subset.SpecFlux],
                     xerr=np.array(subset.Tneg,subset.Tpos).T, yerr=np.array((neg_err,pos_err)),
                     linestyle="", capthick=0, **kwargs)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(f"Swift XRT Lightcurve for GRB {grb}")
        plt.xlabel("Time since trigger [s]")
        plt.ylabel("Flux (~1 keV) [Jy]")
        plt.grid(linestyle="--")
        if xlimits:
            plt.xlim(xlimits)
        if ylimits:
            plt.ylim(ylimits)
    else:
        if band == "optical":
            subset = new_optical.loc[(new_optical["GRB"]==grb) & (new_optical["λ_eff"]>=3000) & (new_optical["λ_eff"]<=8000)]
        elif band == "UV":
            subset = new_optical.loc[(new_optical["GRB"]==grb) & (new_optical["λ_eff"]<3000)]
        elif band == "IR":
            subset = new_optical.loc[(new_optical["GRB"]==grb) & (new_optical["λ_eff"]>8000)]
        else:
            print("Unrecognized band selected")
            return None
        
        fig,ax = plt.subplots()
        neg_err = [0.4*flux.value if np.isinf(flux.minus) else flux.minus for flux in subset["Flux (Jy)"]]
        pos_err = [0.4*flux.value if np.isinf(flux.plus) else flux.plus for flux in subset["Flux (Jy)"]]
        ax.errorbar(subset["Time (s)"],[flux.value for flux in subset["Flux (Jy)"]], marker=".", linestyle="", capthick=0,
                    yerr=np.array((neg_err, [point.plus for point in subset["Flux (Jy)"]])),
                    uplims=[np.isinf(point.minus) for point in subset["Flux (Jy)"]],
                    lolims=[np.isinf(point.plus) for point in subset["Flux (Jy)"]], **kwargs)
        ax.grid(linestyle="--")
        ax.set(xscale="log",yscale="log",xlabel="Time (s)",ylabel=f"Flux ({band}) [Jy]")
        if xlimits:
            ax.set(xlim=xlimits)
        if ylimits:
            ax.set(ylim=ylimits)
    plt.show()

def simulate_spectrum(idx):
    """Intended for use in an environment where a DataFrame called `results` already
    exists and contains matched optical/X-ray data points. This function uses that
    information to plot the broadband "spectrum" at a selected point."""

    global results
    inter_freqs = np.linspace(results.loc[idx,"nu_o"],0.3/4.135667696e-18,100)
    xray_freqs = np.linspace(0.3,10,100)/4.135667696e-18
    
    ox_spec = inter_freqs**(-results.loc[idx,"B_ox"].value)
    ox_spec *= (results.loc[idx,"F_o"]/ox_spec[0])
    x_spec = xray_freqs**(-results.loc[idx,"B_x"])
    x_spec *= (ox_spec[-1]/x_spec[0])
    plt.plot(inter_freqs,ox_spec,label=r"$\beta_{ox}=%f$"%(-results.loc[idx,"B_ox"]))
    plt.plot(xray_freqs,x_spec,label=r"$\beta_x=%f$"%(-results.loc[idx,"B_x"]))
    plt.vlines([10**np.mean((np.log10(0.3),np.log10(10))) / 4.135667696e-18],0,
                10**np.log10((x_spec.max()+x_spec.min())/2),"k",linestyle="--")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    #plt.gca().set_yticklabels([])
    plt.show()
    
log_mean = lambda a, b : 10**np.mean((np.log10(a), np.log10(b)))
log_slope = lambda point1, point2 : np.log10(point2[1]/point1[1])/np.log10(point2[0]/point1[0])
Hz_per_keV = 241797944177033445


alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
alpha += alpha.lower()
numeric = ".0123456789"


#GRBs["Trigger Number"] = GRBs["Trigger Number"].apply(lambda tn: int(val) if (val:=str(tn).strip("*")).isnumeric() else val)

BetaXData = pd.read_csv("BetaXData.csv", header=None, names=["GRB", "Beta_X","Beta_X_pos","Beta_X_neg"])
#BetaXData["GRB"] = [entry.split("-")[-1] for entry in BetaXData["GRB"]]

#print(GRBs)

xrt_data = pd.DataFrame(columns=['GRB', 'Time', 'Tpos', 'Tneg', 'Flux', 'Fluxpos', 'Fluxneg'])
for i,row in GRBs.iterrows():
    GRB_ID = row["GRB"]
    print(GRB_ID)
    mode = None
    try:
        Gamma,mode = get_photonIndex(GRB_ID,grb_list)
        GRBs.loc[i,"Beta_X"] = Gamma.value - 1
        GRBs.loc[i,"Beta_X_neg"] = Gamma.minus
        GRBs.loc[i,"Beta_X_pos"] = Gamma.plus
        print("index ✓",end=", ")
    except:
        print("index ✗",end=", ")
    try:
        fluxdata = XRT_lightcurve(GRB_ID,grb_list)
        xrt_data = pd.concat([xrt_data, fluxdata],ignore_index=True)
        print("lightcurve ✓",end=" ")
    except:
        print("lightcurve ✗",end=" ")
    if mode=="WT":
        print("(used WT spectrum)")
    else:
        print()

# account for upper limits
xrt_data.loc[xrt_data["Fluxneg"]==0, "Fluxneg"] = np.inf

xrt_data.to_csv("Swift_XRT_lightcurves.csv",index=False)



#xrt_data = pd.read_csv("Downloads/Swift_XRT_lightcurves.csv")


'''
for i,row in xrt_data.iterrows():
    log_mean_energy = 10**np.mean((np.log10(0.3),np.log10(10))) # halfway between the endpoints in log space
    spectral_flux = a_u(result,pos_err,neg_err) # erg/s/cm^2/keV
    spectral_flux *= 1e23/241797944177033445 # convert to Jy
    xrt_data.loc[i,"SpecFlux"] = spectral_flux
#print(xrt_data)
'''


class custom_iter: # custom iterator class that allows for retrieval of current element w/out advancing
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(iterable)
        self.current = None
    def __next__(self):
        try:
            self.current = next(self.iterator)
        except StopIteration:
            self.current = None
        finally:
            return self.current
    def __len__(self):
        return len(self.iterable)
    
def split_filters(string):
    UVOT_filters = ["B","U","UVW1","UVM2","UVW2","White"]
    name_idxs = custom_iter([string.index(i) for i in UVOT_filters if i in string])
    split_list = [string[name_idxs.current:next(name_idxs)] for i in range(len(UVOT_filters))]
    split_list = [item for item in split_list if len(item)>0]
    return np.unique(split_list).tolist()


xrt_data[["Tpos", "Tneg", "Fluxpos", "Fluxneg"]] = np.abs(xrt_data[["Tpos", "Tneg", "Fluxpos", "Fluxneg"]])
xrt_data.sort_values(by=["GRB","Time"], ascending=[False,True], inplace=True)


# max_dt (allowable % time difference) is set before this notebook is run

error_B_ox  = True # whether to incorporate Δβₒₓ (due to temporal separation) in errors
restrictive = True # whether to use a permissive inequality cut (w.r.t. error bars) for determining darkness (<< vs. <)
max_dt      = 0.2 # maximum allowable temporal separation between optical and X-ray data points (%)

for i,row in xrt_data.iterrows():
    grb_id = row["GRB"]
    if grb_id not in GRBs["GRB"].values:
        continue
    beta = a_u(float(GRBs.loc[GRBs["GRB"]==grb_id,"Beta_X"]),
               float(GRBs.loc[GRBs["GRB"]==grb_id,"Beta_X_pos"]/1.645), # 90% conf to 1-sigma
               float(GRBs.loc[GRBs["GRB"]==grb_id,"Beta_X_neg"]/1.645)) # 90% conf to 1-sigma
    B = beta.value
    Fx = a_u(row["Flux"], row["Fluxpos"], np.abs(row["Fluxneg"]))
    if B == 1:
        integral = np.log(10) - np.log(0.3)
    else:
        integral = (10**(1-B) - 0.3**(1-B))/(1-B)

    log_mean_energy = 10**np.mean((np.log10(0.3),np.log10(10))) # halfway between the endpoints in log space

    dfdF = (log_mean_energy**(-B))/integral
    dfdB = -np.log(log_mean_energy)*log_mean_energy**(-B)*Fx.value/integral #- log_mean_energy**(-B)*(1-B)*Fx.value*(np.log(0.3)*0.3**(1-B) - np.log(10)*10**(1-B))/((10**(1-B)-0.3**(1-B))**2) - log_mean_energy**(-B)*Fx.value/(10**(1-B)-0.3**(1-B))
    pos_err = np.sqrt(dfdF**2*Fx.plus**2 + dfdB**2*beta.plus**2)
    neg_err = np.sqrt(dfdF**2*Fx.minus**2 + dfdB**2*beta.minus**2)
    result = Fx.value*log_mean_energy**(-B)/integral
    spectral_flux = a_u(result,pos_err,neg_err) # erg/s/cm^2/keV
    spectral_flux *= 1e23/241797944177033445 # convert to Jy
    xrt_data.loc[i,"SpecFlux"] = spectral_flux

#print(new_optical)


log_mean_energy = 10**np.mean((np.log10(0.3),np.log10(10)))
nu_x = a_u(log_mean_energy,10-log_mean_energy,log_mean_energy-0.3) * 241797944177033445 # xray frequency [Hz]
results = pd.DataFrame(columns=["GRB","t_o","dt%","nu_o","F_o","nu_x","F_x","B_ox"])
for i_o in new_optical.index: # for each optical data point
    t_o = float(new_optical.loc[i_o,"Time (s)"]) # optical observation time
    F_o = new_optical.loc[i_o,"Flux (Jy)"] # optical flux
    nu_o = 299792458/float(new_optical.loc[i_o,"λ_eff"]/1e10) # optical frequency [Hz]
    for i_x in xrt_data[xrt_data["GRB"]==new_optical.loc[i_o,"GRB"]].index: # for each x-ray data point on this GRB
        t_x = float(xrt_data.loc[i_x,"Time"]) # x-ray observation time
        dt = np.abs(t_o-t_x)/t_x # time difference
        if dt <= max_dt: # if time separation is allowable
            F_x = xrt_data.loc[i_x,"SpecFlux"]
            Beta_ox = -np.log10(F_x/F_o)/np.log10(nu_x/nu_o)
            if pd.notna(Beta_ox.value):
                match_info = {"GRB":new_optical.loc[i_o,"GRB"], "t_o":t_o, "dt%":dt,
                              "nu_o":nu_o, "F_o":F_o, "nu_x":nu_x, "F_x":F_x, "B_ox":Beta_ox}
                results = pd.concat([results, pd.DataFrame([match_info])], ignore_index=True)
        else: # if data points don't match
            pass


delta_B_ox = lambda dt, alpha=1 : np.abs(alpha*np.log10(1+dt)) # Fitzpatrick Eq. 41

if error_B_ox: # flag set before running
    results["α"] = np.nan
    for (grb,t_o),matches in results.groupby(["GRB","t_o"]):
        i = matches.index[0]
        a = results.loc[i,"α"]
        if pd.notna(a):
            continue
        try:
            a = get_temporalIndex(grb, t_o, GRBs)
            results.loc[matches.index,"α"] = a.value
        except:
            a = 1
            results.loc[matches.index,"α"] = a
        #print("", end="\rWorking on GRB {}...".format(grb+" "*(7-len(grb))))
    #print("\rFinished Δβₒₓ."+" "*11)
    results["B_ox_w_err"] = [match["B_ox"].add_error(delta_B_ox(match["dt%"], match["α"])) for _, match in results.iterrows()]
    B_ox_name = "B_ox_w_err"
    
else:
    B_ox_name = "B_ox"
    
#print("Using",B_ox_name)

results = results.merge(GRBs[["GRB","Beta_X","Beta_X_pos","Beta_X_neg"]],
                        on="GRB",how="left") # match by GRB ID
results["B_x"] = [a_u(*results.loc[i,["Beta_X","Beta_X_pos","Beta_X_neg"]]) for i in results.index] # construct objects
results.drop(['Beta_X', 'Beta_X_pos', 'Beta_X_neg'],axis=1,inplace=True) # discard superfluous columns

# add flag columns for darkness by both methods
for i in results.index:
    if restrictive: # flag set before running. '<<' operator is overloaded for use with a_u class
        results.loc[i, "Jak_dark"] = (results.loc[i, B_ox_name] << 0.5)
        results.loc[i, "vdH_dark"] = (results.loc[i, B_ox_name] << results.loc[i,"B_x"]-0.5)
    else:
        results.loc[i, "Jak_dark"] = (results.loc[i, B_ox_name] < 0.5)
        results.loc[i, "vdH_dark"] = (results.loc[i, B_ox_name] < results.loc[i,"B_x"]-0.5)

# results.sort_values(by=["GRB","t_o"],ascending=[False,True])

dark_matches = results.loc[(results["Jak_dark"] | results["vdH_dark"]), :].sort_values(by=["GRB","dt%"], ascending=[False,True])

# only accept the closest temporal match for each optical data point
close_times = results.copy()
for grb_id in close_times["GRB"].unique():
    working = close_times[close_times["GRB"]==grb_id]
    for optical_point in working["t_o"].unique():
        subworking = working[working["t_o"]==optical_point]
        closest = subworking["dt%"].min()
        close_times.drop(subworking[subworking["dt%"] != closest].index, axis=0, inplace=True)
        


# darkest matched pair for each GRB (not necessarily *dark*)
darkest_times = results.copy()
for grb_id,working in darkest_times.groupby("GRB"):
    if any([np.isfinite(fx.minus) for fx in working["F_x"]]):
        darkest_times.drop(working[[np.isinf(fx.minus) for fx in working["F_x"]]].index,axis=0,inplace=True)
        working = working[[np.isfinite(fx.minus) for fx in working["F_x"]]]
    darkest_beta = working["B_ox"].min()
    darkest_times.drop(working[working["B_ox"] != darkest_beta].index, axis=0, inplace=True)
    
for GRB in close_times["GRB"].unique():
    print(GRB+"\t",close_times["GRB"].tolist().count(GRB))
    
#print(darkest_times)
bursts = ["201216C", "180720B", "190114C", "190829A", "201015A", "221009A"]


'''
for burst in bursts:
    fluxdata = XRT_lightcurve(burst)
    Gamma,mode = get_photonIndex(burst)
    NH,mode = get_columnDensity(burst)
    E_range = np.linspace(0.3,10,100)
    spectrum = (E_range / 4.135667696e-18)**-(Gamma.value-1)
    spec1 = (E_range / 4.135667696e-18)**-(Gamma.maximum-1)
    spec2 = (E_range / 4.135667696e-18)**-(Gamma.minimum-1)

    plt.fill_between(E_range,[0]*len(E_range),spectrum,fc="none",ec="k",hatch="//")
    plt.fill_between(E_range,spec1,spec2,alpha=0.5,color="r")
    plt.xlim(1e-1,25)
    plt.ylim(1e-30,1e-9)
    plt.yscale("log")
    plt.xscale("log")
    plt.gca().set_xticks([0.1,0.3,1,3,10])
    plt.gca().set_xticklabels([0.1,0.3,1,3,10])
    #plt.gca().set_yticks([])
    #plt.gca().set_yticklabels([])
    plt.xlabel("Energy [keV]")
    plt.ylabel("Flux density (log scale)")
    #plt.savefig(f'Downloads/dt05_flux_dens_{burst}.pdf')
    plt.show()


    plt.errorbar(fluxdata.Time,fluxdata.Flux,xerr=np.array(fluxdata.Tneg,fluxdata.Tpos).T,
             yerr=np.array(fluxdata.Fluxneg,fluxdata.Fluxpos).T,linestyle="",capthick=0)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Swift XRT Lightcurve for GRB %s\n$\\beta_X$ = %s" % (burst,Gamma-1))
    plt.xlabel("Time since trigger [s]")
    plt.ylabel("Flux (0.3-10 keV) [erg/s/cm$^2$]")
    plt.grid(linestyle="--")
    #plt.savefig(f'Downloads/dt05_XRT_lightcurve_{burst}.pdf')
    plt.show()
    
    MOSFIRE_J = pd.read_table("https://www2.keck.hawaii.edu/inst/mosfire/data/filter/mosfire_J.txt",delimiter="   ").iloc[:,:2]
    MOSFIRE_J = MOSFIRE_J.rename(columns=dict(zip(MOSFIRE_J.columns,["Wavelength","Transmission"]))).apply(pd.to_numeric,errors="coerce").dropna()
    MOSFIRE_J["Wavelength"] *= 10000
    MOSFIRE_J["Transmission"] *= 100

    MOSFIRE_K = pd.read_table("https://www2.keck.hawaii.edu/inst/mosfire/data/filter/mosfire_K.txt",delimiter="   ",engine='python').iloc[:,:2]
    MOSFIRE_K = MOSFIRE_K.rename(columns=dict(zip(MOSFIRE_K.columns,["Wavelength","Transmission"]))).apply(pd.to_numeric,errors="coerce").dropna()
    MOSFIRE_K["Wavelength"] *= 10000
    MOSFIRE_K["Transmission"] *= 100
    effective_wavelength(MOSFIRE_K,show_plot=True)
    #RbTable.drop([37,55,61,73],axis=0,inplace=True) # smoothing
    RbTable.sort_values(by="lambda_eff",inplace=True) # sort in order of wavelength: UV -> IR
    plt.plot(RbTable["lambda_eff"], RbTable["R_b"], ".--")
    plt.xlabel("$\lambda_\mathrm{eff}$ [$\AA$]")
    plt.ylabel("R(λ)")
    plt.ylim(0,8)
    plt.grid(linestyle="--")
    #plt.savefig(f'Downloads/dt05_eff_wavelength_{burst}.pdf')
    # plt.savefig("./products/RbTable.png",bbox_inches="tight",dpi=300,)
    

    
for grb in close_times["GRB"].unique():
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes((0,0,1,.75))
    
    subset = new_optical.loc[(new_optical["GRB"]==grb)].copy()
    dark_points = close_times.loc[(close_times["GRB"]==grb) & (close_times["vdH_dark"]|close_times["Jak_dark"])]

    subset.loc[:,"band"] = ["Ultraviolet" if wavelength<3000 else "Infrared" if wavelength>8000 else None if pd.isna(wavelength) else "Optical" for wavelength in subset["λ_eff"]]
    bands = {"Ultraviolet":["darkviolet","$\lambda<3000$ Å"],
             "Optical":    ["dodgerblue","$3000<\lambda<8000$ Å"],
             "Infrared":   ["red", "$\lambda<8000$ Å"]}
    for band,info in bands.items():
        subsubset = subset[subset["band"]==band]
        if not len(subsubset):
            print(f"no {band} for GRB {grb}")
            continue
        neg_err = [0.3*flux.value if np.isinf(flux.minus) else flux.minus for flux in subsubset["Flux (Jy)"]]
        pos_err = [0.3*flux.value if np.isinf(flux.plus) else flux.plus for flux in subsubset["Flux (Jy)"]]
        ax.errorbar(subsubset["Time (s)"],[flux.value for flux in subsubset["Flux (Jy)"]],
                    marker="_",linestyle="",color=info[0],label=f"{band}",yerr=np.array((neg_err,pos_err)),
                    uplims=[np.isinf(point.minus) for point in subsubset["Flux (Jy)"]],
                    lolims=[np.isinf(point.plus) for point in subsubset["Flux (Jy)"]],capthick=0)
    
    subset = xrt_data.loc[xrt_data["GRB"]==grb].copy()
    neg_err = [0.4*flux.value if np.isinf(flux.minus) else flux.minus for flux in subset["SpecFlux"]]
    pos_err = [0.4*flux.value if np.isinf(flux.plus) else flux.plus for flux in subset["SpecFlux"]]
    ax.errorbar(subset.Time,[flux.value for flux in subset.SpecFlux],
                xerr=np.array(subset.Tneg,subset.Tpos).T,yerr=np.array((neg_err,pos_err)),
                uplims=[np.isinf(point.minus) for point in subset["SpecFlux"]],
                lolims=[np.isinf(point.plus) for point in subset["SpecFlux"]],
                linestyle="",capthick=0,color="k",label="X-ray")
    
    jak = dark_points[dark_points["Jak_dark"] & ~dark_points["vdH_dark"]]
    vdh = dark_points[dark_points["vdH_dark"] & ~dark_points["Jak_dark"]]
    both = dark_points[dark_points["Jak_dark"] & dark_points["vdH_dark"]]
    print(len(jak),len(vdh),len(both))
    
    if len(jak)>0:
        ax.plot(jak["t_o"],[flux.value for flux in jak["F_o"]],
            marker="o",markersize=8,mfc="none",mec="slateblue",
            linestyle="",label="Dark (Jak only)")
    if len(vdh)>0:
        ax.plot(vdh["t_o"],[flux.value for flux in vdh["F_o"]],
            marker="o",markersize=8,mfc="none",mec="goldenrod",
            linestyle="",label="Dark (vdH only)")
    if len(both)>0:
        ax.plot(both["t_o"],[flux.value for flux in both["F_o"]],
            marker="o",markersize=8,mfc="none",mec="seagreen",
            linestyle="",label="Dark (both)")

    else:
        ax.plot(close_times["t_o"], [flux.value for flux in close_times["F_o"]],
                marker="o",markersize=8,mfc="none",mec="violet",
                linestyle="",label="Not Dark")
    
#     ax.axvline(11*60*60,linestyle="--",color="k",label="11 hr")
    ax.set(xscale="log",yscale="log",xlabel="Time post-trigger [s]",ylabel="Flux [Jy]")
    tmin,tmax = ax.get_xlim()
    ax.legend(loc="best", fancybox=True)
    ax.grid(linestyle="--")

    ax2 = fig.add_axes((0,.75,1,.25))
    matches = results[results["GRB"]==grb].copy()
    for t,sub in matches.groupby("t_o"):
        matches.drop(sub[sub["dt%"]!=sub["dt%"].min()].index,axis=0,inplace=True)
        
    ax2.errorbar(matches["t_o"],UncertaintyArray(matches[B_ox_name]).values,
                 yerr=[[e if np.isfinite(e) else 0.2 for e in UncertaintyArray(matches[B_ox_name]).minus],
                       UncertaintyArray(matches[B_ox_name]).plus],uplims=np.isinf(UncertaintyArray(matches[B_ox_name]).minus),
                 linestyle="",marker=".",color="k",elinewidth=0.75,label=r"$\beta_\mathrm{ox}$")
    ax2.axhline(0.5,color="darkorange",linestyle="--", label=r"$\beta=0.5$")

    B_x = a_u(*GRBs.loc[GRBs["GRB"]==grb,["Beta_X","Beta_X_pos","Beta_X_neg"]].values[0])
    ax2.axhline(B_x.value-0.5,color="k",linestyle=":",label=r"$\beta_\mathrm{x}-0.5$")
    ax2.fill_between([tmin,tmax],B_x.minimum-0.5,B_x.maximum-0.5,alpha=0.5,color="silver")
    ax2.legend(loc="best", fancybox=True)
    default_lolim,default_uplim = ax2.get_ylim()

    ax2.set(xscale="log",xlim=(tmin,tmax),ylim=(default_lolim-.1,default_uplim+.1),#xlabel="Time [s]",
            ylabel="Spectral index",xticklabels=[],title=f"GRB {grb}: {len(dark_points)} dark points")
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax2.grid(linestyle="--")
    #plt.savefig(f'Downloads/dt05_dark_lightcurves_{grb}.pdf')
    #fig.savefig(f"./products/dark lightcurves/{grb}.png",dpi=300,transparent=False,facecolor="white",bbox_inches="tight")
    plt.show()
    #display(close_times.loc[close_times["GRB"]==grb,["GRB","t_o","dt%",B_ox_name,"B_x","α","Jak_dark","vdH_dark"]])
'''



    
notvalid = lambda x : any((np.isinf(x), pd.isna(x)))
b_ox = [b.value for b in darkest_times[B_ox_name]]
oxplus = [0.2 if notvalid(v.plus) else v.plus for v in darkest_times[B_ox_name]]
oxminus = [0.2 if notvalid(v.minus) else v.minus for v in darkest_times[B_ox_name]]
oxuplims = [notvalid(v.minus) for v in darkest_times[B_ox_name]]
oxlolims = [notvalid(v.plus) for v in darkest_times[B_ox_name]]
b_x = [b.value for b in darkest_times["B_x"]]
labels = [i for i in darkest_times["GRB"]]
xplus = [0.2 if notvalid(v.plus) else v.plus for v in darkest_times["B_x"]]
xminus = [0.2 if notvalid(v.minus) else v.minus for v in darkest_times["B_x"]]
t_obs = darkest_times["t_o"].values


fig = plt.figure(figsize=(12,8))
grid = plt.GridSpec(5, 5, hspace=0.15, wspace=0.1)

'''
ax_main = fig.add_subplot(grid[1:-1, 0:-1])
ax_right = fig.add_subplot(grid[1:-1, -1], yticklabels=[])#, xticklabels=[])
ax_top = fig.add_subplot(grid[0, 0:-1], xticklabels=[])#, yticklabels=[])
'''
x = np.linspace(0.2 ,2.2 ,500)
y = np.linspace(-1,1.25,500)
X,Y = np.meshgrid(x,y)

plt.errorbar([b_x[i] for i in range(len(b_x)) if not (oxuplims[i] and oxlolims[i])],
                 [b_ox[i] for i in range(len(b_ox)) if not (oxuplims[i] and oxlolims[i])],
                 xerr=[[xminus[i] for i in range(len(xminus)) if not (oxuplims[i] and oxlolims[i])],
                       [xplus[i] for i in range(len(xplus)) if not (oxuplims[i] and oxlolims[i])]],
                 yerr=[[oxminus[i] for i in range(len(oxminus)) if not (oxuplims[i] and oxlolims[i])],
                       [oxplus[i] for i in range(len(oxplus)) if not (oxuplims[i] and oxlolims[i])]],
                 uplims=[oxuplims[i] for i in range(len(oxuplims)) if not (oxuplims[i] and oxlolims[i])],
                 lolims=[oxlolims[i] for i in range(len(oxlolims)) if not (oxuplims[i] and oxlolims[i])],
                 zorder=0, linestyle="",marker=".", ecolor="dimgrey", elinewidth=0.75, capthick=1)


plt.plot(x, x,linestyle="-.",color="tomato",label=r"$\beta_\mathrm{ox}=\beta_\mathrm{x}$")
plt.plot(x, x-.5,"k:",label=r"$\beta_\mathrm{ox}=\beta_\mathrm{x}-0.5$")
plt.plot(x,[.5]*len(x),linestyle="--",color="darkorange",label=r"$\beta_\mathrm{ox}=0.5$") 


for i in range(len(b_x)):
    if not (oxuplims[i] and oxlolims[i]):
        # Offset the labels slightly based on index
        x_offset = 0.02 * (i % 2)  # Alternate offset for x
        y_offset = 0.02 * ((i // 2) % 2)  # Alternate offset for y
        plt.text(b_x[i] + x_offset, b_ox[i] + y_offset, f'{labels[i]}', fontsize=9, ha='left')
        

#plt.set(xlim=(x.min(),x.max()),ylim=(y.min(),y.max()),
            #xlabel=r"$\beta_\mathrm{x}$",ylabel=r"$\beta_\mathrm{ox}$")
plt.xlabel(r"$\beta_\mathrm{x}$")
plt.ylabel(r"$\beta_\mathrm{ox}$")

plt.xlim(0.5,1.4)
plt.ylim(-.9, 0.9)
#plt.xaxis.set_ticks_position("both")

'''
ax_kde_x = ax_top.twinx()
valid_bx = UncertaintyArray([entry for entry in darkest_times["B_x"] if pd.notna(entry.value)])
ax_kde_x.fill_between(x,valid_bx.pdf(x),facecolor="silver",edgecolor="grey",alpha=.5)
ax_kde_x.set_ylim(0,)
ax_kde_x.set(yticks=[])
ax_top.hist(b_x,23,orientation='vertical', color="C0")
ax_top.set(xlim=ax_main.get_xlim(), ylabel="Count", yticks=range(0,14,3))

ax_kde_y = ax_right.twiny()
valid_box = UncertaintyArray()
for i,row in darkest_times.iterrows():
    if np.isfinite(row[B_ox_name].minus):
        neg = row[B_ox_name].minus
    else:
        neg = 0.5
    if np.isfinite(row[B_ox_name].plus):
        pos = row[B_ox_name].plus
    else:
        pos = 0.5
    if np.isfinite(row[B_ox_name].value):
        valid_box.append(a_u(row[B_ox_name].value,pos,neg))
ax_kde_y.fill_betweenx(y,valid_box.pdf(y),alpha=.5,
                       facecolor="silver",edgecolor="grey",label="Probability density")
ax_kde_y.set_xlim(0,)
ax_kde_y.set(xticks=[])
ax_right.hist([[b.value for b in darkest_times[B_ox_name] if all(np.isfinite(b.items()))],
               [b.value for b in darkest_times[B_ox_name] if not all(np.isfinite(b.items()))]],
              21, orientation='horizontal', stacked=True, color=["C0","lightblue"],
              label=["Detection-based","Upper/lower limit"])
ax_right.set(ylim=(ax_main.get_ylim()), xlabel="Count")
ax_right.set_xlim(0,)
'''
# Z_x = np.sum([bx.pdf(X) for bx in valid_bx],axis=0)
# Z_x /= np.sum(Z_x)
# Z_ox = np.sum([box.pdf(Y) for box in valid_box],axis=0)
# Z_ox /= np.sum(Z_ox)
# ax_main.contourf(X,Y,np.sqrt(Z_x**2+Z_ox**2),30,cmap="Greys",vmax=4e-5,vmin=0)

fig.legend(loc="lower left",bbox_to_anchor=(0.747,0.736))
# fig.colorbar(scatter,ax=ax_right,label=r"$\log_{10}(t_\mathrm{o}\ \mathrm{[s]})$")

plt.title(r"Spectral index distribution (max $dt_\%="+f"{max_dt:.1f}$)")

#plt.savefig("Downloads/beta_distributions.pdf",dpi=300,bbox_inches="tight")

plt.show()

