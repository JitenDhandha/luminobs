import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

def double_power_law(phi_star, Mstar, alpha, beta):
    M = np.linspace(-25,-10,100)
    x = 10**(0.4*(alpha+1)*(M-Mstar))
    y = 10**(0.4*(beta+1)*(M-Mstar))
    return M, phi_star/(x+y)

def schechter(phi_star, Mstar, alpha):
    M = np.linspace(-25,-10,100)
    x = 10**(0.4*(alpha+1)*(M-Mstar))
    y = 10**(0.4*(Mstar-M))
    return M, (phi_star / x ) * np.exp(-y)

def give_me_a_figure():
    fig, ax = plt.subplots(1,1, figsize=(7,5))
    ax.set_xlabel(r'$M_\mathrm{UV}~[\mathrm{mag}$]',fontsize=15)
    ax.set_ylabel(r'$\Phi~[\mathrm{\#/mag}^{-1}\mathrm{Mpc}^{-3}]$',fontsize=15)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(True, which='both', alpha=0.2)
    ax.set_xlim(-24,-16)
    ax.set_ylim(1e-8,1e-2)
    return fig, ax

class JWSTData:
    
    def __init__(self, filename):
        
        self.arxiv_link = None
        self.z = None
        self.UVLF_df = None
        self.fit_df = None
        self._read_UVLF(filename+'.csv')
        self._read_fit(filename+'_fit.csv')
        
    def _read_fit(self, filename):
        with open(filename) as f:
            self.fit_type = f.readline().split('#')[-1].strip()
        if(self.fit_type=="DPL"):
            data = pd.read_csv(filename,
                            delimiter=',',
                            comment='#',
                            names=['z','phi_star','Mstar','alpha','beta'])
        elif(self.fit_type=="Schechter"):
            data = pd.read_csv(filename,
                            delimiter=',',
                            comment='#',
                            names=['z','phi_star','Mstar','alpha'])
        else:
            raise ValueError(f"Fit type {self.fit_type} not recognised")
        self.fit_df = [data.groupby(data.z).get_group(z) for z in self.z]
        
    def _read_UVLF(self, filename):
        with open(filename) as f:
            self.arxiv_link = f.readline().split('#')[-1].strip()
        data = pd.read_csv(filename,
                           delimiter=',',
                           comment='#',
                           names=['z','MUV','MUV bin size','UVLF','UVLF +ve error','UVLF -ve error'],)
        self.z = list(set(data.z))
        self.UVLF_df = [data.groupby(data.z).get_group(z) for z in self.z] 
        
    def get_UVLF(self, z):
        if(z not in self.z):
            raise ValueError(f"Please pick a redshift from {self.z}")
        else:
            return self.UVLF_df[self.z.index(z)]
        
    def get_fit(self, z):
        if(z not in self.z):
            raise ValueError(f"Please pick a redshift from {self.z}")
        else:
            fit_z_df = self.fit_df[self.z.index(z)]
            if(self.fit_type=="DPL"):
                return double_power_law(*list(fit_z_df.iloc[0])[1:])
            elif(self.fit_type=="Schechter"):
                return schechter(*list(fit_z_df.iloc[0])[1:])
            
    def plot_UVLF(self, z, ax, color='black', label=None, xerr=True, fit=False):
        UVLF_z_df = self.get_UVLF(z)
        x = np.copy(UVLF_z_df["MUV"].values)
        y = np.copy(UVLF_z_df["UVLF"].values)
        if(xerr):
            xerr = np.copy(UVLF_z_df["MUV bin size"].values)/2
        else:
            xerr = None
        yerr_p = np.copy(UVLF_z_df["UVLF +ve error"].values)
        yerr_m = np.copy(abs(UVLF_z_df["UVLF -ve error"].values))
        uplims = yerr_m < 1e-30
        yerr_m[uplims] = 0.5 * y[uplims]
        ax.errorbar(x, y,
                    xerr=xerr, 
                    yerr=[yerr_m, yerr_p],
                    uplims=uplims,
                    fmt='o', color=color, 
                    capsize = 4,
                    label=label)
        #xlim, ylim = ax.get_xlim(), ax.get_ylim()
        if(fit):
            fit_x, fit_y = self.get_fit(z)
            ax.plot(fit_x, fit_y, color=color, linestyle='--')
        #ax.set_xlim(xlim)
        #ax.set_ylim(ylim)
    
_current_dir = pathlib.Path(__file__).parent.absolute()+'/data/'
Finkelstein2023 = JWSTData(_current_dir+'Finkelstein2023') # https://arxiv.org/abs/2311.04279
Donnan2023 = JWSTData(_current_dir+'Donnan2023') # https://arxiv.org/abs/2207.12356
Leung2023 = JWSTData(_current_dir+'Leung2023') # https://arxiv.org/abs/2306.06244
McLeod2023 = JWSTData(_current_dir+'McLeod2023') # https://arxiv.org/abs/2304.14469
Bouwens2023 = JWSTData(_current_dir+'Bouwens2023') # https://arxiv.org/abs/2211.02607
PerezGonzalez2023 = JWSTData(_current_dir+'Perez-Gonzalez2023') # https://arxiv.org/abs/2302.02429
Harikane2023 = JWSTData(_current_dir+'Harikane2023') # https://arxiv.org/abs/2208.01612