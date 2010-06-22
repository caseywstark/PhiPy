# Enthought imports
from numpy import linspace
import scipy as sp
from scipy.optimize import leastsq

from enthought.enable.api import Component, ComponentEditor
from enthought.traits.api import HasTraits, Instance, Dict, Range, Array, Float, Enum, String, on_trait_change
from enthought.traits.ui.api import Item, Group, View

from enthought.chaco.api import ArrayPlotData, Plot, Legend
from enthought.chaco.tools.api import PanTool, ZoomTool

# PhiPy imports
from data import data as data_dict
from fowler import kelvin_to_kT, fowler_log, expansion_f, fowler_yield, fowler_fit_error

plot_size = (700, 300)
title = "PhiPy"

def _create_data_plot_component(data_pd):
    plot = Plot(data_pd, title="Data Space")
    plot.x_axis.title = "Photon Energy (eV)"
    plot.y_axis.title = "Yield (arbitrary)"

    # Experimental data
    plot.plot(("energies", "yields"), type="scatter", name="data", marker="circle", color="red", marker_size=5)

    # Yield predicted by fowler fit params, at given energies
    plot.plot(("energies", "yields_fit"), type="scatter", name="fit", marker="circle", color="blue", marker_size=5)

    # Plot Legend
    plot.legend.visible = True
    plot.legend.align = 'ul'
    plot.legend.line_spacing = 6

    # Attach some tools to the plot
    plot.tools.append(PanTool(plot, drag_button="right"))
    plot.overlays.append(ZoomTool(plot))

    return plot

def _create_fowler_plot_component(fowler_pd):
    plot = Plot(fowler_pd, title="Fowler Space")
    plot.x_axis.title = "Chemical Potential (unitless)"
    plot.y_axis.title = "Log(Y/T^2)"
    
    # Universal fowler curve
    plot.plot(("universal_x", "universal_y"), type="line", name="universal curve", color="green")
    
    # Fit points
    plot.plot(("fit_x", "fit_y"), type="scatter", name="fowler fit", marker="circle", color="blue", marker_size=8)
    
    # Plot Legend
    plot.legend.visible = True
    plot.legend.align = 'ul'
    plot.legend.line_spacing = 6
    
    # Attach some tools to the plot
    plot.tools.append(PanTool(plot, drag_button="right"))
    plot.overlays.append(ZoomTool(plot))
    
    return plot

#=============================
# Defines the PhiPy interface.
#=============================
class PhiPy(HasTraits):
    data_source = Enum("Potassium-39", "Indium-22", "Indium-25")
    data = Dict

    energies = Array
    yields = Array
    mus = Array
    log_yields = Array

    data_pd = ArrayPlotData()
    data_plot = Instance(Component)
    fowler_pd = ArrayPlotData()
    fowler_plot = Instance(Component)
    
    # fowler_method = Enum("expansion", "integral")
    initial_phi = Range(low=0.1, high=10., value=3.)
    initial_temp = Range(low=100., high=2000., value=300., mode='slider')
    initial_prefactor = Range(low=-50., high=50., value=0.)
    fowler_order = Range(low=1, high=20, value=10, mode='slider')
    fit_message = String
    
    phi = Float(5.)
    temp = Float(300.)
    prefactor = Float(0)
    
    phi_control = Range(low=0.1, high=10., value=3.)
    temp_control = Range(low=1., high=3000., value=300., mode='slider')
    prefactor_control = Range(low=-20., high=50., value=0.)
    
    traits_view = View(
        Group(
            Item('data_source', label="Choose Data Set"),
            Item('data_plot', editor=ComponentEditor(size=plot_size), show_label=False),
            Item('fowler_plot', editor=ComponentEditor(size=plot_size), show_label=False),
            Group(
                Item('initial_phi', label="Work function guess"),
                Item('initial_temp', label="Temperature guess"),
                Item('initial_prefactor', label="Prefactor guess"),
                Item('fowler_order', label="Fowler order"),
                Item('fit_message', style='readonly', label='Least squares output'),
                orientation = "vertical",
            ),
            Group(
                Item('phi', style='readonly', label="Work Function (eV)"),
                Item('temp', style='readonly', label="Temperature (K)"),
                Item('prefactor', style='readonly', label="Prefactor (B)"),
                orientation = "horizontal"
            ),
            Group(
                Item('phi_control', label="Work function control"),
                Item('temp_control', label="Temperature control"),
                Item('prefactor_control', label="Prefactor control"),
                orientation = "vertical",
            ),
            orientation = "vertical",
        ),
        resizable=True,
        title=title,
    )
    
    def _data_default(self):
        self.data = data_dict[self.data_source]
        self.energies = self.data['photon_energies']
        self.yields = self.data['yields']
        
        return self.data
    
    def _data_plot_default(self):
        return _create_data_plot_component(self.update_data_pd())
    
    def update_data_pd(self):
        self.data_pd.set_data("energies", self.energies)
        self.data_pd.set_data("yields", self.yields)
        
        (phi, temp, prefactor) = map(float, (self.phi_control, self.temp_control, self.prefactor_control))
        fit_yields = fowler_yield(phi, temp, prefactor, self.energies)
        self.data_pd.set_data("yields_fit", fit_yields)
        return self.data_pd
    
    def _fowler_plot_default(self):
        return _create_fowler_plot_component(self.update_fowler_pd())
    
    def update_fowler_pd(self):
        mus = (self.energies - float(self.phi_control)) / (float(self.temp_control) * kelvin_to_kT)
        log_yields = fowler_log(float(self.phi_control), float(self.temp_control), float(self.prefactor_control), self.energies) - float(self.prefactor_control)
        
        mu0 = min(mus)
        mu1 = max(mus)
        mu_distance = mu1 - mu0
        mu_range = linspace(-10, 20, 100) # mu0 - mu_distance, mu1 + mu_distance, 100)
        universal_yields = sp.log(expansion_f(mu_range))
        
        self.fowler_pd.set_data("universal_x", mu_range)
        self.fowler_pd.set_data("universal_y", universal_yields)
        self.fowler_pd.set_data("fit_x", mus)
        self.fowler_pd.set_data("fit_y", log_yields)
        
        return self.fowler_pd
    
    @on_trait_change('data_source')
    def update_data(self):
        self.data = data_dict[self.data_source]
        self.energies = self.data['photon_energies']
        self.yields = self.data['yields']
        
        self._redo_fit()
   
    @on_trait_change('fowler_method, initial_phi, initial_temp, initial_prefactor, fowler_order')
    def _redo_fit(self):
        self.fit()
        self._update_plots()
    
    @on_trait_change('phi_control, temp_control, prefactor_control')
    def _update_plots(self):
        self.update_data_pd()
        self.update_fowler_pd()
        self.data_plot.request_redraw()
        self.fowler_plot.request_redraw()
        if any(self.data_pd['yields_fit']):
            print "new fit yield %f" % self.data_pd['yields_fit'][-1]
        else:
            print "bad fit yield"
    
    def fit(self):
        """ Update phi, temp, and prefactor based on the new initial params. """
        parameters = map(float, [self.initial_phi, self.initial_temp, self.initial_prefactor])
        
        fit_params, x_cov, infodict, mesg, success = leastsq(fowler_fit_error,
            parameters, args=(sp.array(self.energies), sp.array(self.yields), self.fowler_order),
            full_output=True, maxfev=10000)
        
        (self.phi, self.temp, self.prefactor) = fit_params
        (self.phi_control, self.temp_control, self.prefactor_control) = fit_params
        self.fit_message = 'Code: %d | Message: %s' % (success, mesg)
    
    def start(self):
        self.update_data()
        self.configure_traits()

#============
# Let her run
#============
if __name__ == "__main__":
    PhiPy().start()