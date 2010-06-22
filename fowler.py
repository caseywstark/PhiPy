import scipy as sp
from scipy.optimize import leastsq
from scipy.integrate import quad

# defaults
default_parameters = [2.5, .05, 5.0] # phi, T, B
default_order = 10
kelvin_to_kT = 8.6173E-5

# Utility functions (fowler stuff)
def get_mu(energy, phi, temperature):
    """ Calculates the chemical potential (mu) from the args. """
    return (energy - phi)/(temperature * kelvin_to_kT)

def fowler_integrand(y, energy, phi, temperature):
    """ The integrand in the Fowler formula. """
    exponent = -y + get_mu(energy, phi, temperature)
    integrand = sp.log(1.0 + sp.power(sp.e, exponent))
    return integrand

def fowler_integral(phi, temperature, photon_energies, lower_bound=0, upper_bound=9999):
    """ Evaluates the integral in the Forwler formula. """
    """
    results = []
    for energy in photon_energies:
        result = quad(fowler_integrand, lower_bound, upper_bound, args=(energy, phi, temperature))
        results.append(result[0])
    """
    results = quad(fowler_integrand, lower_bound, upper_bound, args=(photon_energies, phi, temperature))
    return results

def expansion_f(mus, order=default_order):
    """
    Evaluates the function f(mu), the expansion of the logarithm integrated,
    from the finite temperature Fowler function.
    
    Takes ``order`` to specify the desired order of the expansion.
    
    """
    results = []
    for mu in mus:
        if mu >= 0:
            the_out = (sp.pi**2)/6 + (mu**2)/2 # first order, must have
            for n in range(1, order):
                try:
                    n_term = (-1)**n*sp.exp(-n*mu)/n**2
                except AttributeError:
                    print >>sys.stderr, ('Fowler function expansion failed at n = %d and order = %d') % (n, order)
                the_out += n_term
            results.append(the_out)
        else:
            the_out = 0
            for n in range(1, order):
                try:
                    n_term = -(-1)**n*sp.exp(-n*mu)/n**2
                except AttributeError:
                    print >>sys.stderr, ('Fowler function expansion failed at n = %d and order = %d') % (n, order)
                the_out += n_term
            results.append(the_out)
    return results

def fowler(phi, temperature, energy, order=default_order):
    """
    The finite temperature fowler function. Evaluates the finite temperature Fowler function at the given frequency
    ``nu``, work function ``phi``, and temperature ``T``. These three parameters determine the chemical potential mu, which is fed to ``expansion_f``.
    
    """
    mu = get_mu(energy, phi, temperature)
    return expansion_f(mu, order=order)

def fowler_yield(phi, T, B, energy):
    """ The PI yield predicted by Fowler's model at a given ``phi``, ``T``, and ``B``. """
    C = sp.exp(B)
    fowl = sp.array(fowler(phi, T, energy))
    result = C * (T)**2. * fowl
    return result

def fowler_log_yield(the_yield, T):
    """ The fowler space equivalent of the yield \( \log \left( Y / T^2 \right) \). """
    T = T*kelvin_to_kT
    return sp.log(the_yield / T**2)

def fowler_log(phi, T, B, energy):
    """ Evaluates the right side of the fowler relation \( \log \left( Y / T^2 \right) = B + \log f \left( \frac{ h \nu - \phi }{ k_b T } \right) \) """
    return B + sp.log(fowler(phi, T, energy))

def fowler_fit_error(params, photon_energies, yields, order=default_order):
    """ The error function for the fit. """
    phi, T, B = (params[0], params[1], params[2])
    return B + sp.log(fowler(phi, T, photon_energies, order)) - sp.log(yields / T**2)

class Element:
    """ Definition of a chemical element. """
    def __init__(self, symbol, name, Z):
        self.symbol = symbol
        self.name = name
        self.Z = Z
    
    def __str__(self):
        return '%s-%s' % (self.name, self.Z)

class Cluster:
    """ Definition of an atomic cluster. """
    def __init__(self, element, size):
        self.element = element
        self.size = size
    
    def __str__(self):
        return '%s-%s' % (self.element.symbol, self.size)