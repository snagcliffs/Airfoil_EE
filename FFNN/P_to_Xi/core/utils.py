import numpy as np
from scipy.interpolate import griddata

def interp(field, Cx, Cy, XX, YY, method='linear', mask=None):
    """
    field - 1D array of cell values
    Cx, Cy - cell x-y values
    X, Y - meshgrid x-y values
    """
    ngrid = len(XX.flatten())
    grid_field = np.squeeze(np.reshape(griddata((Cx, Cy), field, (XX, YY), method=method), (ngrid, 1)))
    grid_field = grid_field.reshape(XX.shape)
    
    if mask is not None:
        for m in mask: grid_field[m[1],m[0]] = 0
        
    return grid_field

def get_wing_boundary(alpha=5, n_points=50):
    
    # Parameters for naca 4412 airfoil
    m = 0.04
    p = 0.4
    t = 0.12
    c = 1
    x_nose = -0.25
    
    X = []
    Y = []
    
    for j in range(n_points):

        x = j/(n_points-1)

        # Airfoil thickness
        yt = 5*t*(0.2969*np.sqrt(x)-0.126*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)

        # Center coord height
        if x < p:
            yc = m/p**2*(2*p*(x/c)-(x/c)**2)
            dyc = 2*m/p**2*(p-x/c)
        else:
            yc = m/(1-p)**2*(1-2*p+2*p*(x/c)-(x/c)**2)
            dyc = 2*m/(1-p)**2*(p-x/c)

        theta = np.arctan(dyc)
        xu = x - yt*np.sin(theta) + x_nose
        yu = yc + yt*np.cos(theta)

        xj = np.round(xu*np.cos(-alpha*np.pi/180) + yu*np.sin(alpha*np.pi/180), 5)
        yj = np.round(-xu*np.sin(alpha*np.pi/180) + yu*np.cos(alpha*np.pi/180), 5)

        X.append(xj)
        Y.append(yj)

    for j in range(n_points):

        x = 1-(j+1)/n_points # Now going backwards

        # Airfoil thickness
        yt = 5*t*(0.2969*np.sqrt(x)-0.126*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)

        # Center coord height
        if x < p:
            yc = m/p**2*(2*p*(x/c)-(x/c)**2)
            dyc = 2*m/(1-p)**2*(p/c-x/c**2)
        else:
            yc = m/(1-p)**2*(1-2*p+2*p*(x/c)-(x/c)**2)
            dyc = 2*m/(1-p)**2*(p/c-x/c**2)

        theta = np.arctan(dyc)
        xb = x + yt*np.sin(theta) + x_nose
        yb = yc - yt*np.cos(theta)

        xj = np.round(xb*np.cos(-alpha*np.pi/180) + yb*np.sin(alpha*np.pi/180), 5)
        yj = np.round(-xb*np.sin(alpha*np.pi/180) + yb*np.cos(alpha*np.pi/180), 5)

        X.append(xj)
        Y.append(yj)
    
    return X,Y