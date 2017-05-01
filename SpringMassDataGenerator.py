from scipy.integrate import odeint
from numpy import random

def genSpringMassData(stoptime, numpoints):

    m1_range = (1, 10)
    m2_range = (1, 10)

    # Spring constants
    k1_range = (1, 30)
    k2_range = (1, 30)

    # Natural lengths
    L1_range = (1, 10)
    L2_range = (1, 10)

    # Friction coefficients
    b1_range = (0, 0)
    b2_range = (0, 0)

    # Initial conditions
    # x1 and_range  x2 are the initial displacements; v1 and v2 are the initial velocities
    x1_range = (1, 10)
    v1_range = (1, 10)
    x2_range = (11, 20)
    v2_range = (1, 10)

    def f_randSamp(x): return random.uniform(x[0], x[1])

    paras = [f_randSamp(x) for x in [m1_range, m2_range,
                                     k1_range, k2_range,
                                     L1_range, L2_range,
                                     b1_range, b2_range,
                                     x1_range, v1_range,
                                     x2_range, v2_range]]
    m1, m2, k1, k2, L1, L2, b1, b2, x1, v1, x2, v2 = paras

    def vectorfield(w, t, p):
        """
        Defines the differential equations for the coupled spring-mass system.

        Arguments:
            w :  vector of the state variables:
                      w = [x1,v1,x2,v2]
            t :  time
            p :  vector of the parameters:
                      p = [m1,m2,k1,k2,L1,L2,b1,b2]
        """
        x1, v1, x2, v2 = w
        m1, m2, k1, k2, L1, L2, b1, b2 = p

        # Create f = (x1',v1',x2',v2'):
        f = [v1,
             (-b1 * v1 - k1 * (x1 - L1) + k2 * (x2 - x1 - L2)) / m1,
             v2,
             (-b2 * v2 - k2 * (x2 - x1 - L2)) / m2]
        return f

    # ODE solver parameters
    abserr = 1.0e-8
    relerr = 1.0e-6

    # ---------------------------------------------------------------------------- #
    # Create the time samples for the output of the ODE solver.                    #
    # I use a large number of points, only because I want to make                  #
    # a plot of the solution that looks nice.                                      #
    # ---------------------------------------------------------------------------- #
    t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

    # Pack up the parameters and initial conditions:
    p = [m1, m2, k1, k2, L1, L2, b1, b2]
    w0 = [x1, v1, x2, v2]

    # Call the ODE solver.
    wsol = odeint(vectorfield, w0, t, args=(p,),
                  atol=abserr, rtol=relerr)

    return wsol
