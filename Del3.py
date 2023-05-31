import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# fields
surface_data = np.array([[0, 0, 0]])
grid_x = np.array([0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.5])
grid_y = np.array([0, 0.25, 0.75, 1.25, 1.75, 2.0])
pop_dens = np.array([[100, 100, 100, 100, 100, 100],
                      [100, 100, 80, 40, 40, 40],
                      [100, 100, 80, 40, 40, 40],
                      [100, 100, 100, 70, 80, 40],
                      [50, 90, 100, 80, 70, 60],
                      [0, 10, 0, 10, 80, 80],
                      [0, 0, 50, 30, 10, 10],
                      [40, 40, 0, 0, 80, 60],
                      [80, 80, 50, 60, 50, 0],
                      [80, 80, 80, 80, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
for i in range(0, len(pop_dens[:])):
    for j in range(0, len(pop_dens[i])):
        surface_data_point = np.array([[grid_x[i], grid_y[j], pop_dens[i, j]]])
        surface_data=np.concatenate([surface_data, surface_data_point])
surface_data = surface_data[1:, :]
print(surface_data)

pop_dens_x = np.array([[0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.5],
                    [100, 75, 75, np.average([100, 100, 75, 50]), np.average([100, 100, 75, 50]), np.average([0, 0, 50, 100]), np.average([0, 25, 50, 25]), np.average([30, 20, 0, 100]), np.average([100, 25, 100, 50]), np.average([100, 100, 100, 0]), 0]])
poly_x = np.polyfit(pop_dens_x[0], pop_dens_x[1], deg=8)
poly_y = np.polyfit(np.array([0, 0.25, 0.75, 1.25, 1.75, 2.0]), np.array([np.average(pop_dens[:,0]), np.average(pop_dens[:,1]), np.average(pop_dens[:,2]), np.average(pop_dens[:,3]), np.average(pop_dens[:,4]), np.average(pop_dens[:,5])]), deg=5)
poly_y = [1/384, 0, -1/8, 0, 1]  #Taylor expansion of z = cos(y/2)
plt.figure('Distribution')
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(0, 4.51, 0.01)
Y = np.arange(0, 2.01, 0.01)
X, Y = np.meshgrid(X, Y)
#Z = (poly_x[0]*X**8 + poly_x[1]*X**7 + poly_x[2]*X**6 + poly_x[3]*X**5 + poly_x[4]*X**4 + poly_x[5]*X**3 + poly_x[6]*X**2 + poly_x[7]*X**1 + poly_x[8]*X**0) *\
#    (poly_y[0]*Y**5 +  poly_y[1]*Y**4 +poly_y[2]*Y**3 +poly_y[3]*Y**2 +poly_y[4]*Y**1 + poly_y[5]*Y**0)/100
Z = (poly_x[0]*X**8 + poly_x[1]*X**7 + poly_x[2]*X**6 + poly_x[3]*X**5 + poly_x[4]*X**4 + poly_x[5]*X**3 + poly_x[6]*X**2 + poly_x[7]*X**1 + poly_x[8]*X**0) *\
    (poly_y[0]*Y**4 + poly_y[1]*Y**3 + poly_y[2]*Y**2 + poly_y[3]*Y**1 + poly_y[4]*Y**0)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(np.arange(0, 1.1, 0.1))))
ax.set_zlim(0, 100)
ax.set_zticks(np.arange(0, 100.1, 50))
ax.set_xlabel('X [km]', rotation=0)
ax.set_ylabel('Y [km]', rotation=0)
ax.set_zlabel('%', rotation=0)

# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

d0 = np.array([[1],
               [1.9],
               [3],
               [0.5]])  # x1, y1, x2, y2
bounds = np.array([[0.1, 4.4],
                   [0, 2.0],
                   [0.1, 4.4],
                   [0, 2.0]])
def objective(d):
    pos1 = np.array([d[0], d[1]])   # station 1, 2, 3
    pos2 = np.array([d[2], d[3]])
    poscent = np.array([0, 1.0])      # centraal
    pospier = np.array([4.5, 1.0])    # pier

    # Objective 1:
    # Find the average distance to stations --> minimize for accessibility
    lower = pos1[0]/2
    upper = pos2[0] + (4.5-pos2[0])/2
    mid = (pos1[0] + pos2[0])/2
    x_int = np.polyint(poly_x)
    xx_int = np.polyint(np.append(poly_x, 0))
    y_int = np.polyint(poly_y)
    yy_int = np.polyint(np.append(poly_y, 0))
    x_avg_1 = (np.polyval(xx_int, mid) - np.polyval(xx_int, lower)) / \
              (np.polyval(x_int, mid) - np.polyval(x_int, lower))
    x_avg_2 = (np.polyval(xx_int, upper) - np.polyval(xx_int, mid)) / \
              (np.polyval(x_int, upper) - np.polyval(x_int, mid))
    y_avg_1 = (np.polyval(yy_int, 2) - np.polyval(yy_int, 0)) / \
              (np.polyval(y_int, 2) - np.polyval(y_int, 0))
    y_avg_2 = (np.polyval(yy_int, 2) - np.polyval(yy_int, 0)) / \
              (np.polyval(y_int, 2) - np.polyval(y_int, 0))
    dist_1 = ((pos1[0]-x_avg_1)**2+(pos1[1]-y_avg_1)**2)**0.5
    dist_2 = ((pos2[0]-x_avg_2)**2+(pos2[1]-y_avg_2)**2)**0.5
    crit_1 = dist_1 + dist_2
    w_1 = 0.5

    # Objective 2:
    # Minimize length of line to save cost

    x_cs = np.array([poscent[0], pos1[0], pos2[0], pospier[0]])
    y_cs = np.array([poscent[1], pos1[1], pos2[1], pospier[1]])
    cs = sp.interpolate.CubicSpline(x=x_cs, y=y_cs, bc_type=((1, 0.5), (2, 0.0)))
    step = int(0.01)
    arc_length = 0
    for k in range(0, 450, 1):
        k = k/100
        p_u = np.array([k+0.01, cs(k+0.01)])
        p_l = np.array([k, cs(k)])
        arc_length = arc_length + ((p_u[0]-p_l[0])**2 + (p_u[1]-p_l[1])**2)**0.5
    w_2 = 0.5
    return w_1 * crit_1 + w_2*arc_length

def nonlincon(d):
    pos1 = np.array([d[0], d[1]])  # station 1, 2, 3
    pos2 = np.array([d[2], d[3]])
    poscent = np.array([0, 1.0])  # centraal
    pospier = np.array([4.5, 1.0])  # pier

    # Constraint 1 --> station 2 not before station 1 in x-direction
    c0 = pos1[0] - pos2[0]+0.1

    # Constraint 2 --> turn radius not too sharp along line
    x_cs = np.array([poscent[0], pos1[0], pos2[0], pospier[0]])
    y_cs = np.array([poscent[1], pos1[1], pos2[1], pospier[1]])
    cs = sp.interpolate.CubicSpline(x=x_cs, y=y_cs, bc_type=((1, 0.5), (2, 0.0)))
    Radius_list = np.array([0])
    for i in range(0, 451, 1):
        i = i/100
        curvature1 = max((0.0001, abs(cs(i, 2))))
        curvature2 = (1+cs(i, 1)**2)**1.5
        Radius = abs(curvature2/curvature1)
        Radius_list = np.append(Radius_list, Radius)
    Radius_list = Radius_list[1:]
    R_min = min(Radius_list)
    c1 = 2 - R_min
    return np.array([c0, c1])

cons = sp.optimize.NonlinearConstraint(nonlincon, np.array([-np.inf]), np.array([0]))


result = sp.optimize.minimize(fun=objective, x0=d0, bounds=bounds, constraints=cons)
print(result)

x = result.x
print(x)
pos1 = np.array([x[0], x[1]])  # station 1, 2
pos2 = np.array([x[2], x[3]])
poscent = np.array([0, 1.0])  # centraal
pospier = np.array([4.5, 1.0])
x_cs = np.array([poscent[0], pos1[0], pos2[0], pospier[0]])
y_cs = np.array([poscent[1], pos1[1], pos2[1], pospier[1]])
cs = sp.interpolate.CubicSpline(x=x_cs, y=y_cs, bc_type=((1, 0.5), (2, 0.0)))
plt.figure('Design')
plt.plot(np.arange(0, 4.51, 0.01), cs(np.arange(0, 4.51, 0.01)))
plt.plot(np.arange(0, 4.51, 0.01), abs((1+cs(np.arange(0, 4.51, 0.01), 1)**2)**1.5/cs(np.arange(0, 4.51, 0.01), 1)))
plt.grid()
plt.xlim([0, 4.5])
plt.ylim([0, 2])
plt.scatter(x_cs, y_cs)
plt.show()
