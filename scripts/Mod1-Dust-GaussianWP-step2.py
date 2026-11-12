import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.special import erfc
from scipy.special import hyp1f1
from scipy.special import gamma

#################################
#Plot the 3d wave function without normalization

# Make sure this data is the same as in step 1
hi=0.01; hf=4; Nh=800
hdom=np.linspace(hi,hf,Nh)
dh=hdom[1]-hdom[0]

# T partition
ti=-20; tf=0; Nt=900; 
tdom=np.linspace(ti,tf,Nt)

xiin=0; xifin=8; Nx=700 #600
xidom=np.linspace(xiin,xifin,Nx)
dx=xidom[1]-xidom[0]


# Load square amplitude without normalization and make a 3d plot
final= np.load("final.npy") 
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(tdom, hdom)
ax.plot_surface(X, Y, final) #
ax.set_xlabel(r"$T$", fontsize=16, fontweight='bold', color="blue")
ax.set_ylabel(r"$H$", fontsize=16, fontweight='bold', color="blue")
ax.set_title(r"$|\Psi|^2$", fontsize=16, fontweight='bold', color="blue")
ax.set_box_aspect((1, 1.2, 0.5))
plt.show()

################################
### Normalizar en el tiempo
norte=[np.nan for el in tdom]

for n in range(Nt):
    suma=0
    for m in range(Nh):
        suma+=final[m][n]*dh
    norte[n]=suma

###Plot a snapshot
b=0
c=0

wavenor=[[final[m][n]/norte[n] for n in range(Nt-b)] for m in range(Nh-c)]
normalized=np.array(wavenor)
tdom2=[tdom[n] for n in range(Nt-b)]
hdom2=[hdom[m] for m in range(Nh-c)]

###Plot 3d
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(tdom2, hdom2)
#ax.plot_surface(X, Y, normalized, cmap="cividis")
ax.plot_wireframe(X, Y, normalized, color="black") # rcount=60, ccount=60,
ax.set_xlabel(r"$T$", fontsize=16, fontweight='bold', color="blue")
ax.set_ylabel(r"$H$", fontsize=16, fontweight='bold', color="blue")
ax.set_title(r"$|\psi|^2$", fontsize=16, fontweight='bold', color="blue")
ax.set_box_aspect((1, 1.2, 0.5))
plt.show()

####################################
#Profiles at different times
n0=0
n1=200
n2=400
n3=600
n4=850
n5=899

p0=[normalized[m][n0] for m in range(Nh)]
p1=[normalized[m][n1] for m in range(Nh)]
p2=[normalized[m][n2] for m in range(Nh)]
p3=[normalized[m][n3] for m in range(Nh)]
p4=[normalized[m][n4] for m in range(Nh)]
p5=[normalized[m][n5] for m in range(Nh)]


plt.figure()
plt.plot(hdom, p0, label=r"$T_0=$" f"{tdom[n0]:.1f}")
plt.plot(hdom, p1, label=r"$T_1=$" f"{tdom[n1]:.1f}")
plt.plot(hdom, p2, label=r"$T_2=$" f"{tdom[n2]:.1f}")
plt.plot(hdom, p3, label=r"$T_3=$" f"{tdom[n3]:.1f}")
plt.plot(hdom, p4, label=r"$T_4=$" f"{tdom[n4]:.1f}")
plt.plot(hdom, p5, label=r"$T_5=$" f"{tdom[n5]:.1f}")
#plt.ylim(0,1.3)
plt.title(r'$|\psi\, (T_i,H)|^2$', fontsize=16, fontweight='bold')
plt.xlabel(r'$H$', fontsize=16, fontweight='bold')
#plt.grid()
plt.legend()
plt.show()

#####################################
## Calcular el valor medio de H 
mv=[np.nan for el in tdom]
for n in range(Nt):
    suma=0
    for m in range(Nh):
        suma+=hdom[m]*final[m][n]*dh
    mv[n]=suma/norte[n]

plt.figure()
plt.plot(tdom, mv)
plt.xlabel(r'$T$')
plt.title(r'$<H(T)>$')
plt.grid()
plt.show()
