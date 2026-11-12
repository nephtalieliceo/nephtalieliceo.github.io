import numpy as np
import matplotlib.pyplot as plt

# Define inflationary model
la=10**(-6) #Coupling constant

def F(H): #F(H) function of Geometric Inflation
    return H**2+la*H**8*np.exp(H**4)
def f(H): # It is defined as F'(H)/H
    return 2+la*(8*H**6+4*H**10)*np.exp(H**4)
def W(H): # This is the function G(H)=3 F(H)/f(H).
    return 3*(H**2*np.exp(-H**4)+la*H**8)/(2*np.exp(-H**4)+la*(8*H**6+4*H**10))

##############################################

# H partition
hi=0.01; hf=4; Nh=800
hdom=np.linspace(hi,hf,Nh)
dh=hdom[1]-hdom[0]

# T partition
ti=-20; tf=0; Nt=900; 
tdom=np.linspace(ti,tf,Nt)

# Separation constant partition
xiin=0; xifin=8; Nx=700 #600
xidom=np.linspace(xiin,xifin,Nx)
dx=xidom[1]-xidom[0]

############################################

# Numerical integration of the phase
largeH=[(4/9)*el**3 for el in hdom]
smallH=[-2/(3*el) for el in hdom]

fase=[0 for el in hdom]
suma=0
for m in range(Nh):
    fase[m]=suma
    suma+=dh/W(hdom[m])

c=fase[Nh-1]-largeH[Nh-1]
fase2=[fase[m]-c for m in range(Nh)] # Correct phase with constant
print(c)
plt.figure()
plt.plot(hdom, fase2, color="blue")
plt.plot(hdom, largeH, ls="--", color="orange")
plt.plot(hdom, smallH, ls="--", color="tab:green")
plt.title(r"$-\int G^{-1}(H) dH$")
plt.show()

##########################################
## Compute eigenfunctions of xi at t=0

A = W(hdom)**(-1/2)
B = np.outer(xidom, fase2)

re0 = A[None, :] * np.cos(B)
im0 = A[None, :] * np.sin(B)

valxi=150

re=[re0[valxi][m] for m in range(Nh)]
im=[im0[valxi][m] for m in range(Nh)]
norsq=[re0[valxi][m]**2+im0[valxi][m]**2 for m in range(Nh)]

plt.figure()

plt.plot(hdom, re)
plt.plot(hdom, im)
#plt.plot(hdom, norsq)
plt.show()

##########################################
### Obtener la evolución temporal

phase = np.outer(xidom, tdom)

re = re0[:, :, None] * np.cos(phase[:, None, :]) \
   - im0[:, :, None] * np.sin(phase[:, None, :])

im = re0[:, :, None] * np.sin(phase[:, None, :]) \
   + im0[:, :, None] * np.cos(phase[:, None, :])

print("ya")
# %%
valxi=150
valt=Nt-50

re1=[re[valxi][m][valt] for m in range(Nh)]
re2=[re[valxi][m][valt-1] for m in range(Nh)]

plt.figure()

plt.plot(hdom, re1)
plt.plot(hdom, re2)
plt.show()

#########################################
### perform the integral over xi with a Gaussian distro

def mu(xi): #Gaussian distribution
    return np.exp(-(xi-4)**2/2)/np.sqrt(2*np.pi)

"""
wpRe=[[np.nan for t in tdom] for h in hdom]
wpIm=[[np.nan for t in tdom] for h in hdom]
normaSq=[[np.nan for t in tdom] for h in hdom]

for n in range(Nt):
    for m in range(Nh):
        inte=0
        inte2=0
        for l in range(Nx):
            inte+=mu(xidom[l])*re[l][m][n]*dx
            inte2+=mu(xidom[l])*im[l][m][n]*dx
        wpRe[m][n]=inte
        wpIm[m][n]=inte2
        normaSq[m][n]=inte**2+inte2**2
"""
weights = mu(xidom) * dx

wpRe = np.sum(weights[:, None, None] * re, axis=0)
wpIm = np.sum(weights[:, None, None] * im, axis=0)

normaSq = wpRe**2 + wpIm**2

np.save("WpRe.npy", wpRe)
np.save("WpIm.npy", wpIm)
np.save("final.npy", normaSq)
