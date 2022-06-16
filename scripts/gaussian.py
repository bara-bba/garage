import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from numpy import genfromtxt

ur_force = genfromtxt("/home/bara/Dropbox/Camozzi-POLIMI project weekly meetings/Lorenzo Barattin/ur5_force.csv",
                delimiter=',')
# panda = genfromtxt("/home/bara/PycharmProjects/garage/panda_force.csv",
#                    delimiter=',')

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def estimate(x, n):
    mean = sum(x)/n
    sigma = sum((x - mean) ** 2)/n
    return mean, sigma

ur0 = ur_force[:, 1]
ur1 = ur_force[:, 2]
ur2 = ur_force[:, 3]
ur3 = ur_force[:, 4]
ur4 = ur_force[:, 5]
ur5 = ur_force[:, 6]

# panda0 = panda[:, 0]
# panda1 = panda[:, 1]
# panda2 = panda[:, 2]

n0, bins0, patches0 = plt.hist(ur0, bins=80)
n1, bins1, patches1 = plt.hist(ur1, bins=80)
n2, bins2, patches2 = plt.hist(ur2, bins=80)
n3, bins3, patches3 = plt.hist(ur3, bins=80)
n4, bins4, patches4 = plt.hist(ur4, bins=80)
n5, bins5, patches5 = plt.hist(ur5, bins=80)

# pn0, pbins0, ppatches0 = plt.hist(panda0, bins=100)
# pn1, pbins1, ppatches1 = plt.hist(panda1, bins=100)
# pn2, pbins2, ppatches2 = plt.hist(panda2, bins=100)
plt.close()

mean0, sigma0 = estimate(bins0[:-1], bins0.size-1)
mean1, sigma1 = estimate(bins1[:-1], bins1.size-1)
mean2, sigma2 = estimate(bins2[:-1], bins2.size-1)
mean3, sigma3 = estimate(bins3[:-1], bins3.size-1)
mean4, sigma4 = estimate(bins4[:-1], bins4.size-1)
mean5, sigma5 = estimate(bins5[:-1], bins5.size-1)

# pmean0, psigma0 = estimate(pbins0[:-1], pbins0.size-1)
# pmean1, psigma1 = estimate(pbins1[:-1], pbins1.size-1)
# pmean2, psigma2 = estimate(pbins2[:-1], pbins2.size-1)


popt0, pcov0 = curve_fit(gaussian, bins0[:-1], n0, p0=[1, mean0, sigma0])
popt1, pcov1 = curve_fit(gaussian, bins1[:-1], n1, p0=[1, mean1, sigma1])
popt2, pcov2 = curve_fit(gaussian, bins2[:-1], n2, p0=[1, mean2, sigma2])
popt3, pcov3 = curve_fit(gaussian, bins3[:-1], n3, p0=[1, mean3, sigma3])
popt4, pcov4 = curve_fit(gaussian, bins4[:-1], n4, p0=[1, mean4, sigma4])
popt5, pcov5 = curve_fit(gaussian, bins5[:-1], n5, p0=[1, mean5, sigma5])

# ppopt0, ppcov0 = curve_fit(gaussian, pbins0[:-1], pn0, p0=[1, pmean0, psigma0])
# ppopt1, ppcov1 = curve_fit(gaussian, pbins1[:-1], pn1, p0=[1, pmean1, psigma1])
# ppopt2, ppcov2 = curve_fit(gaussian, pbins2[:-1], pn2, p0=[1, pmean2, psigma2])


fitted0 = gaussian(bins0[:-1], popt0[0], popt0[1], popt0[2])
fitted1 = gaussian(bins1[:-1], popt1[0], popt1[1], popt1[2])
fitted2 = gaussian(bins2[:-1], popt2[0], popt2[1], popt2[2])
fitted3 = gaussian(bins3[:-1], popt3[0], popt3[1], popt3[2])
fitted4 = gaussian(bins4[:-1], popt4[0], popt4[1], popt4[2])
fitted5 = gaussian(bins5[:-1], popt5[0], popt5[1], popt5[2])

# pfitted0 = gaussian(pbins0[:-1], ppopt0[0], ppopt0[1], ppopt0[2])
# pfitted1 = gaussian(pbins1[:-1], ppopt1[0], ppopt1[1], ppopt1[2])
# pfitted2 = gaussian(pbins2[:-1], ppopt2[0], ppopt2[1], ppopt2[2])

fig, ax = plt.subplots(2, 2, figsize=(15, 24))
hist_setting = dict(alpha=0.5, bins=80)

plt.suptitle("UR5 Noise Distribution")

# Force Noise
plt.subplot(221)
plt.plot(bins0[:-1], fitted0, label="x-estimation", color="#d45769")
plt.hist(ur0, **hist_setting, label="x-data", color="#d45769")
plt.plot(bins1[:-1], fitted1, label="y-estimation", color="#e69d45")
plt.hist(ur1, **hist_setting, label="y-data", color="#e69d45")
plt.plot(bins2[:-1], fitted2, label="z-estimation", color="#308695")
plt.hist(ur2, **hist_setting, label="z-data", color="#308695")
plt.xlim(-1.25, 1)
plt.legend()


plt.subplot(222)
plt.plot(bins0[:-1], fitted0/np.sqrt(2*np.pi*(popt0[2]**2))/popt0[0], label=u"\u03bc = " + f"{popt0[1]:.3f} \n" + u"\u03c3 = " + f"{popt0[2]:.3f}", color="#d45769")
plt.plot(bins1[:-1], fitted1/np.sqrt(2*np.pi*(popt1[2]**2))/popt1[0], label=u"\u03bc = " + f"{popt1[1]:.3f} \n" + u"\u03c3 = " + f"{popt1[2]:.3f}", color="#e69d45")
plt.plot(bins2[:-1], fitted2/np.sqrt(2*np.pi*(popt2[2]**2))/popt2[0], label=u"\u03bc = " + f"{popt2[1]:.3f} \n" + u"\u03c3 = " + f"{popt2[2]:.3f}", color="#308695")
plt.xlim(-1.25, 1)
plt.legend()

# Torque Noise
plt.subplot(223)
plt.plot(bins3[:-1], fitted3, label="rx-estimation", color="#d45769")
plt.hist(ur3, **hist_setting, label="rx-data", color="#d45769")
plt.plot(bins4[:-1], fitted4, label="ry-estimation", color="#e69d45")
plt.hist(ur4, **hist_setting, label="ry-data", color="#e69d45")
plt.plot(bins5[:-1], fitted5, label="rz-estimation", color="#308695")
plt.hist(ur5, **hist_setting, label="rz-data", color="#308695")
plt.legend()

plt.subplot(224)
plt.plot(bins3[:-1], fitted3/np.sqrt(2*np.pi*(popt3[2]**2))/popt3[0], label=u"\u03bc = " + f"{popt3[1]:.3f} \n" + u"\u03c3 = " + f"{popt3[2]:.3f}", color="#d45769")
plt.plot(bins4[:-1], fitted4/np.sqrt(2*np.pi*(popt4[2]**2))/popt4[0], label=u"\u03bc = " + f"{popt4[1]:.3f} \n" + u"\u03c3 = " + f"{popt4[2]:.3f}", color="#e69d45")
plt.plot(bins5[:-1], fitted5/np.sqrt(2*np.pi*(popt5[2]**2))/popt5[0], label=u"\u03bc = " + f"{popt5[1]:.3f} \n" + u"\u03c3 = " + f"{popt5[2]:.3f}", color="#308695")
plt.legend()

hist_setting = dict(alpha=0.5, bins=100)

# # Force Noise
# plt.subplot(325)
# plt.plot(pbins0[:-1], pfitted0, label="x-estimation", color="#d45769")
# plt.hist(panda0, **hist_setting, label="x-data", color="#d45769")
# plt.plot(pbins1[:-1], pfitted1, label="y-estimation", color="#e69d45")
# plt.hist(panda1, **hist_setting, label="y-data", color="#e69d45")
# plt.plot(pbins2[:-1], pfitted2, label="z-estimation", color="#308695")
# plt.hist(panda2, **hist_setting, label="z-data", color="#308695")
# plt.xlim(-1.25, 1)
# plt.legend()
#
# plt.subplot(326)
# plt.plot(pbins0[:-1], pfitted0/np.sqrt(2*np.pi*(ppopt0[2]**2))/ppopt0[0], label=u"\u03bc = " + f"{ppopt0[1]:.3f} \n" + u"\u03c3 = " + f"{ppopt0[2]:.3f}", color="#d45769")
# plt.plot(pbins1[:-1], pfitted1/np.sqrt(2*np.pi*(ppopt0[2]**2))/ppopt1[0], label=u"\u03bc = " + f"{ppopt1[1]:.3f} \n" + u"\u03c3 = " + f"{ppopt1[2]:.3f}", color="#e69d45")
# plt.plot(pbins2[:-1], pfitted2/np.sqrt(2*np.pi*(ppopt0[2]**2))/ppopt2[0], label=u"\u03bc = " + f"{ppopt2[1]:.3f} \n" + u"\u03c3 = " + f"{ppopt2[2]:.3f}", color="#308695")
# plt.xlim(-1.25, 1)
# plt.legend()

plt.show()

