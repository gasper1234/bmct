from bmct_fun import *

vals = [-0.1928535593545851, -0.218757671308439, -0.23670637521496568, -0.24931746228667812, -0.25929561229934855, -0.2657628210384682, -0.2714322694116054, -0.2748925700131557, -0.27749519275233464, -0.28002744396730367, -0.2815600803382501, -0.28247537671650624, -0.28479873384548093, -0.28516185496449603, -0.28444024825940467, -0.2850419520933513, -0.28783323835770724, -0.2874688580731728, -0.28640272556845126, -0.28638080424062007]

dim_s = np.arange(10, 30, 1)
dim_res = np.zeros_like(dim_s, dtype=float)

vals_comp = []

for i in range(len(dim_s)):
	vals_comp.append(bisec_lamb(dim_s[i]))

print(1/abs(bisec_lamb(10000)))

print(vals_comp)

plt.plot(dim_s, vals_comp, label='numerično')
plt.plot(dim_s, vals, 'rx', label='MC')
plt.legend()
plt.xlabel('m')
plt.ylabel(r'$\lambda$')
plt.xticks(dim_s[::2])
plt.show()
