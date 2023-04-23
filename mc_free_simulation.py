from bmct_fun import *

# število obravnavanih poligonov (pri 7 dobimo enakomerno porazdelitev tudi za monte carlo)
# pri 20 ali vec se porazdelitev za monte carlo ne spreminja vec bistveno
dim = 25
# število celic
N = 1000000
diff = 1/N
# čas simulacije (število)
t = 4*10**8

dim_s = np.arange(10, 30, 1)
dim_res = np.zeros_like(dim_s, dtype=float)

# zacetni pogoj (mreza sestkotnika)
init_vec = np.zeros((dim))
init_vec[3] = 1

for i in range(len(dim_s)):
	dim = dim_s[i]
	init_vec = np.zeros((dim))
	init_vec[3] = 1
	t_mc = walk_free(init_vec, t, diff)
	x_s = np.arange(1, dim+1, 1)
	y_s = np.log(t_mc)
	m,b = np.polyfit(x_s, y_s, 1)
	print(m)
	dim_res[i] = m

print(dim_res.tolist())

plt.plot(dim_s, dim_res)
plt.show()

# izracun markovske verige
start= time.time()
t_prop = propag_free(init_vec, t, diff*4)
print(time.time()-start)


# izracun monte carlo
start= time.time()
t_mc = walk_free(init_vec, t, diff)
print(time.time()-start)

# izracun naklona koncne porazdelitve pri monte carlo, ta je kot kazejo rezultati eksponentna
x_s = np.arange(1, dim+1, 1)
y_s = np.log(t_mc)
m,b = np.polyfit(x_s, y_s, 1)
print('naklon', m, 'zacetna vrednost', b)

# plot zacetne in koncne porazdelitve
fig, ax = plt.subplots(2, sharex=True)
x_s = np.arange(1, dim+1, 1)
ax[0].bar(x_s, t_mc,label='Monte Carlo stacionarno') # plot koncne porazdelitve
ax[0].plot(x_s, np.exp(b)*np.exp(m*x_s), 'r--') # plot fita
ax[0].legend()
ax[1].bar(x_s, t_prop, label='Markov chain stacinarno') # plot zacente
ax[1].legend()
ax[0].set_yscale('log')
ax[0].set_ylabel(r'$p_i(t_\infty)$')
ax[1].set_ylabel(r'$p_i(t_\infty)$')
ax[1].set_xticks(x_s)
ax[1].set_xticklabels(x_s+3)
plt.show()

# todo nareid graf naklona v odvisnosti od dimnenzije