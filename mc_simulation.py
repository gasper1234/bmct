from bmct_fun import *

# število obravnavanih poligonov (pri 7 dobimo enakomerno porazdelitev tudi za monte carlo)
# pri 20 ali vec se porazdelitev za monte carlo ne spreminja vec bistveno
dim = 20
# število celic
N = 100000
diff = 1/N
# čas simulacije (število)
t = 10000000
t_s = np.zeros((t, dim))

# zacetni pogoj (mreza sestkotnika)
init_vec = np.zeros((dim))
init_vec[3] = 1

t_s[0] = init_vec

# izracun markovske verige
start= time.time()
t_s_prop = propag(np.copy(t_s), diff*4)
print(time.time()-start)


# izracun monte carlo
start= time.time()
t_s_mc = walk(t_s, diff)
print(time.time()-start)


t_s = np.linspace(0, t, t/100)
# plot casovnega poteka
plot_numbers = [0, 1, 2, 3] # razvoj katerih N-kotnikov prikazemo
cols = ['k', 'red', 'blue', 'orange', 'green', 'yellow']
for i in range(len(plot_numbers)):
	plt.plot(t_s, t_s_prop[:, plot_numbers[i]][::100], '--', color=cols[i], label=plot_numbers[i]+3)
	plt.plot(t_s, t_s_mc[:, plot_numbers[i]][::100], color=cols[i], alpha=0.5)
plt.xscale('log')
plt.yscale('log')
plt.ylim(0.005, 1.05)
plt.xlim(10, t)
plt.legend()
plt.show()


# izracun naklona koncne porazdelitve pri monte carlo, ta je kot kazejo rezultati eksponentna
x_s = np.arange(1, dim+1, 1)
y_s = np.log(t_s_mc[-1])
m,b = np.polyfit(x_s, y_s, 1)
print('naklon', m, 'zacetna vrednost', b)

# plot zacetne in koncne porazdelitve
fig, ax = plt.subplots(2)
x_s = np.arange(1, dim+1, 1)
ax[0].bar(x_s, t_s_mc[-1],label='Monte Carlo stacionarno') # plot koncne porazdelitve
ax[0].plot(x_s, np.exp(b)*np.exp(m*x_s), 'rx') # plot fita
ax[0].legend()
ax[1].bar(x_s, t_s_prop[-1], label='Markov chain stacinarno') # plot zacente
ax[1].legend()
ax[0].set_yscale('log')
plt.show()