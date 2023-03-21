from bmct_fun import *

dim = 10
N = 10000
diff = 1/N
t = 100000
t_s = np.zeros((t, dim))
plot_numbers = [0, 2, 3, 4]
cols = ['k', 'red', 'blue', 'orange', 'green', 'yellow']


init_vec = np.zeros((dim))
init_vec[3] = 1

t_s[0] = init_vec

# izracun propagacija
t_s_prop = propag(np.copy(t_s), diff*4)

# izracun MC

start= time.time()
t_s_mc = walk(t_s, diff)
print(time.time()-start)


# plot
for i in range(len(plot_numbers)):
	plt.plot(t_s_prop[:, plot_numbers[i]], '--', color=cols[i], label=plot_numbers[i]+3)
	plt.plot(t_s_mc[:, plot_numbers[i]], color=cols[i], alpha=0.5)
plt.xscale('log')
plt.yscale('log')
plt.ylim(0.01, 1)
plt.xlim(10, t)
plt.legend()
plt.show()