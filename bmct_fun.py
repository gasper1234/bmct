import matplotlib.pyplot as plt
import numpy as np
import time
from numba import njit

# monte carlo del simulacije, za pospesitev kode je uporabljena numba

@njit # poisce nakljucno vrednost med 0 in 1, ki je manjsa od val
def random_less(val):
	r_val = np.random.random()
	while r_val < val:
		r_val = np.random.random()
	return r_val

@njit # poisce nakljucno vrednost med 0 in 1, ki je vecja od val
def random_more(val):
	r_val = np.random.random()
	while r_val > val:
		r_val = np.random.random()
	return r_val

@njit # poisce pozicijo v vektorju p, ki pripada nakljucenmu stevilu
def find_pos(vec, val):
	acum = vec[0]
	for i in range(len(vec)):
		if val <= acum:
			return i
		else:
			acum += vec[i+1]

@njit # 1 korak Monte Carlo
def step(vec, diff):
	vec_1 = np.copy(vec)
	# izbira pozicij v vetkorju, ki jima zmanjsamo stevilo stranic
	m1, m2 = random_less(vec[0]), random_less(vec[0])
	m_pos_1, m_pos_2 = find_pos(vec, m1), find_pos(vec, m2)
	# izbira pozicij v vetkorju, ki jima povecamo stevilo stranic
	p1, p2 = random_more(1-vec[-1]), random_more(1-vec[-1])
	p_pos_1, p_pos_2 = find_pos(vec, p1), find_pos(vec, p2)
	# zmanjsamo stevilo stranic (prestavimo celico iz p_m v p_m-1)
	vec_1[int(m_pos_1)] -= diff
	vec_1[int(m_pos_2)] -= diff
	vec_1[int(m_pos_1-1)] += diff
	vec_1[int(m_pos_2-1)] += diff
	# povecamo stevilo stranic (prestavimo celico iz p_m v p_m+1)
	vec_1[int(p_pos_1)] -= diff
	vec_1[int(p_pos_2)] -= diff
	vec_1[int(p_pos_1+1)] += diff
	vec_1[int(p_pos_2+1)] += diff
	return vec_1

@njit # iteracija korakov
def walk(t_s, diff):
	for i in range(1, len(t_s)):
		t_s[i] = step(t_s[i-1], diff)
	return t_s

@njit
def walk_free(t_0, iter, diff):
	t = np.copy(t_0)
	for i in range(iter):
		t = step(t, diff)
	return t
# markovske verige

# matrika T
@njit
def make_T(dim, dt):
	T_1 = np.diag(np.ones((dim))-dt)
	T_2 = np.diag(np.ones((dim-1))*dt/2, 1) + np.diag(np.ones((dim-1))*dt/2, -1)
	T = T_1 + T_2
	T[0, 0] = 1-dt/2
	T[dim-1, dim-1] = 1-dt/2
	return T

# casovni razvoj z matriko T
@njit
def propag(t_s, diff):
	T = make_T(len(t_s[0]), diff)
	check = len(t_s)//100
	for i in range(1, len(t_s)):
		t_s[i] = T.dot(t_s[i-1])
		if i % check == 0:
			print('norm', np.sum(t_s[i]))
	return t_s

@njit
def propag_free(t_0, iter, diff):
	T = make_T(len(t_0), diff)
	check = iter//100
	t = np.copy(t_0)
	for i in range(iter):
		t = T.dot(t)
		if i % check == 0:
			print('norm', np.sum(t))
	return t

# od tu naprej le se preizkusanje kako se obnasajo drugacne matrike T

def make_T_up(dim, dt):
	T_1 = np.diag(np.ones((dim))-dt)
	T_2 = np.diag(np.ones((dim-1))*dt, -1)
	T = T_1 + T_2
	T[dim-1, dim-1] = 1
	return T

def make_T_down(dim, dt):
	T_1 = np.diag(np.ones((dim))-dt)
	T_2 = np.diag(np.ones((dim-1))*dt, 1)
	T = T_1 + T_2
	T[0, 0] = 1
	return T


def propag_12(t_s, diff):
	T_1 = make_T_up(len(t_s[0]), diff)
	T_2 = make_T_down(len(t_s[0]), diff)
	check = len(t_s)//100
	for i in range(1, len(t_s)):
		if i % 2 == 0:
			t_s[i] = T_2.dot(t_s[i-1])
		else:
			t_s[i] = T_1.dot(t_s[i-1])
		if i % check == 0:
			print('norm', np.sum(t_s[i]))
	return t_s

def make_T_sides(dim, dt):
	sides = np.arange(3, 3+dim, 1)
	T_1 = np.diag(np.ones((dim))-sides*dt)
	T_2 = np.diag(np.ones((dim-1))*sides[1:]*dt/2, 1) + np.diag(np.ones((dim-1))*sides[:-1]*dt/2, -1)
	T = T_1 + T_2
	T[0, 0] = 1-T[1, 0]
	T[dim-1, dim-1] = 1-T[dim-2, dim-1]
	return T

def make_T_leak(dim, dt):
	T_1 = np.diag(np.ones((dim))-dt)
	T_2 = np.diag(np.ones((dim-1))*dt/2, 1) + np.diag(np.ones((dim-1))*dt/2, -1)
	T = T_1 + T_2
	T[0, 0] = 1-dt/2
	T[1, 0] = 3*dt-4
	T[dim-1, dim-1] = 1-dt/2
	return T

def propag1(t_s, diff):
	dim = len(t_s[0])
	sides = np.arange(3, 3+dim, 1)
	T = make_T(dim, diff)
	check = len(t_s)//100
	for i in range(1, len(t_s)):
		t_s[i] = T.dot(t_s[i-1])
		t_s[i, 0] += diff/dim
		t_s[i] /= np.sum(t_s[i])
		if i % check == 0:
			print('norm', np.sum(t_s[i]))
	return t_s

def mass_center(m, lamb):
	suma = 0
	suma_b = 0
	for i in range(1, m+1):
		suma += i*np.exp(lamb*i)
		suma_b += np.exp(lamb*i)
	b = 1/suma_b
	return suma*b

def bisec_lamb(m):
	min = -1
	max = 0
	mid = 0.5*(max+min)
	guess = mass_center(m, mid)
	while abs(guess-4) > 10**(-8):
		if guess > 4:
			max = mid
		else:
			min = mid
		mid = 0.5*(min+max)
		guess = mass_center(m, mid)
	return mid