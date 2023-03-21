import matplotlib.pyplot as plt
import numpy as np
import time
from numba import njit

@njit
def random_more(val):
	r_val = np.random.random()
	while r_val < val:
		r_val = np.random.random()
	return r_val

@njit
def random_less(val):
	r_val = np.random.random()
	while r_val > val:
		r_val = np.random.random()
	return r_val

@njit
def find_pos(vec, val):
	acum = vec[0]
	for i in range(len(vec)):
		if val <= acum:
			return i
		else:
			acum += vec[i+1]

@njit
def step(vec, diff):
	vec_1 = np.copy(vec)
	m1, m2 = random_more(vec[0]), random_more(vec[0])
	p1, p2 = random_less(1-vec[-1]), random_less(1-vec[-1])
	m_pos_1, m_pos_2 = find_pos(vec, m1), find_pos(vec, m2)
	p_pos_1, p_pos_2 = find_pos(vec, p1), find_pos(vec, p2)
	vec_1[int(m_pos_1)] -= diff
	vec_1[int(m_pos_2)] -= diff
	vec_1[int(m_pos_1-1)] += diff
	vec_1[int(m_pos_2-1)] += diff
	vec_1[int(p_pos_1)] -= diff
	vec_1[int(p_pos_2)] -= diff
	vec_1[int(p_pos_1+1)] += diff
	vec_1[int(p_pos_2+1)] += diff
	return vec_1

@njit
def walk(t_s, diff):
	for i in range(1, len(t_s)):
		t_s[i] = step(t_s[i-1], diff)
	return t_s

def make_T(dim, dt):
	T_1 = np.diag(np.ones((dim))-dt)
	T_2 = np.diag(np.ones((dim-1))*dt/2, 1) + np.diag(np.ones((dim-1))*dt/2, -1)
	T = T_1 + T_2
	T[0, 0] = 1-dt/2
	T[dim-1, dim-1] = 1-dt/2
	return T

print(make_T(10, 0.1))

def propag(t_s, diff):
	T = make_T(len(t_s[0]), diff)
	check = len(t_s)//100
	for i in range(1, len(t_s)):
		t_s[i] = T.dot(t_s[i-1])
		if i % check == 0:
			print('norm', np.sum(t_s[i]))
	return t_s
