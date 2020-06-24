
import numpy as np
import scipy as sp

a = np.array([[[[[0.2173,0.2240,0.2824,0.2763],[1.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0]]]]])
tau = np.array([[[[[1000000,394.4,36.54,4.304],[9.15,1.0,1.0,1.0],[116.0,1.0,1.0,1.0]]]]])
r = np.array([[[[[28.627296,0.019773,4.334433,0.0],[9.078874,0.0,-0.287247,0.000343],[67.843356,0.0,0.0,-0.000999]]]]])
PI_conc = np.array([[[[278.0,733.822081,271.258492]]]])
emis2conc = np.array([[[[0.468952,0.351714,0.200980]]]])
f = np.array([[[[[5.754389,0.001215,-0.069598],[0.061736,-0.000049,0.038416],[-0.054407,0.000157,0.106208]]]]])

d = np.array([[[[283.0,9.88,0.85]]]])
q = np.array([[[[0.311333,0.165417,0.242]]]])

def calculate_alpha(G,G_A,T,r,g0,g1,iirf100_max = 97.0):

	iirf100_val = r[...,0] + r[...,1] * (G-G_A) + r[...,2] * T + r[...,3] * G_A

	iirf100_val = np.abs(iirf100_val)

	iirf100_val = (iirf100_val>iirf100_max) * iirf100_max + iirf100_val * (iirf100_val<iirf100_max)

	alpha_val = g0 * np.sinh(iirf100_val / g1)

	return alpha_val

def step_concentration(R_old,E,alpha,a,tau,PI_conc,emis2conc,dt=1):

	R_new = E * emis2conc[...,np.newaxis] * a * alpha * (tau/dt) * ( 1. - np.exp( -dt/(alpha*tau) ) ) + R_old * np.exp( -dt/(alpha * tau) )

	C = PI_conc + np.sum(R_new + R_old,axis=-1) / 2

	G_A = np.sum(R_new,axis=-1) / emis2conc

	return C,R_new,G_A

def unstep_concentration(R_old,C,alpha,a,tau,PI_conc,emis2conc,dt=1):

	E = ( C - PI_conc - np.sum(R_old * np.exp( -dt/(alpha*tau) ) , axis=-1 ) ) / ( emis2conc * np.sum( a * alpha * ( tau / dt ) * ( 1. - np.exp( -dt / ( alpha * tau ) ) ) , axis=-1 ) )

	R_new = E[...,np.newaxis] * emis2conc[...,np.newaxis] * a * alpha * (tau/dt) * ( 1. - np.exp( -dt/(alpha*tau) ) ) + R_old * np.exp( -dt/(alpha * tau) )

	G_A = np.sum(R_new,axis=-1) / emis2conc

	return E,R_new,G_A

def step_forcing(C,PI_conc,f):

	RF = f[...,0] * np.log( C/PI_conc ) + f[...,1] * ( C - PI_conc ) + f[...,2] * ( np.sqrt(C) - np.sqrt(PI_conc) )

	return RF

def step_temperature(S_old,F,q,d,dt=1):

	S_new = q * F * ( 1 - np.exp(-dt/d) ) + S_old * np.exp(-dt/d)

	T = np.sum(S_old + S_new,axis=-1) / 2

	return S_new,T

def GIR_model(emissions = False, concentrations = False, forcing = False, a=a, tau=tau, r=r, PI_conc=PI_conc, emis2conc=emis2conc, f=f, d=d, q=q, dim_scens=1, dim_gas_param_sets=1, dim_thermal_param_sets=1, dim_gases=3, timestep=1.0):

	# model requires emissions array or concentrations array even if running forcing timeseries only. 
	# this is because model needs an estimate of the PI concentrations which are supplied in the conc. array, unless emissions driven.

	if type(emissions) == bool:
		concentration_driven = True
		emissions = np.zeros_like(concentrations)
	else:
		concentration_driven = False

	n_year = emissions.shape[-1]

	g1 = np.sum( a * tau * ( 1. - ( 1. + 100/tau ) * np.exp(-100/tau) ), axis=-1 )
	g0 = ( np.sinh( np.sum( a * tau * ( 1. - np.exp(-100/tau) ) , axis=-1) / g1 ) )**(-1.)

	if type(forcing) is bool:
		ext_forcing = np.zeros((dim_scens,dim_gas_param_sets,dim_thermal_param_sets,1,n_year))
	else:
		ext_forcing = forcing

	C = np.zeros((dim_scens,dim_gas_param_sets,dim_thermal_param_sets,dim_gases,n_year))
	RF = np.zeros((dim_scens,dim_gas_param_sets,dim_thermal_param_sets,dim_gases,n_year))
	T = np.zeros((dim_scens,dim_gas_param_sets,dim_thermal_param_sets,n_year))
	alpha = np.zeros((dim_scens,dim_gas_param_sets,dim_thermal_param_sets,dim_gases,n_year))
	alpha[...,0] = calculate_alpha(G=np.zeros(C[...,0].shape),G_A=np.zeros(C[...,0].shape),T=np.zeros(C[...,0].shape),r=r,g0=g0,g1=g1)

	if concentration_driven:

		diagnosed_emissions = np.zeros((dim_scens,dim_gas_param_sets,dim_thermal_param_sets,dim_gases,n_year))
		C = concentrations
		C_end = np.zeros(C.shape)
		RF[:] = step_forcing(C,PI_conc[...,np.newaxis],f[...,np.newaxis,:])
		C_end[...,0] = C[...,0]*2 - PI_conc
		diagnosed_emissions[...,0],R,G_A = unstep_concentration(R_old=np.zeros(a.shape),C=C_end[...,0],alpha=alpha[...,0,np.newaxis],a=a,tau=tau,PI_conc=PI_conc,emis2conc=emis2conc,dt=timestep)
		S,T[...,0] = step_temperature(S_old=np.zeros(d.shape),F=np.sum(RF[...,0],axis=-1)[...,np.newaxis]+ext_forcing[...,0],q=q,d=d,dt=timestep)
		for t in np.arange(1,n_year):
			G = np.sum(diagnosed_emissions,axis=-1)
			alpha[...,t] = calculate_alpha(G=G,G_A=G_A,T=np.sum(S,axis=-1)[...,np.newaxis],r=r,g0=g0,g1=g1)
			C_end[...,t] = C[...,t]*2 - C_end[...,t-1]
			diagnosed_emissions[...,t],R,G_A = unstep_concentration(R_old=R,C=C_end[...,t],alpha=alpha[...,t,np.newaxis],a=a,tau=tau,PI_conc=PI_conc,emis2conc=emis2conc,dt=timestep)
			S,T[...,t] = step_temperature(S_old=S,F=np.sum(RF[...,t],axis=-1)[...,np.newaxis]+ext_forcing[...,t],q=q,d=d,dt=timestep)

		C_out = concentrations
		E_out = diagnosed_emissions

	if not concentration_driven:
		G = np.cumsum(emissions,axis=-1)
		C[...,0],R,G_A = step_concentration(R_old = np.zeros(a.shape),alpha=alpha[...,0,np.newaxis],E=emissions[...,0,np.newaxis],a=a,tau=tau,PI_conc=PI_conc,emis2conc=emis2conc,dt=timestep)
		RF[...,0] = step_forcing(C=C[...,0],PI_conc=PI_conc,f=f)
		S,T[...,0] = step_temperature(S_old=np.zeros(d.shape),F=np.sum(RF[...,0],axis=-1)[...,np.newaxis]+ext_forcing[...,0],q=q,d=d,dt=timestep)

		for t in np.arange(1,n_year):
			alpha[...,t] = calculate_alpha(G=G[...,t-1],G_A=G_A,T=np.sum(S,axis=-1)[...,np.newaxis],r=r,g0=g0,g1=g1)
			C[...,t],R,G_A = step_concentration(R_old = R,alpha=alpha[...,t,np.newaxis],E=emissions[...,t,np.newaxis],a=a,tau=tau,PI_conc=PI_conc,emis2conc=emis2conc,dt=timestep)
			RF[...,t] = step_forcing(C=C[...,t],PI_conc=PI_conc,f=f)
			S,T[...,t] = step_temperature(S_old=S,F=np.sum(RF[...,t],axis=-1)[...,np.newaxis]+ext_forcing[...,t],q=q,d=d,dt=timestep)

		C_out = C
		E_out = emissions

	ext_forcing = np.zeros(np.sum(RF,axis=-2)[...,np.newaxis,:].shape) + ext_forcing
	RF = np.concatenate((RF,ext_forcing),axis=-2)
	RF = np.concatenate((RF,np.sum(RF,axis=-2)[...,np.newaxis,:]),axis=-2)
        
	alpha_out = alpha
	RF_out = RF
	T_out = T

	print('GIR run over ', dim_scens, ' scenario, ', dim_gas_param_sets, ' gas parameter sets, ', dim_thermal_param_sets, ' thermal parameter sets, ', dim_gases, ' gases, and ', n_year, ' years...')
	return E_out, C_out, RF_out, T_out, alpha_out


def make_input_dimensions_test(input_array, num_scens, num_gas_params, num_thermal_params, num_gases, num_years):

	fail_bool = False

	# input array
	if np.array(input_array.shape[:-1]).size == np.array([num_scens, num_gas_params, num_thermal_params, num_gases]).size:
		print('same')

	if np.array(input_array.shape[:-1]).size == np.array([num_scens, num_gas_params, num_thermal_params, num_gases]).size:
		for i in range(0,np.array(input_array.shape[:-1]).size):
			if np.array(input_array.shape[:-1])[i] == np.array([num_scens, num_gas_params, num_thermal_params, num_gases])[i]:
				pass
			else:
				print('====================================================')
				print('FAILED: input array wrong shape')
	elif num_gases != input_array.shape[-2]:
		print('====================================================')
		print('FAILED: number of gases doesn\'t match size of input array')
		fail_bool = True
	elif (num_scens == 1) and (num_gas_params == 1) and (num_thermal_params == 1) and (np.array(input_array.shape) == np.array([num_gases,num_years])).all():
		input_array = input_array[np.newaxis, np.newaxis, np.newaxis, ...]
	else:
		print('====================================================')
		print('FAILED: Be more careful with input_array\'s shape!')
		print('Dimensions should be of form: [num_scens, num_gas_params, num_thermal_params, num_gases, num_years]')
		fail_bool = True
    
	if fail_bool == False:
		print('input_array outputted in form [num_scens, num_gas_params, num_thermal_params, num_gases, num_years]...')	
		return input_array
	else:
		return input_array

    
def make_param_dimensions(a, tau, r, PI_conc, emis2conc, f, d, q, num_scens, num_gas_params, num_thermal_params, num_gases, num_therm_boxes):

	fail_bool = False

	# a array
	if (np.array(a.shape[:-1]) == np.array([num_scens, num_gas_params, num_thermal_params, num_gases])).all():
		pass
	elif num_gases != a.shape[-2]:
		print('====================================================')
		print('FAILED: number of gases doesn\'t match size of \'a\' array')
		fail_bool = True
	elif (num_scens == 1) and (num_gas_params == 1) and (num_thermal_params == 1) and (np.array(a.shape) == np.array([num_gases,4])).all():
		a = a[np.newaxis, np.newaxis, np.newaxis, ...]
	else:
		print('====================================================')
		print('FAILED: Be more careful with a\'s input array shape!')
		print('Dimensions should be of form: [num_scens, num_gas_params, num_thermal_params, num_gases, 4]')
		fail_bool = True

	# tau array
	if (np.array(tau.shape[:-1]) == np.array([num_scens, num_gas_params, num_thermal_params, num_gases])).all():
		pass
	elif num_gases != tau.shape[-2]:
		print('====================================================')
		print('FAILED: number of gases doesn\'t match size of \'tau\' array')
		fail_bool = True
	elif (num_scens == 1) and (num_gas_params == 1) and (num_thermal_params == 1) and (np.array(tau.shape) == np.array([num_gases,4])).all():
		tau = tau[np.newaxis, np.newaxis, np.newaxis, ...]
	else:
		print('====================================================')
		print('FAILED: Be more careful with tau\'s input array shape!')
		print('Dimensions should be of form: [num_scens, num_gas_params, num_thermal_params, num_gases, 4]')
		fail_bool = True

	# r array
	if (np.array(r.shape[:-1]) == np.array([num_scens, num_gas_params, num_thermal_params, num_gases])).all():
		pass
	elif num_gases != r.shape[-2]:
		print('====================================================')
		print('FAILED: number of gases doesn\'t match size of \'r\' array')
		fail_bool = True
	elif (num_scens == 1) and (num_gas_params == 1) and (num_thermal_params == 1) and (np.array(r.shape) == np.array([num_gases,4])).all():
		r = r[np.newaxis, np.newaxis, np.newaxis, ...]
	else:
		print('====================================================')
		print('FAILED: Be more careful with r\'s input array shape!')
		print('Dimensions should be of form: [num_scens, num_gas_params, num_thermal_params, num_gases, 4]')
		fail_bool = True

	# PI_conc array
	if (np.array(PI_conc.shape) == np.array([num_scens, num_gas_params, num_thermal_params, num_gases])).all():
		pass
	elif num_gases != PI_conc.shape[-1]:
		print('====================================================')
		print('FAILED: number of gases doesn\'t match size of \'PI_conc\' array')
		fail_bool = True
	elif (num_scens == 1) and (num_gas_params == 1) and (num_thermal_params == 1) and (np.array(PI_conc.shape) == np.array([num_gases])).all():
		PI_conc = PI_conc[np.newaxis, np.newaxis, np.newaxis, ...]
	else:
		print('====================================================')
		print('FAILED: Be more careful with PI_conc\'s input array shape!')
		print('Dimensions should be of form: [num_scens, num_gas_params, num_thermal_params, num_gases]')
		fail_bool = True

	# emis2conc array
	if (np.array(emis2conc.shape) == np.array([num_scens, num_gas_params, num_thermal_params, num_gases])).all():
		pass
	elif num_gases != emis2conc.shape[-1]:
		print('====================================================')
		print('FAILED: number of gases doesn\'t match size of \'emis2conc\' array')
		fail_bool = True
	elif (num_scens == 1) and (num_gas_params == 1) and (num_thermal_params == 1) and (np.array(emis2conc.shape) == np.array([num_gases])).all():
		emis2conc = emis2conc[np.newaxis, np.newaxis, np.newaxis, ...]
	else:
		print('====================================================')
		print('FAILED: Be more careful with emis2conc\'s input array shape!')
		print('Dimensions should be of form: [num_scens, num_gas_params, num_thermal_params, num_gases]')
		fail_bool = True

	# f array
	if (np.array(f.shape[:-1]) == np.array([num_scens, num_gas_params, num_thermal_params, num_gases])).all():
		pass
	elif num_gases != f.shape[-2]:
		print('====================================================')
		print('FAILED: number of gases doesn\'t match size of \'f\' array')
		fail_bool = True
	elif (num_scens == 1) and (num_gas_params == 1) and (num_thermal_params == 1) and (np.array(f.shape) == np.array([num_gases,3])).all():
		f = f[np.newaxis, np.newaxis, np.newaxis, ...]
	else:
		print('====================================================')
		print('FAILED: Be more careful with f\'s input array shape!')
		print('Dimensions should be of form: [num_scens, num_gas_params, num_thermal_params, num_gases, 3]')
		fail_bool = True

	# d array
	if (np.array(d.shape) == np.array([num_scens, num_gas_params, num_thermal_params, num_gases])).all():
		pass
	elif num_therm_boxes != d.shape[-1]:
		print('====================================================')
		print('FAILED: number of gases doesn\'t match size of \'d\' array')
		fail_bool = True
	elif (num_scens == 1) and (num_gas_params == 1) and (num_thermal_params == 1) and (np.array(d.shape) == np.array([num_therm_boxes])).all():
		d = d[np.newaxis, np.newaxis, np.newaxis, ...]
	else:
		print('====================================================')
		print('FAILED: Be more careful with d\'s input array shape!')
		print('Dimensions should be of form: [num_scens, num_gas_params, num_thermal_params, num_gases]')
		fail_bool = True

	# q array
	if (np.array(q.shape) == np.array([num_scens, num_gas_params, num_thermal_params, num_gases])).all():
		pass
	elif num_therm_boxes != q.shape[-1]:
		print('====================================================')
		print('FAILED: number of gases doesn\'t match size of \'q\' array')
		fail_bool = True
	elif (num_scens == 1) and (num_gas_params == 1) and (num_thermal_params == 1) and (np.array(q.shape) == np.array([num_therm_boxes])).all():
		q = q[np.newaxis, np.newaxis, np.newaxis, ...]
	else:
		print('====================================================')
		print('FAILED: Be more careful with q\'s input array shape!')
		print('Dimensions should be of form: [num_scens, num_gas_params, num_thermal_params, num_gases]')
		fail_bool = True


	if fail_bool == False:
		print('Parameters outputted in form [num_scens, num_gas_params, num_thermal_params, num_gases]...')	
		return a, tau, r, PI_conc, emis2conc, f, d, q
	else:
		return a, tau, r, PI_conc, emis2conc, f, d, q


