# UnFaIRv2.0 - Stuart Jenkins and Nick Leach

# structure

# 1 - Input format: Scenarios sets
	# input and output should be managed by dataframes
	# use multiindexing in funciton to give user dataframe format to input emissions timeseries.
	#     input['scen_name']['year']['gas_name'] = value

# 2 - input for parameter sets
    # use dataframe, where outer index is the parameter set number, inner index is the parameter name and column is the gas
	# potentially separate gas cycle parameters and thermal parameters since the model dimensions are different

# 3 - make into numpy array with correct dimensions
	# Dimensions : [scenario, gas params, thermal params, gas, time/gas pools]

# 4 - compute output with standard functions

# 5 - format output in nice way, connecting emissions, C, RF, T and parameter set used in nice way
	# Add help kwargs to assist users get the right input shape?

import numpy as np
import pandas as pd
import scipy as sp

def return_empty_emissions(df_to_copy=False, start_year=1765, end_year=2500, timestep=1, scen_names=[0], gases_in = ['CO2','CH4','N2O']):

	# Returns an emissions dataframe of the correct format for use in UnFaIR with the given scenario names
    
    # Note that this is inclusive of the full end year
    
    if type(df_to_copy)==pd.core.frame.DataFrame:
        
        dict_of_zeros = {}
        
        for gas in gases_in:
            dict_of_zeros[gas] = np.zeros(df_to_copy.index.size)
        
        df = pd.DataFrame(dict_of_zeros, index=df_to_copy.index)
        
        df = pd.concat([df]*df_to_copy.columns.levels[0].size, keys=df_to_copy.columns.levels[0], axis=1)
        
        df.index = df.index.rename('Year')

        df.columns = df.columns.rename(['Scenario','Gas'])
        
    else:
        
        data_size = int(np.floor((end_year+1-start_year)/timestep))
        
        dict_of_zeros = {}
        
        for gas in gases_in:
            dict_of_zeros[gas] = np.zeros(data_size)

        df = pd.DataFrame(dict_of_zeros, index=np.arange(start_year,end_year+1,timestep)+(timestep!=1)*timestep/2)

        df = pd.concat([df]*len(scen_names), keys=scen_names, axis=1)

        df.index = df.index.rename('Year')

        df.columns = df.columns.rename(['Scenario','Gas'])

    return df

def return_empty_forcing(df_to_copy=False, start_year=1765, end_year=2500, timestep=1, scen_names=[0]):
    
    if type(df_to_copy)==pd.core.frame.DataFrame:
        
        df = pd.DataFrame({'forcing':np.zeros(df_to_copy.index.size)}, index=df_to_copy.index)
        
        df = pd.concat([df]*df_to_copy.columns.levels[0].size, keys=df_to_copy.columns.levels[0], axis=1)
        
        df.index = df.index.rename('Year')

        df.columns = df.columns.rename(['Scenario','Variable'])
        
    else:
        
        data_size = int(np.floor((end_year+1-start_year)/timestep))

        df = pd.DataFrame({'forcing':np.zeros(data_size)}, index=np.arange(start_year,end_year+1,timestep)+(timestep!=1)*timestep/2)

        df = pd.concat([df]*len(scen_names), keys=scen_names, axis=1)

        df.index = df.index.rename('Year')

        df.columns = df.columns.rename(['Scenario','Variable'])

    return df

def input_to_numpy(input_df):

	# converts the dataframe input into a numpy array for calculation, dimension order = [name, gas, time/parameter]

	return input_df.values.T.reshape(input_df.columns.levels[0].size, input_df.columns.levels[1].size, input_df.index.size)

def default_gas_forcing_params():

	# returns a dataframe of default parameters in the format UnFaIR requires (pd.concat -> additional sets)

	gas_parameter_list = ['a1','a2','a3','a4','tau1','tau2','tau3','tau4','r0','rC','rT','rA','PI_conc','emis2conc','f1','f2','f3']

	gas_cycle_parameters = pd.DataFrame(columns=['CO2','CH4','N2O'],index=gas_parameter_list).apply(pd.to_numeric)

	gas_cycle_parameters.loc['a1':'a4'] = np.array([[0.2173,0.2240,0.2824,0.2763],[1,0,0,0],[1,0,0,0]]).T
	gas_cycle_parameters.loc['tau1':'tau4'] = np.array([[1000000,394.4,36.54,4.304],[9.15,1,1,1],[116.,1,1,1]]).T
	gas_cycle_parameters.loc['r0':'rA'] = np.array([[28.6273,0.019773,4.334433,0],[9.078874,0,-0.287247,0.000343],[67.843356,0,0,-0.000999]]).T
	gas_cycle_parameters.loc['PI_conc'] = np.array([278.0,733.822081,271.23849])
	gas_cycle_parameters.loc['emis2conc'] = 1/(5.148*10**18/1e18*np.array([12.,16.,28.])/28.97)
	gas_cycle_parameters.loc['f1':'f3'] = np.array([[5.754e+00, 1.215e-03, -6.96e-02],[6.17e-02, -4.94e-05, 3.84e-02],[-0.0544, 0.000157, 0.106]]).T
	
	####### NOTES ########
	
	## with no feedbacks on CH4 and N2O (rT,rA = 0), set r0 = 9.96,63.3 for CH4,N2O respectively
	
	## Old f parameters:
    ## gas_cycle_parameters.loc['f1':'f3'] = np.array([[3.172-0.063, -2.205e-03, 3.271e-01],[-0.06009, -0.0001022, 0.05197-0.00246],[1.044e-03, 8.725e-05, 1.151e-01-0.009]]).T
	## np.array([[5.78188211,0,0],[0,0,0.03895942],[0,0,0.11082109]]).T
    
    ## "added" values to f parameters are interaction effect at present day
    
	gas_cycle_parameters.loc['F_2x'] = step_forcing(2*gas_cycle_parameters.loc['PI_conc'].values,gas_cycle_parameters.loc['PI_conc'].values,gas_cycle_parameters.loc['f1':'f3'].T.values)

	gas_cycle_parameters = pd.concat([gas_cycle_parameters], keys = ['default'], axis = 1)

	gas_cycle_parameters.index = gas_cycle_parameters.index.rename('param_name')

	gas_cycle_parameters.columns = gas_cycle_parameters.columns.rename(['Gas_cycle_set','Gas'])

	return gas_cycle_parameters.apply(pd.to_numeric)

def default_gas_forcing_param_uncertainty():

	# returns a dataframe of default parameters in the format UnFaIR requires (pd.concat -> additional sets)

	gas_parameter_list = ['a1','a2','a3','a4','tau1','tau2','tau3','tau4','r0','rC','rT','rA','PI_conc','emis2conc','f1','f2','f3']

	gas_parameter_uncertainty = pd.DataFrame(columns=['CO2','CH4','N2O'],index=gas_parameter_list)

	gas_parameter_uncertainty.loc['a1':'a4'] = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]]).T
	gas_parameter_uncertainty.loc['tau1':'tau4'] = np.array([[0,0,0,0],[0.1,0,0,0],[0.08,0,0,0]]).T
	gas_parameter_uncertainty.loc['r0':'rA'] = np.array([[0.788,0.788,0.788,0],[0,0,0.15,0.13],[0,0,0,0.16]]).T
	gas_parameter_uncertainty.loc['PI_conc'] = np.array([0,0,0])
	gas_parameter_uncertainty.loc['emis2conc'] = np.array([0,0,0])
	gas_parameter_uncertainty.loc['f1':'f3'] = np.array([[0,0,0],[0,0,0],[0,0,0]]).T
	gas_parameter_uncertainty.loc['F_2x'] = np.array([0,0,0])

	gas_parameter_uncertainty = pd.concat([gas_parameter_uncertainty], keys = ['normal'], axis = 1)

	gas_parameter_uncertainty.index = gas_parameter_uncertainty.index.rename('param_name')

	gas_parameter_uncertainty.columns = gas_parameter_uncertainty.columns.rename(['Distribution','Gas'])

	return gas_parameter_uncertainty.apply(pd.to_numeric)

def default_thermal_params():

	# returns a dataframe of default parameters in the format UnFaIR requires (pd.concat -> additional sets)

	thermal_parameter_list = ['d','tcr_ecs']

	thermal_parameters = pd.DataFrame(columns=[1,2],index=thermal_parameter_list)
	thermal_parameters.loc['d'] = np.array([239.0,4.1])
	thermal_parameters.loc['tcr_ecs'] = np.array([1.58,2.66])

	thermal_parameters = pd.concat([thermal_parameters], keys = ['default'], axis = 1)

	thermal_parameters.index = thermal_parameters.index.rename('param_name')

	thermal_parameters.columns = thermal_parameters.columns.rename(['Thermal_param_set','Box'])

	return thermal_parameters.apply(pd.to_numeric)

def default_thermal_param_uncertainty():

	# returns a dataframe of default parameters in the format UnFaIR requires (pd.concat -> additional sets)

	thermal_parameter_list = ['d','tcr_ecs']

	thermal_parameter_uncertainty = pd.DataFrame(columns=[1,2],index=thermal_parameter_list)
	thermal_parameter_uncertainty = pd.concat([thermal_parameter_uncertainty]*2,keys=[5,95],axis=1)
	thermal_parameter_uncertainty.loc['d'] = [239,1.6,239,8.4]
	thermal_parameter_uncertainty.loc['tcr_ecs'] = [1,1.6,2.5,4.5]

	thermal_parameter_uncertainty.index = thermal_parameter_uncertainty.index.rename('param_name')

	thermal_parameter_uncertainty.columns = thermal_parameter_uncertainty.columns.rename(['Percentile','Box'])

	return thermal_parameter_uncertainty.apply(pd.to_numeric)

def draw_monte_carlo_param_set(N , input_parameters , input_uncertainties , type = 'normal'):

	# function that takes a single set of parameter medians and corresponding % uncertainty dataframe, and creates a
	# new dataframe with N samples of the parameter set.

	if type == 'normal':

		param_set = [input_parameters[input_parameters.columns.levels[0][0]]]

		for i in np.arange(N):

			param_set += [param_set[0] * np.random.normal(np.ones(input_parameters.shape),input_uncertainties)]

		param_set = pd.concat(param_set, keys = ['median']+[x + type for x in [str(i) for i in np.arange(N-1)]], axis = 1)

		return param_set

	if type == 'lognormal':

		param_set = input_parameters[input_parameters.columns.levels[0][0]]

		loc = ((param_set**2 - input_uncertainties[5]*input_uncertainties[95]) / (input_uncertainties[5]+input_uncertainties[95]-2*param_set)).fillna(0)
		mu = np.log(param_set+loc)
		scale = ( np.log(input_uncertainties[95]+loc) - mu ) / 1.645

		# Constrain to be within +/- 3 sigma
		constrain_high = param_set.copy()
		constrain_low = param_set.copy()
		constrain_low.loc[:] = sp.stats.lognorm.ppf(0.003,scale.fillna(0),-loc.fillna(0),np.exp(mu.fillna(0)))
		constrain_high.loc[:] = sp.stats.lognorm.ppf(0.973,scale.fillna(0),-loc.fillna(0),np.exp(mu.fillna(0)))
		constrain_low,constrain_high = constrain_low.fillna(-10**10),constrain_high.fillna(10**10)

		param_set = [param_set]

		for i in np.arange(N):
		    while True:
		        new_param_set = np.random.lognormal(mu.fillna(0),scale.fillna(0))-loc.fillna(0)

		        if all(new_param_set<constrain_high) and all(new_param_set>constrain_low):

		            break

		    param_set += [new_param_set]

		param_set = pd.concat(param_set, keys = ['median']+[x + 'lognorm' for x in [str(i) for i in np.arange(N-1)]], axis = 1)

		return param_set

def tcr_ecs_to_q(input_parameters=True , F_2x=3.79866 , help=False):

	# converts a tcr / ecs / d dataframe into a d / q dataframe for use in UnFaIRv2
	# F2x is the FaIR v2.0 default forcing parameter value

	if help:
		tcr_ecs_test = default_thermal_params()
		tcr_ecs_test = pd.concat([tcr_ecs_test['default']]*2,keys=['default','1'],axis=1)
		tcr_ecs_test.loc['tcr_ecs'] = [1.6,2.75,1.4,2.4]
		tcr_ecs_test = tcr_ecs_test.loc[['d','tcr_ecs']]
		print('Example input format:')
		return tcr_ecs_test

	if type(input_parameters.columns) != pd.core.indexes.multi.MultiIndex:
		return 'input_parameters not in MultiIndex DataFrame. Set help=True for formatting of input.'
	else:
		output_params = input_parameters.copy()
		param_arr = input_to_numpy(input_parameters)
		k = 1.0 - (param_arr[:,:,0]/70.0)*(1.0 - np.exp(-70.0/param_arr[:,:,0]))
		output_params.loc['q'] = ( ( param_arr[:,0,1][:,np.newaxis] - param_arr[:,1,1][:,np.newaxis] * np.roll(k,shift=1) )/( F_2x * ( k - np.roll(k,shift=1) ) ) ) .flatten()

		return output_params.loc[['d','q']]

def q_to_tcr_ecs(input_parameters=True , F_2x=3.79866 , help=False):

	# converts a tcr / ecs / d dataframe into a d / q dataframe for use in UnFaIRv2

	if help:
		tcr_ecs_test = default_thermal_params()
		tcr_ecs_test = pd.concat([tcr_ecs_test['default']]*2,keys=['default','1'],axis=1)
		tcr_ecs_test.loc['q'] = [0.33,0.41,0.31,0.43]
		tcr_ecs_test = tcr_ecs_test.loc[['d','q']]
		print('Example input format:')
		return tcr_ecs_test

	if type(input_parameters.columns) != pd.core.indexes.multi.MultiIndex:
		return 'input_parameters not in MultiIndex DataFrame. Set help=True for formatting of input.'
	else:
		
		output_params = pd.DataFrame(index = ['ECS','TCR'],columns = input_parameters.columns.levels[0])
		
		for param_set in input_parameters.columns.levels[0]:
    
			params = input_parameters.xs(param_set,level=0,axis=1)

			ECS = F_2x * params.loc['q'].sum()

			TCR = F_2x * ( params.loc['q'] * (1 - (params.loc['d']/70) * ( 1 - np.exp(-70/params.loc['d']) ) ) ).sum()

			output_params.loc[:,param_set] = [ECS,TCR]

		return output_params

def calculate_alpha(G,G_A,T,r,g0,g1,iirf100_max = 97.0):

	iirf100_val = r[...,0] + r[...,1] * (G-G_A) + r[...,2] * T + r[...,3] * G_A

	iirf100_val = np.abs(iirf100_val)

	iirf100_val = (iirf100_val>iirf100_max) * iirf100_max + iirf100_val * (iirf100_val<iirf100_max)

	alpha_val = g0 * np.sinh(iirf100_val / g1)

	return alpha_val

def step_concentration(R,E,alpha,a,tau,PI_conc,emis2conc,dt=1):

	R = E * emis2conc[...,np.newaxis] * a * alpha * (tau/dt) * ( 1. - np.exp( -dt/(alpha*tau) ) ) + R * np.exp( -dt/(alpha * tau) )

	C = PI_conc + np.sum(R,axis=-1)

	G_A = (C - PI_conc) / emis2conc

	return C,R,G_A

def step_forcing(C,PI_conc,f):

	RF = f[...,0] * np.log( C/PI_conc ) + f[...,1] * ( C - PI_conc ) + f[...,2] * ( np.sqrt(C) - np.sqrt(PI_conc) )

	return RF

def step_temperature(S,F,q,d,dt=1):

	S = q * F * ( 1 - np.exp(-dt/d) ) + S * np.exp(-dt/d)

	T = np.sum(S,axis=-1)

	return S,T

# Run modes:
	# Full forward
	# Concentration driven
	# Forcing scenarios single emission scenario
	# Set temperature response (gas cycle mode)
	# Concentrations to Emissions switch
# Checks on:
	# timeseries length
	# parameter formatting
	# parameter size / shape
	# same number of scenarios in emissions and forcing
	#

def run_UnFaIR( emissions_in = False , \
			    concentrations_in = False , \
				forcing_in = 0.0 , \
				gas_parameters = default_gas_forcing_params() , \
				thermal_parameters = tcr_ecs_to_q(default_thermal_params()) , \
				show_run_info = True ):

	# Determine the number of scenario runs , parameter sets , gases , integration period, timesteps
	
	# There are 3 modes : emissions_driven , concentration_driven & emission_concentration_switch
	
	# The model differentiates between these as follows (it assumes you input them in the correct format of multiindex df, and scenarios that match for the case of emissions_concentrations_switch:
	
	if not concentrations_in is False: # check if concentration driven
		concentration_driven = True		
		if emissions_in is False: # make sure pure concentration driven
			emissions_in = return_empty_emissions(concentrations_in,gases_in=concentrations_in.columns.levels[1])
			emissions_concentration_switch = False
		else: # otherwise emissions -> concentration run
			emissions_concentration_switch = True
	else: # emissions only
		concentration_driven=False
		emissions_concentration_switch = False
		
	if concentration_driven:
		if emissions_concentration_switch:
			time_index = pd.Index(sorted(list(set(concentrations_in.index.append(emissions_in.index)))))
		else:
			time_index = concentrations_in.index
	else:
		time_index = emissions_in.index

	dim_scenario = emissions_in.columns.levels[0].size
	scen_names = list(emissions_in.columns.levels[0])
	dim_gas_param = gas_parameters.columns.levels[0].size
	gas_set_names = list(gas_parameters.columns.levels[0])
	dim_thermal_param = thermal_parameters.columns.get_level_values(0).unique().size
	thermal_set_names = list(thermal_parameters.columns.get_level_values(0).unique())
	n_gas = emissions_in.columns.levels[1].size
	gas_names = list(gas_parameters.columns.levels[1])
	n_year = time_index.size
	
	timestep = np.append(np.diff(time_index)[0],np.diff(time_index))

	# Reformat inputs into the right shape

	emissions = input_to_numpy(emissions_in)[:,np.newaxis,np.newaxis,...]
	
	if concentration_driven:
		if emissions_concentration_switch:
			concentrations = input_to_numpy(concentrations_in.loc[:emissions_in.index[0]-1e-8])[:,np.newaxis,np.newaxis,...] # only want concentrations UP TO the start of emissions
		else:
			concentrations = input_to_numpy(concentrations_in)[:,np.newaxis,np.newaxis,...]

	if type(forcing_in)==float or type(forcing_in)==int:
		ext_forcing = forcing_in + np.zeros((*(emissions.shape[:-2]),1,time_index.size))
	elif type(forcing_in)==pd.core.frame.DataFrame:
		if not emissions_concentration_switch:
			if type(forcing_in.columns)==pd.core.indexes.multi.MultiIndex:
				if forcing_in.index.equals(emissions_in.index):
					if forcing_in.columns.levels[0].equals(emissions_in.columns.levels[0]) or forcing_in.columns.levels[0].size == 1:
						ext_forcing = input_to_numpy(forcing_in)[:,np.newaxis,np.newaxis,...]
					else:
						print('Error')
						return "Multiple forcing scenarios given but differ from emissions scenarios"
				else:
					print('Error')
					return "forcing timeseries length different to emission timeseries"
			else:
				print('Error')
				return "forcing DataFrame not MultiIndex, use pd.concat([df],keys=['scenario1'...],axis=1)"
		else:
			ext_forcing = input_to_numpy(forcing_in)[:,np.newaxis,np.newaxis,...]
	else:
		print('Error')
		return "forcing not pandas DataFrame, use return_empty_forcing to check correct inpurt formatting"

	if show_run_info:
		print('Integrating ' + str(dim_scenario) + ' scenarios, ' + \
			   str(dim_gas_param) + ' gas cycle parameter sets, ' + \
			   str(dim_thermal_param) + ' thermal response parameter sets, over ' + \
			   str(list(emissions_in.columns.levels[1])) + ', between ' + \
			   str(time_index[0]) + ' and ' + str(time_index[-1]) + '...')

	# Slice the parameter sets into numpy arrays of the right shape
	# Dimensions : [scenario, gas params, thermal params, gas, time, (gas/thermal pools)]

	a = input_to_numpy(gas_parameters.loc['a1':'a4'])[np.newaxis,:,np.newaxis,...]
	tau = input_to_numpy(gas_parameters.loc['tau1':'tau4'])[np.newaxis,:,np.newaxis,...]
	r = input_to_numpy(gas_parameters.loc['r0':'rA'])[np.newaxis,:,np.newaxis,...]
	emis2conc = gas_parameters.loc['emis2conc'].values.reshape(gas_parameters.loc['emis2conc'].index.levels[0].size,gas_parameters.loc['emis2conc'].index.levels[1].size)[np.newaxis,:,np.newaxis,...]
	PI_conc = gas_parameters.loc['PI_conc'].values.reshape(gas_parameters.loc['PI_conc'].index.levels[0].size,gas_parameters.loc['PI_conc'].index.levels[1].size)[np.newaxis,:,np.newaxis,...]
	

	f = input_to_numpy(gas_parameters.loc['f1':'f3'])[np.newaxis,:,np.newaxis,...]

	d = thermal_parameters.loc['d'].values.reshape(thermal_parameters.loc['d'].index.get_level_values(0).unique().size,thermal_parameters.loc['d'].index.get_level_values(1).unique().size)[np.newaxis,np.newaxis,...]
	q = thermal_parameters.loc['q'].values.reshape(thermal_parameters.loc['q'].index.get_level_values(0).unique().size,thermal_parameters.loc['q'].index.get_level_values(1).unique().size)[np.newaxis,np.newaxis,...]


	# Create appropriate shape variable arrays / calculate RF if concentration driven
	
	C = np.zeros((dim_scenario,dim_gas_param,dim_thermal_param,n_gas,n_year))
	RF = np.zeros((dim_scenario,dim_gas_param,dim_thermal_param,n_gas,n_year))
	if concentration_driven:
		C[...,:concentrations.shape[-1]] = concentrations.copy()
		RF[...,:concentrations.shape[-1]] = step_forcing(concentrations,PI_conc[...,np.newaxis],f[...,np.newaxis,:])
		if emissions_concentration_switch:
			G = np.zeros((dim_scenario,dim_gas_param,dim_thermal_param,n_gas,n_year))
	else:
		G = np.cumsum(emissions,axis=-1)
	alpha = np.zeros((dim_scenario,dim_gas_param,dim_thermal_param,n_gas,n_year))
	T = np.zeros((dim_scenario,dim_gas_param,dim_thermal_param,n_year))
		
	

	# Initialize the first timestep
	g1 = np.sum( a * tau * ( 1. - ( 1. + 100/tau ) * np.exp(-100/tau) ), axis=-1 )
	g0 = ( np.sinh( np.sum( a * tau * ( 1. - np.exp(-100/tau) ) , axis=-1) / g1 ) )**(-1.)
	
	if not concentration_driven:
		alpha[...,0] = calculate_alpha(G=np.zeros(C[...,0].shape),G_A=np.zeros(C[...,0].shape),T=T[...,0,np.newaxis],r=r,g0=g0,g1=g1)
		C[...,0],R,G_A = step_concentration(R = np.zeros(a.shape),alpha=alpha[...,0,np.newaxis],E=emissions[...,0,np.newaxis],\
											a=a,tau=tau,PI_conc=PI_conc,emis2conc=emis2conc,dt=timestep[0])
		RF[...,0] = step_forcing(C=C[...,0],PI_conc=PI_conc,f=f)
	S,T[...,0] = step_temperature(S=np.zeros(d.shape),F=np.sum(RF[...,0],axis=-1)[...,np.newaxis]+ext_forcing[...,0],q=q,d=d,dt=timestep[0])

	# Step over remaining timesteps

	if concentration_driven:
		
		for t in np.arange(1,concentrations.shape[-1]):
			S,T[...,t] = step_temperature(S=S,F=np.sum(RF[...,t],axis=-1)[...,np.newaxis]+ext_forcing[...,t],q=q,d=d,dt=timestep[t])
			
		if emissions_concentration_switch:

			concentration_driven = False

			past_emissions, R, G[...,:t+1], G_A = unstep_concentration(C[...,:t+1], T[...,:t+1], a, tau, r, PI_conc, emis2conc, timestep[...,:t+1], concentration_driven = True)
			
			new_emissions = np.zeros((*(past_emissions.shape[:-1]),n_year))
			
			new_emissions[...,:past_emissions.shape[-1]] = past_emissions
			new_emissions[...,past_emissions.shape[-1]:] = emissions
			emissions = new_emissions.copy()
			
			G = np.cumsum(emissions,axis=-1)
			
			alpha[...,0] = calculate_alpha(G=np.zeros(C[...,0].shape),G_A=np.zeros(C[...,0].shape),T=T[...,0,np.newaxis],r=r,g0=g0,g1=g1)
			C[...,0],R,G_A = step_concentration(R = np.zeros(a.shape),alpha=alpha[...,0,np.newaxis],E=emissions[...,0,np.newaxis],\
												a=a,tau=tau,PI_conc=PI_conc,emis2conc=emis2conc,dt=timestep[0])
			RF[...,0] = step_forcing(C=C[...,0],PI_conc=PI_conc,f=f)
			S,T[...,0] = step_temperature(S=np.zeros(d.shape),F=np.sum(RF[...,0],axis=-1)[...,np.newaxis]+ext_forcing[...,0],q=q,d=d,dt=timestep[0])

# 			adjust_time = t+1 # we need to add on all the time passed in concentration mode for this to work...
# 			adjust_emission_time = 1 # since emissions starts from index 0, not end of concentrations

# 	else:

# 		adjust_time = 0
# 		adjust_emission_time = 0
		
			
	if not concentration_driven:

# 		for t in np.arange(1-adjust_emission_time,emissions.shape[-1]) + adjust_time:
		for t in np.arange(1,emissions.shape[-1]):

			alpha[...,t] = calculate_alpha(G=G[...,t-1],G_A=G_A,T=T[...,t-1,np.newaxis],r=r,g0=g0,g1=g1)
			C[...,t],R,G_A = step_concentration(R = R,alpha=alpha[...,t,np.newaxis],E=emissions[...,t,np.newaxis],\
												a=a,tau=tau,PI_conc=PI_conc,emis2conc=emis2conc,dt=timestep[t])
			RF[...,t] = step_forcing(C=C[...,t],PI_conc=PI_conc,f=f)
			S,T[...,t] = step_temperature(S=S,F=np.sum(RF[...,t],axis=-1)[...,np.newaxis]+ext_forcing[...,t],q=q,d=d,dt=timestep[t])

	ext_forcing = np.zeros(np.sum(RF,axis=-2)[...,np.newaxis,:].shape) + ext_forcing
	RF = np.concatenate((RF,ext_forcing),axis=-2)
	RF = np.concatenate((RF,np.sum(RF,axis=-2)[...,np.newaxis,:]),axis=-2)
	
	if concentration_driven:
        # Calculate diagnosed emissions
		diagnosed_emissions = unstep_concentration(C[...,:], T[...,:], a, tau, r, PI_conc, emis2conc, timestep[...,:], concentration_driven = True)[0]
        # Also calculate diagnosed alpha values:
		g1 = np.sum( a * tau * ( 1. - ( 1. + 100/tau ) * np.exp(-100/tau) ), axis=-1 )
		g0 = ( np.sinh( np.sum( a * tau * ( 1. - np.exp(-100/tau) ) , axis=-1) / g1 ) )**(-1.)
		G = np.cumsum(diagnosed_emissions,axis=-1)
		G_A = (C - PI_conc[...,np.newaxis]) / emis2conc[...,np.newaxis]
		alpha = np.zeros((dim_scenario,dim_gas_param,dim_thermal_param,n_gas,n_year))
		alpha[...,0] = calculate_alpha(G=np.zeros(C[...,0].shape),G_A=np.zeros(C[...,0].shape),T=T[...,0,np.newaxis],r=r,g0=g0,g1=g1)
		for t in np.arange(1,diagnosed_emissions.shape[-1]):
			alpha[...,t] = calculate_alpha(G=G[...,t-1],G_A=G_A[...,t-1],T=T[...,t-1,np.newaxis],r=r,g0=g0,g1=g1)
		
		C_out = concentrations_in
		E_out = pd.DataFrame(diagnosed_emissions.T.swapaxes(1,-1).swapaxes(2,-2).reshape(n_year,n_gas*dim_scenario*dim_gas_param*dim_thermal_param),index = time_index,columns=pd.MultiIndex.from_product([scen_names,gas_set_names,thermal_set_names,gas_names],names=['Scenario','Gas cycle set','Thermal set','Gas name']))
		alpha_out = pd.DataFrame(alpha.T.swapaxes(1,-1).swapaxes(2,-2).reshape(n_year,n_gas*dim_scenario*dim_gas_param*dim_thermal_param),index = time_index,columns=pd.MultiIndex.from_product([scen_names,gas_set_names,thermal_set_names,gas_names],names=['Scenario','Gas cycle set','Thermal set','Gas name']))
		RF_out = pd.DataFrame(RF.T.swapaxes(1,-1).swapaxes(2,-2).reshape(n_year,(n_gas+2)*dim_scenario*dim_gas_param*dim_thermal_param),index = time_index,columns=pd.MultiIndex.from_product([scen_names,gas_set_names,thermal_set_names,gas_names+['External','Total']],names=['Scenario','Gas cycle set','Thermal set','Gas name']))
		T_out = pd.DataFrame(T.T.swapaxes(1,-1).reshape(n_year,dim_scenario*dim_gas_param*dim_thermal_param),index = time_index,columns=pd.MultiIndex.from_product([scen_names,gas_set_names,thermal_set_names],names=['Scenario','Gas cycle set','Thermal set']))
			
		
	else:
		C_out = pd.DataFrame(C.T.swapaxes(1,-1).swapaxes(2,-2).reshape(n_year,n_gas*dim_scenario*dim_gas_param*dim_thermal_param),index = time_index,columns=pd.MultiIndex.from_product([scen_names,gas_set_names,thermal_set_names,gas_names],names=['Scenario','Gas cycle set','Thermal set','Gas name']))
		alpha_out = pd.DataFrame(alpha.T.swapaxes(1,-1).swapaxes(2,-2).reshape(n_year,n_gas*dim_scenario*dim_gas_param*dim_thermal_param),index = time_index,columns=pd.MultiIndex.from_product([scen_names,gas_set_names,thermal_set_names,gas_names],names=['Scenario','Gas cycle set','Thermal set','Gas name']))
		E_out = emissions_in
		RF_out = pd.DataFrame(RF.T.swapaxes(1,-1).swapaxes(2,-2).reshape(n_year,(n_gas+2)*dim_scenario*dim_gas_param*dim_thermal_param),index = time_index,columns=pd.MultiIndex.from_product([scen_names,gas_set_names,thermal_set_names,gas_names+['External','Total']],names=['Scenario','Gas cycle set','Thermal set','Gas name']))
		T_out = pd.DataFrame(T.T.swapaxes(1,-1).reshape(n_year,dim_scenario*dim_gas_param*dim_thermal_param),index = time_index,columns=pd.MultiIndex.from_product([scen_names,gas_set_names,thermal_set_names],names=['Scenario','Gas cycle set','Thermal set']))
		
	if emissions_concentration_switch:
		E_out = pd.DataFrame(emissions.T.swapaxes(1,-1).swapaxes(2,-2).reshape(n_year,n_gas*dim_scenario*dim_gas_param*dim_thermal_param),index = time_index,columns=pd.MultiIndex.from_product([scen_names,gas_set_names,thermal_set_names,gas_names],names=['Scenario','Gas cycle set','Thermal set','Gas name']))
	else:
		past_emissions_out = False

	out_dict = { \
				'C':C_out, \
				'RF':RF_out, \
				'T':T_out, \
				'alpha':alpha_out, \
				'Emissions':E_out , \
				'gas_parameters':gas_parameters , \
				'thermal parameters':thermal_parameters \
			   }

	for axis in [x for x in list(out_dict.keys())[:-2] if type(x)==pd.core.frame.DataFrame]:
		out_dict[axis].index = out_dict[axis].index.rename('Year')

	return out_dict


############################### Dev Tools ###############################


def prescribed_temps_gas_cycle(emissions_in , \
				gas_parameters , \
			    T):
	
	# for running the gas cycle module only, with a prescribed temperature dataset. For fitting cycle parameters
	
	dim_scenario = emissions_in.columns.levels[0].size
	scen_names = list(emissions_in.columns.levels[0])
	dim_gas_param = gas_parameters.columns.levels[0].size
	gas_set_names = list(gas_parameters.columns.levels[0])
	n_gas = emissions_in.columns.levels[1].size
	gas_names = list(gas_parameters.columns.levels[1])
	n_year = emissions_in.index.size
	
	emissions = input_to_numpy(emissions_in)[:,np.newaxis,...]
	
	timestep = np.append(np.diff(emissions_in.index)[0],np.diff(emissions_in.index))
	
	T = T[np.newaxis,np.newaxis,:]
	
	a = input_to_numpy(gas_parameters.loc['a1':'a4'])[np.newaxis,:,np.newaxis,...]
	tau = input_to_numpy(gas_parameters.loc['tau1':'tau4'])[np.newaxis,:,np.newaxis,...]
	r = input_to_numpy(gas_parameters.loc['r0':'rA'])[np.newaxis,:,np.newaxis,...]
	emis2conc = gas_parameters.loc['emis2conc'].values.reshape(gas_parameters.loc['emis2conc'].index.levels[0].size,gas_parameters.loc['emis2conc'].index.levels[1].size)[np.newaxis,:,np.newaxis,...]
	PI_conc = gas_parameters.loc['PI_conc'].values.reshape(gas_parameters.loc['PI_conc'].index.levels[0].size,gas_parameters.loc['PI_conc'].index.levels[1].size)[np.newaxis,:,np.newaxis,...]
	

	f = input_to_numpy(gas_parameters.loc['f1':'f3'])[np.newaxis,:,np.newaxis,...]
	
	G = np.cumsum(emissions,axis=-1)
	C = np.zeros((dim_scenario,dim_gas_param,n_gas,n_year))
	alpha = np.zeros((dim_scenario,dim_gas_param,n_gas,n_year))
	
	g1 = np.sum( a * tau * ( 1. - ( 1. + 100/tau ) * np.exp(-100/tau) ), axis=-1 )
	g0 = ( np.sinh( np.sum( a * tau * ( 1. - np.exp(-100/tau) ) , axis=-1) / g1 ) )**(-1.)
	
	alpha[...,0] = calculate_alpha(G=np.zeros(C[...,0].shape),G_A=np.zeros(C[...,0].shape),T=T[...,0,np.newaxis],r=r,g0=g0,g1=g1)
	C[...,0],R,G_A = step_concentration(R = np.zeros(a.shape),alpha=alpha[...,0,np.newaxis],E=emissions[...,0,np.newaxis],\
										a=a,tau=tau,PI_conc=PI_conc,emis2conc=emis2conc,dt=timestep[0])
	
	for t in np.arange(1,emissions.shape[-1]):

		alpha[...,t] = calculate_alpha(G=G[...,t-1],G_A=G_A,T=T[...,t-1,np.newaxis],r=r,g0=g0,g1=g1)
		C[...,t],R,G_A = step_concentration(R = R,alpha=alpha[...,t,np.newaxis],E=emissions[...,t,np.newaxis],\
												a=a,tau=tau,PI_conc=PI_conc,emis2conc=emis2conc,dt=timestep[t])
		
	C_out = pd.DataFrame(C.T.swapaxes(1,-1).swapaxes(2,-2).reshape(n_year,n_gas*dim_scenario*dim_gas_param),index = emissions_in.index,columns=pd.MultiIndex.from_product([scen_names,gas_set_names,gas_names],names=['Scenario','Gas cycle set','Gas name']))
	alpha_out = pd.DataFrame(alpha.T.swapaxes(1,-1).swapaxes(2,-2).reshape(n_year,n_gas*dim_scenario*dim_gas_param),index = emissions_in.index,columns=pd.MultiIndex.from_product([scen_names,gas_set_names,gas_names],names=['Scenario','Gas cycle set','Gas name']))
	E_out = emissions_in
	
	out_dict = { \
				'C':C_out, \
				'alpha':alpha_out, \
				'Emissions':E_out , \
				'gas_parameters':gas_parameters , \
			   }

	for axis in [x for x in list(out_dict.keys())[:-2] if type(x)==pd.core.frame.DataFrame]:
		out_dict[axis].index = out_dict[axis].index.rename('Year')

	return out_dict


def unstep_concentration(C, T, a, tau, r, PI_conc, emis2conc, timestep, concentration_driven = False):
	
	## This is intended to be used with arrays of the shape the main model (ie in the main model) uses for calculation, and outputs data in a similar format (ie. not pandas!)
	
	# Dimensions : [scenario, gas params, thermal params, gas, time, (gas/thermal pools)]
	
# 	if concentration_driven:
# 		PI_conc = C[...,0]
	
	dim_scenario = T.shape[0]
	dim_gas_param = T.shape[1]
	dim_thermal_param = T.shape[2]
	n_gas = C.shape[3]
	n_year = C.shape[4]
	
	alpha = np.zeros((dim_scenario,dim_gas_param,dim_thermal_param,n_gas,n_year))
	G = alpha.copy()
	emissions = alpha.copy()
	R = np.zeros(a.shape)
	
	G_A = (C - PI_conc[...,np.newaxis]) / emis2conc[...,np.newaxis]
	
	g1 = np.sum( a * tau * ( 1. - ( 1. + 100/tau ) * np.exp(-100/tau) ), axis=-1 )
	g0 = ( np.sinh( np.sum( a * tau * ( 1. - np.exp(-100/tau) ) , axis=-1) / g1 ) )**(-1.)
	
	# inital timestep all variables are 0
	
	dt = timestep[0]
	
	alpha[...,0] = calculate_alpha(G=G[...,0],G_A=G[...,0],T=G[...,0],r=r,g0=g0,g1=g1)
	
	emissions[...,0] = ( ( C[...,0] - PI_conc - np.sum(R * np.exp( -dt/(alpha[...,0,np.newaxis] * tau) ) ,axis=-1 ) ) / emis2conc ) / np.sum( a * alpha[...,0,np.newaxis] * ( tau / dt ) * ( 1. - np.exp( -dt / ( alpha[...,0,np.newaxis] * tau ) ) ) , axis=-1 )
	
	R = emissions[...,0,np.newaxis] * emis2conc[...,np.newaxis] * a * alpha[...,0,np.newaxis] * ( tau / dt ) * ( 1. - np.exp( -dt/(alpha[...,0,np.newaxis]*tau) ) ) + R * np.exp( -dt/(alpha[...,0,np.newaxis] * tau) )
	
	G = np.cumsum(emissions,axis=-1)
	
	for t in np.arange(1,C.shape[-1]):
		
		dt = timestep[t]
		
		alpha[...,t] = calculate_alpha(G=G[...,t-1],G_A=G_A[...,t-1],T=T[...,t-1,np.newaxis],r=r,g0=g0,g1=g1)
	
		emissions[...,t] = ( ( C[...,t] - PI_conc - np.sum( R * np.exp( -dt / ( alpha[...,t,np.newaxis] * tau ) ) ,axis=-1 ) ) / emis2conc ) / np.sum( a * alpha[...,t,np.newaxis] * ( tau / dt ) * ( 1. - np.exp( -dt / ( alpha[...,t,np.newaxis] * tau ) ) ) , axis=-1 )
	
		R = emissions[...,t,np.newaxis] * emis2conc[...,np.newaxis] * a * alpha[...,t,np.newaxis] * ( tau / dt ) * ( 1. - np.exp( -dt/(alpha[...,t,np.newaxis]*tau) ) ) + R * np.exp( -dt/(alpha[...,t,np.newaxis] * tau) )
		
		G = np.cumsum(emissions,axis=-1)
	
	return emissions, R, G, G_A[...,-1]


def unstep_forcing(forcing_in,gas_parameters=default_gas_forcing_params(),thermal_params=default_thermal_params()):
    
    f = input_to_numpy(gas_parameters.loc['f1':'f3'])[np.newaxis,:,np.newaxis,...]
    
    forcing_in = return_empty_emissions(forcing_in,gases_in=forcing_in.columns.levels[1]) + forcing_in.values
    
    forcing = input_to_numpy(forcing_in)[:,np.newaxis,np.newaxis,...]
    
    time_index = forcing_in.index

    dim_scenario = forcing_in.columns.levels[0].size
    scen_names = list(forcing_in.columns.levels[0])
    dim_gas_param = gas_parameters.columns.levels[0].size
    gas_set_names = list(gas_parameters.columns.levels[0])
    gas_names = list(gas_parameters.columns.levels[1])
    dim_thermal_param = thermal_params.columns.get_level_values(0).unique().size
    thermal_set_names = list(thermal_params.columns.get_level_values(0).unique())
    n_gas = forcing_in.columns.levels[1].size
    n_year = time_index.size

    f = input_to_numpy(gas_parameters.loc['f1':'f3'])[np.newaxis,:,np.newaxis,...]

    PI_conc = gas_parameters.loc['PI_conc'].values.reshape(gas_parameters.loc['PI_conc'].index.levels[0].size,gas_parameters.loc['PI_conc'].index.levels[1].size)[np.newaxis,:,np.newaxis,...]
    
    def root_function(C,PI_conc,f,forcing_target):
    
        RF = f[...,0] * np.log( C/PI_conc ) + f[...,1] * ( C - PI_conc ) + f[...,2] * ( np.sqrt(C) - np.sqrt(PI_conc) )
    
        return RF - forcing_target

    concentrations = np.zeros(forcing.shape)
    
    for scenario in np.arange(dim_scenario):
    
        for gas_param in np.arange(dim_gas_param):

            for thermal_param in np.arange(dim_thermal_param):

                for gas in np.arange(n_gas):

                    concentrations[scenario,gas_param,thermal_param,gas,:]=sp.optimize.root(root_function,\
                                                                                            np.zeros(forcing[scenario,gas_param,thermal_param,gas,:].shape)+\
                                                                                            PI_conc[0,gas_param,0,gas],\
                                                                                            args=(PI_conc[0,gas_param,0,gas],\
                                                                                                  f[0,gas_param,0,gas,:],\
                                                                                                  forcing[scenario,gas_param,thermal_param,gas,:])).x.squeeze()

    C_out = pd.DataFrame(concentrations.T.swapaxes(1,-1).swapaxes(2,-2).reshape(n_year,n_gas*dim_scenario*dim_gas_param*dim_thermal_param),index = time_index,columns=pd.MultiIndex.from_product([scen_names,gas_set_names,thermal_set_names,gas_names],names=['Scenario','Gas cycle set','Thermal set','Gas name']))
    
    return C_out