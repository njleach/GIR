# GIR - Nick Leach and Stuart Jenkins

import numpy as np
import pandas as pd
import scipy as sp
from pathlib import Path

def return_empty_emissions(df_to_copy=False, start_year=1765, end_year=2500, timestep=1, scen_names=[0], gases_in = ['CO2','CH4','N2O'], help=False):

    if help:
        print('This function returns a dataframe of zeros in the correct format for use in GIR. Pass an existing emission/ concentration array to return a corresponding forcing array.')
    
    if type(df_to_copy)==pd.core.frame.DataFrame:
        df = pd.DataFrame(index = df_to_copy.index,columns=pd.MultiIndex.from_product([df_to_copy.columns.levels[0],gases_in],names=['Scenario','Gas'])).fillna(0).apply(pd.to_numeric)
        
    else:

        df = pd.DataFrame(index=np.arange(start_year,end_year+1,timestep)+(timestep!=1)*timestep/2,columns=pd.MultiIndex.from_product([scen_names,gases_in],names=['Scenario','Gas'])).fillna(0).apply(pd.to_numeric)
        df.index.rename('Year',inplace=True)

    return df

def return_empty_forcing(df_to_copy=False, start_year=1765, end_year=2500, timestep=1, scen_names=[0], help=False):
    
    if help:
        print('This function returns a dataframe of zeros in the correct format for use in GIR. Pass an existing emission/ concentration array to return a corresponding forcing array.')
    
    if type(df_to_copy)==pd.core.frame.DataFrame:
        df = pd.DataFrame(index = df_to_copy.index,columns=pd.MultiIndex.from_product([df_to_copy.columns.levels[0],['forcing']],names=['Scenario','Variable'])).fillna(0).apply(pd.to_numeric)
        
    else:
        
        df = pd.DataFrame(index=np.arange(start_year,end_year+1,timestep)+(timestep!=1)*timestep/2,columns=pd.MultiIndex.from_product([scen_names,['forcing']],names=['Scenario','Gas'])).fillna(0).apply(pd.to_numeric)
        df.index.rename('Year',inplace=True)

    return df

def input_to_numpy(input_df):

    # converts the dataframe input into a numpy array for calculation, dimension order = [name, gas, time/parameter]

    return input_df.values.T.reshape(input_df.columns.levels[0].size, input_df.columns.levels[1].size, input_df.index.size)


def get_gas_parameter_defaults(choose_gases=['CO2','CH4','N2O'],CH4_forc_feedbacks=False, help=False):
    
    if help:
        print('This function returns the GIR default parameter set for a gas set of your choice. You can choose from the following gas species:')
        possible_gases = list(pd.read_pickle(Path(__file__).parent / "./Parameter_Sets/Complete_parameter_set.p").columns.levels[-1])
        return possible_gases
    
    CHOOSE_params = pd.read_pickle(Path(__file__).parent / "./Parameter_Sets/Complete_parameter_set.p").reindex(choose_gases,axis=1,level=1)
    
    if CH4_forc_feedbacks=='indirect':
        
        CHOOSE_params.loc['f2',(param_set_name,'CH4')] += 0.000182 + 5.4e-05 # add on the indirect forcings
        
    elif CH4_forc_feedbacks=='ozone_parameterisation':
        
        CHOOSE_params.loc['f2',(param_set_name,'CH4')] += 3.7e-04 + 6.9e-05 - 4.6e-05 # add on the indirect forcings
        
    else:
        
        return CHOOSE_params
    
def get_thermal_parameter_defaults(TCR_ECS=np.array([1.6,2.76]),F_2x=3.84):
    
    thermal_parameter_list = ['d','q']

    thermal_parameters = pd.DataFrame(columns=[1,2,3],index=thermal_parameter_list)
    
    d = np.array([283,9.88,0.85])
    q = np.array([0,0,0.242])
    k = 1-(d/70)*(1-np.exp(-70/d))
    q[:2] = ((TCR_ECS[0]/F_2x - k[2]*q[2]) - np.roll(k[:2],axis=0,shift=1)*(TCR_ECS[1]/F_2x - q[2]))/(k[:2] - np.roll(k[:2],axis=0,shift=1))
    
    thermal_parameters.loc['d'] = d
    thermal_parameters.loc['q'] = q

    thermal_parameters = pd.concat([thermal_parameters], keys = ['default'], axis = 1)

    thermal_parameters.index = thermal_parameters.index.rename('param_name')

    thermal_parameters.columns = thermal_parameters.columns.rename(['Thermal_param_set','Box'])

    return thermal_parameters.apply(pd.to_numeric)


def get_more_gas_cycle_params(N,choose_gases=['CO2','CH4','N2O'],CH4_forc_feedbacks=False, help=False):
    
    param_defaults = get_gas_parameter_defaults(choose_gases=choose_gases,CH4_forc_feedbacks=CH4_forc_feedbacks)

    param_uncert = pd.read_pickle(Path(__file__).parent / "./Parameter_Sets/Complete_parameter_uncertainty.p")

    param_ensemble = pd.concat(N*[param_defaults['default']],keys=['gas'+str(x) for x in np.arange(N)],axis=1)

    for gas in choose_gases:

        for param in param_defaults.index:

            select_param = param_uncert.loc[param,('default',gas)]

            if select_param:

                param_sample = select_param[0].rvs(*select_param[1],N)

                param_ensemble.loc[param,(slice(None),gas)] = param_sample
                
    return param_ensemble


def get_more_thermal_params(N=100,F_2x=3.84):
    
    from copulas.multivariate import GaussianMultivariate
    
    d1_d2_q1_copula = GaussianMultivariate.load(Path(__file__).parent / "./Parameter_Sets/d1_d2_q1_CMIP6_copula.pkl")

    d1_d2_q1_df = d1_d2_q1_copula.sample(10*N)

    while (d1_d2_q1_df<0).any(axis=1).sum() != 0:
        d1_d2_q1_df.loc[(d1_d2_q1_df<0).any(axis=1)] = d1_d2_q1_copula.sample((d1_d2_q1_df<0).any(axis=1).sum()).values

    d2_samples = d1_d2_q1_df['d2'].values
    d3_samples = d1_d2_q1_df['d1'].values
    q3_samples = d1_d2_q1_df['q1'].values

    d1_samples = sp.stats.truncnorm(-2,2,loc=283,scale=116).rvs(10*N)

    TCR_samples = np.random.lognormal(np.log(2.5)/2,np.log(2.5)/(2*1.645),10*N)
    RWF_samples = sp.stats.truncnorm(-2.75,2.75,loc=0.582,scale=0.06).rvs(10*N)
    ECS_samples = TCR_samples/RWF_samples

    d = np.array([d1_samples,d2_samples,d3_samples])

    k = 1-(d/70)*(1-np.exp(-70/d))

    q = ((TCR_samples/F_2x - k[2]*q3_samples)[np.newaxis,:] - np.roll(k[:2],axis=0,shift=1)*(ECS_samples/F_2x - q3_samples)[np.newaxis,:])/(k[:2] - np.roll(k[:2],axis=0,shift=1))

    sample_df = pd.DataFrame(index=['d','q'],columns = [1,2,3]).apply(pd.to_numeric)
    df_list = []

    i=0
    j=0

    while j<N:

        curr_df = sample_df.copy()
        curr_df.loc['d'] = d[:,i]
        curr_df.loc['q',3] = q3_samples[i]
        curr_df.loc['q',[1,2]] = q[:,i]

        if curr_df.loc['q',2]<=0:
            i+=1
            continue

        df_list += [curr_df]
        j+=1
        i+=1

    thermal_params = pd.concat(df_list,axis=1,keys=['therm'+str(x) for x in np.arange(N)])
    
    return thermal_params


def tcr_ecs_to_q(input_parameters=True , F_2x=3.84 , help=False):

	# converts a 2-box tcr / ecs / d dataframe into a d / q dataframe for use in GIR
	# F2x is the GIR default forcing parameter value

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

def q_to_tcr_ecs(input_parameters=True , F_2x=3.84 , help=False):

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


def run_GIR( emissions_in = False , \
				concentrations_in = False , \
				forcing_in = False , \
				gas_parameters = get_gas_parameter_defaults() , \
				thermal_parameters = get_thermal_parameter_defaults() , \
				show_run_info = True ):

	# Determine the number of scenario runs , parameter sets , gases , integration period, timesteps

	# There are 3 modes : emissions_driven , concentration_driven & emission_concentration_switch

	# The model differentiates between these as follows: it assumes you input them in the correct format of multiindex df, and scenarios that match for the case of emissions_concentrations_switch:

	if not concentrations_in is False: # check if concentration driven
		concentration_driven = True
		if emissions_in is False: # make sure pure concentration driven
			emissions_in = return_empty_emissions(concentrations_in,gases_in=concentrations_in.columns.levels[1])
			emissions_concentration_switch = False
			time_index = concentrations_in.index
		else: # otherwise emissions -> concentration run
			emissions_concentration_switch = True
			time_index = pd.Index(sorted(list(set(concentrations_in.index.append(emissions_in.index)))))
	else: # emissions only
		concentration_driven=False
		emissions_concentration_switch = False
		time_index = emissions_in.index

	dim_scenario,scen_names = emissions_in.columns.levels[0].size,list(emissions_in.columns.levels[0])
	dim_gas_param,gas_set_names = gas_parameters.columns.levels[0].size,list(gas_parameters.columns.levels[0])
	dim_thermal_param,thermal_set_names = thermal_parameters.columns.levels[0].size,list(thermal_parameters.columns.levels[0])
	n_gas,gas_names = gas_parameters.columns.levels[1].size,list(gas_parameters.columns.levels[1])
	n_year = time_index.size

	names_list = [scen_names,gas_set_names,thermal_set_names,gas_names]
	names_titles = ['Scenario','Gas cycle set','Thermal set','Gas name']

	timestep = np.append(np.diff(time_index)[0],np.diff(time_index))

	# Reformat inputs into the right shape, first sorting the scenarios and gases to ensure everything matches up

	gas_parameters = gas_parameters.reindex(gas_names,axis=1,level=1)
	emissions_in = emissions_in.reindex(scen_names,axis=1,level=0).reindex(gas_names,axis=1,level=1)
	emissions = input_to_numpy(emissions_in)[:,np.newaxis,np.newaxis,...]

	if concentration_driven:
		concentrations_in = concentrations_in.reindex(scen_names,axis=1,level=0).reindex(gas_names,axis=1,level=1)
		if emissions_concentration_switch:
			concentrations = input_to_numpy(concentrations_in.loc[:emissions_in.index[0]-1e-8])[:,np.newaxis,np.newaxis,...] # only want concentrations UP TO the start of emissions
		else:
			concentrations = input_to_numpy(concentrations_in)[:,np.newaxis,np.newaxis,...]
            
	# If thermal parameter names are identical to gas parameter names, assume they are dependent (correspond exactly)
    
	if gas_set_names == thermal_set_names:
		d = input_to_numpy(thermal_parameters.loc[['d']])[np.newaxis,:,np.newaxis,...,0]
		q = input_to_numpy(thermal_parameters.loc[['q']])[np.newaxis,:,np.newaxis,...,0]
		dim_thermal_param=1
		names_list = [scen_names,gas_set_names,gas_names]
		names_titles = ['Scenario','Parameter set','Gas name']
	else:
		d = input_to_numpy(thermal_parameters.loc[['d']])[np.newaxis,np.newaxis,...,0]
		q = input_to_numpy(thermal_parameters.loc[['q']])[np.newaxis,np.newaxis,...,0]

	if show_run_info:
		print('Integrating ' + str(dim_scenario) + ' scenarios, ' + str(dim_gas_param) + ' gas cycle parameter sets, ' + str(dim_thermal_param) + ' independent thermal response parameter sets, over ' + str(list(emissions_in.columns.levels[1])) + ', between ' + str(time_index[0]) + ' and ' + str(time_index[-1]) + '...')
        
	if forcing_in is False:
		ext_forcing = np.zeros((dim_scenario,dim_gas_param,dim_thermal_param,1,n_year))
	else:
		forcing_in = forcing_in.reindex(scen_names,axis=1,level=0)
		ext_forcing = input_to_numpy(forcing_in)[:,np.newaxis,np.newaxis,...]

	# Slice the parameter sets into numpy arrays of the right shape
	# Dimensions : [scenario, gas params, thermal params, gas, time, (gas/thermal pools)]

	a = input_to_numpy(gas_parameters.loc['a1':'a4'])[np.newaxis,:,np.newaxis,...]
	tau = input_to_numpy(gas_parameters.loc['tau1':'tau4'])[np.newaxis,:,np.newaxis,...]
	r = input_to_numpy(gas_parameters.loc['r0':'rA'])[np.newaxis,:,np.newaxis,...]
	emis2conc = input_to_numpy(gas_parameters.loc[['emis2conc']])[np.newaxis,:,np.newaxis,...,0]
	PI_conc = input_to_numpy(gas_parameters.loc[['PI_conc']])[np.newaxis,:,np.newaxis,...,0]
    
	g1 = np.sum( a * tau * ( 1. - ( 1. + 100/tau ) * np.exp(-100/tau) ), axis=-1 )
	g0 = ( np.sinh( np.sum( a * tau * ( 1. - np.exp(-100/tau) ) , axis=-1) / g1 ) )**(-1.)
    
	f = input_to_numpy(gas_parameters.loc['f1':'f3'])[np.newaxis,:,np.newaxis,...]

	# Create appropriate shape variable arrays / calculate RF if concentration driven

	C = np.zeros((dim_scenario,dim_gas_param,dim_thermal_param,n_gas,n_year))
	RF = np.zeros((dim_scenario,dim_gas_param,dim_thermal_param,n_gas,n_year))
	T = np.zeros((dim_scenario,dim_gas_param,dim_thermal_param,n_year))
	alpha = np.zeros((dim_scenario,dim_gas_param,dim_thermal_param,n_gas,n_year))
	alpha[...,0] = calculate_alpha(G=np.zeros(C[...,0].shape),G_A=np.zeros(C[...,0].shape),T=np.zeros(C[...,0].shape),r=r,g0=g0,g1=g1)
    
	if concentration_driven:

		C[...,:concentrations.shape[-1]] = concentrations.copy()
		RF[...,:concentrations.shape[-1]] = step_forcing(concentrations,PI_conc[...,np.newaxis],f[...,np.newaxis,:])
		S = np.zeros(d.shape)
		for t in np.arange(concentrations.shape[-1]):
			S,T[...,t] = step_temperature(S=S,F=np.sum(RF[...,t],axis=-1)[...,np.newaxis]+ext_forcing[...,t],q=q,d=d,dt=timestep[t])
            
		G = np.zeros((dim_scenario,dim_gas_param,dim_thermal_param,n_gas,n_year))
		diagnosed_emissions, R, G[...,:t+1], G_A = unstep_concentration(C[...,:t+1], T[...,:t+1], a, tau, r, PI_conc, emis2conc, timestep[...,:t+1], concentration_driven = True)
        
		if not emissions_concentration_switch:
			G_A = (C - PI_conc[...,np.newaxis]) / emis2conc[...,np.newaxis]
			for t in np.arange(1,diagnosed_emissions.shape[-1]):
				alpha[...,t] = calculate_alpha(G=G[...,t-1],G_A=G_A[...,t-1],T=T[...,t-1,np.newaxis],r=r,g0=g0,g1=g1)

			C_out = concentrations_in
			E_out = pd.DataFrame(np.moveaxis(diagnosed_emissions,-1,0).reshape(diagnosed_emissions.shape[-1],-1),index = time_index,columns=pd.MultiIndex.from_product(names_list,names=names_titles))

		else:
			new_emissions = np.zeros((dim_scenario,dim_gas_param,dim_thermal_param,n_gas,n_year))
			new_emissions[...,:diagnosed_emissions.shape[-1]] = diagnosed_emissions
			new_emissions[...,diagnosed_emissions.shape[-1]:] = emissions
			emissions = new_emissions.copy()
			concentration_driven = False

	if not concentration_driven:
		G = np.cumsum(emissions,axis=-1)
		C[...,0],R,G_A = step_concentration(R = np.zeros(a.shape),alpha=alpha[...,0,np.newaxis],E=emissions[...,0,np.newaxis],a=a,tau=tau,PI_conc=PI_conc,emis2conc=emis2conc,dt=timestep[0])
		RF[...,0] = step_forcing(C=C[...,0],PI_conc=PI_conc,f=f)
		S,T[...,0] = step_temperature(S=np.zeros(d.shape),F=np.sum(RF[...,0],axis=-1)[...,np.newaxis]+ext_forcing[...,0],q=q,d=d,dt=timestep[0])

		for t in np.arange(1,emissions.shape[-1]):
			alpha[...,t] = calculate_alpha(G=G[...,t-1],G_A=G_A,T=T[...,t-1,np.newaxis],r=r,g0=g0,g1=g1)
			C[...,t],R,G_A = step_concentration(R = R,alpha=alpha[...,t,np.newaxis],E=emissions[...,t,np.newaxis],a=a,tau=tau,PI_conc=PI_conc,emis2conc=emis2conc,dt=timestep[t])
			RF[...,t] = step_forcing(C=C[...,t],PI_conc=PI_conc,f=f)
			S,T[...,t] = step_temperature(S=S,F=np.sum(RF[...,t],axis=-1)[...,np.newaxis]+ext_forcing[...,t],q=q,d=d,dt=timestep[t])

		C_out = pd.DataFrame(np.moveaxis(C,-1,0).reshape(C.shape[-1],-1),index = time_index,columns=pd.MultiIndex.from_product(names_list,names=names_titles))

		if emissions_concentration_switch:
			E_out = pd.DataFrame(np.moveaxis(emissions,-1,0).reshape(emissions.shape[-1],-1),index = time_index,columns=pd.MultiIndex.from_product(names_list,names=names_titles))

		else:
			E_out = emissions_in

	ext_forcing = np.zeros(np.sum(RF,axis=-2)[...,np.newaxis,:].shape) + ext_forcing
	RF = np.concatenate((RF,ext_forcing),axis=-2)
	RF = np.concatenate((RF,np.sum(RF,axis=-2)[...,np.newaxis,:]),axis=-2)
        
	alpha_out = pd.DataFrame(np.moveaxis(alpha,-1,0).reshape(alpha.shape[-1],-1),index = time_index,columns=pd.MultiIndex.from_product(names_list,names=names_titles))
	RF_out = pd.DataFrame(np.moveaxis(RF,-1,0).reshape(RF.shape[-1],-1),index = time_index,columns=pd.MultiIndex.from_product([x+['External','Total']*(x==names_list[-1]) for x in names_list],names=names_titles))
	T_out = pd.DataFrame(np.moveaxis(T,-1,0).reshape(T.shape[-1],-1),index = time_index,columns=pd.MultiIndex.from_product(names_list[:-1],names=names_titles[:-1]))

	out_dict = {'C':C_out, \
				'RF':RF_out, \
				'T':T_out, \
				'alpha':alpha_out, \
				'Emissions':E_out , \
				'gas_parameters':gas_parameters , \
				'thermal parameters':thermal_parameters}

	for axis in [x for x in list(out_dict.keys())[:-2] if type(x)==pd.core.frame.DataFrame]:
		out_dict[axis].index = out_dict[axis].index.rename('Year')

	return out_dict


############################### Advanced Tools #################################


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

	# If this switch is turned on, the model will set the pre-industrial concentration value equal to the first timestep it's given (avoiding initialisation shock)
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


def unstep_forcing(forcing_in,gas_parameters=get_gas_parameter_defaults(),thermal_params=get_thermal_parameter_defaults()):
    
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


def get_cmip6_thermal_params():
    
    JT_params = pd.read_csv(Path(__file__).parent / "./J_Tsutsui_params/2019-09-20_1417/parms_cmip6_20190920.csv")

    JT_params = JT_params.loc[(JT_params.iloc[:,1] == 'tas')&((JT_params.iloc[:,2] == 'irm-2')|(JT_params.iloc[:,2] == 'irm-3'))]

    JT_UnFaIR_params = pd.DataFrame(columns=[1,2,3],index=['d','q'])

    JT_UnFaIR_params.index = JT_UnFaIR_params.index.rename('param_name')

    JT_UnFaIR_params.columns = JT_UnFaIR_params.columns.rename('Box')

    param_list = []

    for i in JT_params.index:

        curr_params = JT_UnFaIR_params.copy()

        curr_params.loc['d'] = (JT_params.loc[i,'tau0':'tau2']).values

        curr_params.loc['q'] = (JT_params.loc[i,'a0':'a2'] / JT_params.loc[i,'lambda']).values

        param_list += [curr_params]

    JT_UnFaIR_params = pd.concat(param_list, keys = JT_params.iloc[:,0]+'_'+JT_params.iloc[:,2], axis = 1)

    JT_UnFaIR_params.columns = JT_UnFaIR_params.columns.rename(['CMIP6-model_IR(n)','Box'])

    JT_UnFaIR_params = JT_UnFaIR_params.apply(pd.to_numeric)

    JT_UnFaIR_params.loc['d',([x for x in JT_UnFaIR_params.columns.levels[0] if 'irm-2' in x],3)] = 1.
    JT_UnFaIR_params.loc['q',([x for x in JT_UnFaIR_params.columns.levels[0] if 'irm-2' in x],3)] = 0
    
    return JT_UnFaIR_params