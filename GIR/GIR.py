# GIR - Nick Leach and Stuart Jenkins

import numpy as np
import pandas as pd
import numexpr as ne
import scipy as sp
from pathlib import Path
from tqdm import tqdm

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
        
        CHOOSE_params.loc['f2',('default','CH4')] += 0.000182 + 5.4e-05 # add on the indirect forcings
        
    elif CH4_forc_feedbacks=='ozone_parameterisation':
        
        CHOOSE_params.loc['f2',('default','CH4')] += 3.7e-04 + 6.9e-05 - 4.6e-05 # add on the indirect forcings
        
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


def tcr_ecs_to_q(input_parameters=True , F_2x=3.76 , help=False):

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
		k = 1.0 - (param_arr[:,:,0]/69.66)*(1.0 - np.exp(-69.66/param_arr[:,:,0]))
		output_params.loc['q'] = ( ( param_arr[:,0,1][:,np.newaxis] - param_arr[:,1,1][:,np.newaxis] * np.roll(k,shift=1) )/( F_2x * ( k - np.roll(k,shift=1) ) ) ) .flatten()

		return output_params.loc[['d','q']]

def q_to_tcr_ecs(input_parameters=True , F_2x=3.76 , help=False):

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

			TCR = F_2x * ( params.loc['q'] * (1 - (params.loc['d']/69.66) * ( 1 - np.exp(-69.66/params.loc['d']) ) ) ).sum()

			output_params.loc[:,param_set] = [ECS,TCR]

		return output_params

def calculate_alpha(G,G_A,T,r,g0,g1,iirf100_max = False):

#     iirf100_val = r[...,0] + r[...,1] * (G-G_A) + r[...,2] * T + r[...,3] * G_A
#     iirf100_val = np.abs(iirf100_val)
#     if iirf100_max:
#         iirf100_val = (iirf100_val>iirf100_max) * iirf100_max + iirf100_val * (iirf100_val<iirf100_max)
#     alpha_val = g0 * np.sinh(iirf100_val / g1)

    iirf100_val = ne.evaluate("abs(r0 + rU * (G-G_A) + rT * T + rA * G_A)",{'r0':r[...,0],'rU':r[...,1],'rT':r[...,2],'rA':r[...,3],'G':G,'G_A':G_A,'T':T})
    if iirf100_max:
        iirf100_val = ne.evaluate("where(iirf100_val>iirf100_max,iirf100_max,iirf100_val)")
    alpha_val = ne.evaluate("g0 * sinh(iirf100_val / g1)")

    return alpha_val

def step_concentration(R_old,G_A_old,E,alpha,a,tau,PI_conc,emis2conc,dt=1):
    
#     decay_rate = dt/(alpha*tau)
#     decay_factor = np.exp( -decay_rate )
#     R_new = E * a * 1/decay_rate * ( 1. - decay_factor ) + R_old * decay_factor
#     G_A = np.sum(R_new,axis=-1)
#     C = PI_conc + emis2conc * (G_A + G_A_old) / 2

    decay_rate = ne.evaluate("dt/(alpha*tau)")
    decay_factor = ne.evaluate("exp(-decay_rate)")
    R_new = ne.evaluate("E * a / decay_rate * ( 1. - decay_factor ) + R_old * decay_factor")
    G_A = ne.evaluate("sum(R_new,axis=4)")
    C = ne.evaluate("PI_conc + emis2conc * (G_A + G_A_old) / 2")

    return C,R_new,G_A

def unstep_concentration(R_old,C,alpha,a,tau,PI_conc,emis2conc,dt=1):

    E = ( C - PI_conc - np.sum(R_old * np.exp( -dt/(alpha*tau) ) , axis=-1 ) ) / ( emis2conc * np.sum( a * alpha * ( tau / dt ) * ( 1. - np.exp( -dt / ( alpha * tau ) ) ) , axis=-1 ) )

    R_new = E[...,np.newaxis] * emis2conc[...,np.newaxis] * a * alpha * (tau/dt) * ( 1. - np.exp( -dt/(alpha*tau) ) ) + R_old * np.exp( -dt/(alpha * tau) )

    G_A = np.sum(R_new,axis=-1) / emis2conc

    return E,R_new,G_A

def step_forcing(C,PI_conc,f):
    
    # if the logarithmic/sqrt term is undefined (ie. C is zero or negative), this contributes zero to the overall forcing. An exception will appear, however.

#     logforc = f[...,0] * np.log(C / PI_conc)
#     linforc = f[...,1] * ( C - PI_conc )
#     sqrtforc = f[...,2] * (np.sqrt(C) - np.sqrt(PI_conc))
#     logforc[np.isnan(logforc)] = 0
#     sqrtforc[np.isnan(sqrtforc)] = 0

    logforc = ne.evaluate("f1 * where( (C/PI_conc) <= 0, 0, log(C/PI_conc) )",{'f1':f[...,0],'C':C,'PI_conc':PI_conc})
    linforc = ne.evaluate("f2 * (C - PI_conc)",{'f2':f[...,1],'C':C,'PI_conc':PI_conc})
    sqrtforc = ne.evaluate("f3 * ( (sqrt( where(C<0 ,0 ,C ) ) - sqrt(PI_conc)) )",{'f3':f[...,2],'C':C,'PI_conc':PI_conc})

    RF = logforc + linforc + sqrtforc

    return RF

def step_temperature(S_old,F,q,d,dt=1):

#     decay_factor = np.exp(-dt/d)
#     S_new = q * F * ( 1 - decay_factor ) + S_old * decay_factor
#     T = np.sum(S_old + S_new,axis=-1) / 2
    
    decay_factor = ne.evaluate("exp(-dt/d)")
    S_new = ne.evaluate("q * F * (1 - decay_factor) + S_old * decay_factor")
    T = ne.evaluate("sum( (S_old + S_new)/2, axis=3 )")

    return S_new,T

def run_GIR( emissions_in = False , concentrations_in = False , forcing_in = False , gas_parameters = get_gas_parameter_defaults() , thermal_parameters = get_thermal_parameter_defaults() , show_run_info = True ):

    # Determine the number of scenario runs , parameter sets , gases , integration period, timesteps

    # There are 2 modes : emissions_driven , concentration_driven

    # The model will assume if both are given then emissions take priority

    if emissions_in is False: # check if concentration driven
        concentration_driven = True
        emissions_in = return_empty_emissions(concentrations_in,gases_in=concentrations_in.columns.levels[1])
        time_index = concentrations_in.index
    else: # otherwise emissions driven
        concentration_driven=False
        time_index = emissions_in.index

    [(dim_scenario,scen_names),(dim_gas_param,gas_set_names),(dim_thermal_param,thermal_set_names)]=[(x.size,list(x)) for x in [emissions_in.columns.levels[0],gas_parameters.columns.levels[0],thermal_parameters.columns.levels[0]]]
    gas_names = [x for x in gas_parameters.columns.levels[1] if '|' not in x]
    n_gas = len(gas_names)
    n_forc,forc_names = gas_parameters.columns.levels[1].size,list(gas_parameters.columns.levels[1])
    n_year = time_index.size

    ## map the concentrations onto the forcings (ie. so the correct indirect forcing parameters read the correct concentration arrays)
    gas_forc_map = [gas_names.index(forc_names[x].split('|')[0]) for x in np.arange(len(forc_names))]

    names_list = [scen_names,gas_set_names,thermal_set_names,gas_names]
    names_titles = ['Scenario','Gas cycle set','Thermal set','Gas name']
    forc_names_list = [scen_names,gas_set_names,thermal_set_names,forc_names]
    forc_names_titles = ['Scenario','Gas cycle set','Thermal set','Forcing component']

    timestep = np.append(np.diff(time_index),np.diff(time_index)[-1])

    # check if no dimensions are degenerate
    if (set(scen_names) != set(gas_set_names))&(set(scen_names) != set(thermal_set_names))&(set(gas_set_names) != set(thermal_set_names)):
        gas_shape, gas_slice = [1,dim_gas_param,1],gas_set_names
        therm_shape, therm_slice = [1,1,dim_thermal_param],thermal_set_names
    # check if all degenerate
    elif (set(scen_names) == set(gas_set_names))&(set(scen_names) == set(thermal_set_names)):
        gas_shape, gas_slice = [dim_scenario,1,1],scen_names
        therm_shape, therm_slice = [dim_scenario,1,1],scen_names
        dim_gas_param = 1
        dim_thermal_param = 1
        [x.pop(1) for x in [names_list,names_titles,forc_names_list,forc_names_titles]]
        [x.pop(1) for x in [names_list,names_titles,forc_names_list,forc_names_titles]]
    # check other possibilities
    else:
        if set(scen_names) == set(gas_set_names):
            gas_shape, gas_slice = [dim_scenario,1,1],scen_names
            therm_shape, therm_slice = [1,1,dim_thermal_param],thermal_set_names
            dim_gas_param = 1
            [x.pop(1) for x in [names_list,names_titles,forc_names_list,forc_names_titles]]
        elif set(scen_names) == set(thermal_set_names):
            gas_shape, gas_slice = [1,dim_gas_param,1],gas_set_names
            therm_shape, therm_slice = [dim_scenario,1,1],scen_names
            dim_thermal_param = 1
            [x.pop(2) for x in [names_list,names_titles,forc_names_list,forc_names_titles]]
        else:
            gas_shape, gas_slice = [1,dim_gas_param,1],gas_set_names
            therm_shape, therm_slice = [1,dim_gas_param,1],gas_set_names
            dim_thermal_param = 1
            [x.pop(2) for x in [names_list,names_titles,forc_names_list,forc_names_titles]]

    ## Reindex to align columns:

    emissions = emissions_in.reindex(scen_names,axis=1,level=0).reindex(gas_names,axis=1,level=1).values.T.reshape(dim_scenario,1,1,n_gas,n_year)

    if forcing_in is False:
        ext_forcing = np.zeros((dim_scenario,1,1,1,n_year))
    else:
        forcing_in = forcing_in.reindex(scen_names,axis=1,level=0)
        ext_forcing = forcing_in.loc[:,(scen_names,slice(None))].values.T.reshape(dim_scenario,1,1,1,n_year)

    if concentration_driven:
        concentrations_in = concentrations_in.reindex(scen_names,axis=1,level=0).reindex(gas_names,axis=1,level=1)

    gas_cycle_parameters = gas_parameters.reindex(gas_slice,axis=1,level=0).reindex(gas_names,axis=1,level=1)
    thermal_parameters = thermal_parameters.reindex(therm_slice,axis=1,level=0)

    ## get parameter arrays
    a,tau,r,PI_conc,emis2conc=[gas_cycle_parameters.loc[x].values.T.reshape(gas_shape+[n_gas,-1]) for x in [['a1','a2','a3','a4'],['tau1','tau2','tau3','tau4'],['r0','rC','rT','rA'],'PI_conc','emis2conc']]
    f = gas_parameters.reindex(forc_names,axis=1,level=1).loc['f1':'f3'].values.T.reshape(gas_shape+[n_forc,-1])
    d,q = [thermal_parameters.loc[x].values.T.reshape(therm_shape+[-1]) for x in ['d','q']]

    if show_run_info:
        print('Integrating ' + str(dim_scenario) + ' scenarios, ' + str(dim_gas_param) + ' gas cycle parameter sets, ' + str(dim_thermal_param) + ' independent thermal response parameter sets, over ' + str(list(emissions_in.columns.levels[1])) + ', between ' + str(time_index[0]) + ' and ' + str(time_index[-1]) + '...',flush=True)

    # Dimensions : [scenario, gas params, thermal params, gas, time, (gas/thermal pools)]

    g1 = np.sum( a * tau * ( 1. - ( 1. + 100/tau ) * np.exp(-100/tau) ), axis=-1 )
    g0 = ( np.sinh( np.sum( a * tau * ( 1. - np.exp(-100/tau) ) , axis=-1) / g1 ) )**(-1.)

    # Create appropriate shape variable arrays / calculate RF if concentration driven

    C = np.empty((dim_scenario,dim_gas_param,dim_thermal_param,n_gas,n_year))
    RF = np.empty((dim_scenario,dim_gas_param,dim_thermal_param,n_forc,n_year))
    T = np.empty((dim_scenario,dim_gas_param,dim_thermal_param,n_year))
    alpha = np.empty((dim_scenario,dim_gas_param,dim_thermal_param,n_gas,n_year))
    alpha[...,0] = calculate_alpha(G=0,G_A=0,T=0,r=r,g0=g0,g1=g1)

    if concentration_driven:

        diagnosed_emissions = np.empty((dim_scenario,dim_gas_param,dim_thermal_param,n_gas,n_year))
        C[:] = concentrations_in.reindex(scen_names,axis=1,level=0).reindex(gas_names,axis=1,level=1).values.T.reshape(dim_scenario,1,1,n_gas,n_year)
        C_end = np.empty(C.shape)
        RF[:] = step_forcing(C[...,gas_forc_map,:],PI_conc[...,gas_forc_map,:],f[...,np.newaxis,:])
        C_end[...,0] = C[...,0]*2 - PI_conc[...,0]
        diagnosed_emissions[...,0],R,G_A = unstep_concentration(R_old=np.zeros(a.shape),C=C_end[...,0],alpha=alpha[...,0,np.newaxis],a=a,tau=tau,PI_conc=PI_conc[...,0],emis2conc=emis2conc[...,0],dt=timestep[0])
        S,T[...,0] = step_temperature(S_old=np.zeros(d.shape),F=np.sum(RF[...,0],axis=-1)[...,np.newaxis]+ext_forcing[...,0],q=q,d=d,dt=timestep[0])
        for t in tqdm(np.arange(1,n_year),unit=' timestep'):
            G = np.sum(diagnosed_emissions,axis=-1)
            alpha[...,t] = calculate_alpha(G=G,G_A=G_A,T=np.sum(S,axis=-1)[...,np.newaxis],r=r,g0=g0,g1=g1)
            C_end[...,t] = C[...,t]*2 - C_end[...,t-1]
            diagnosed_emissions[...,t],R,G_A = unstep_concentration(R_old=R,C=C_end[...,t],alpha=alpha[...,t,np.newaxis],a=a,tau=tau,PI_conc=PI_conc[...,0],emis2conc=emis2conc[...,0],dt=timestep[t])
            S,T[...,t] = step_temperature(S_old=S,F=np.sum(RF[...,t],axis=-1)[...,np.newaxis]+ext_forcing[...,t],q=q,d=d,dt=timestep[t])

        C_out = concentrations_in
        E_out = pd.DataFrame(np.moveaxis(diagnosed_emissions,-1,0).reshape(diagnosed_emissions.shape[-1],-1),index = time_index,columns=pd.MultiIndex.from_product(names_list,names=names_titles))

    if not concentration_driven:
        G = np.cumsum(emissions,axis=-1)
        C[...,0],R,G_A = step_concentration(R_old = 0,G_A_old = 0,alpha=alpha[...,0,np.newaxis],E=emissions[...,0,np.newaxis],a=a,tau=tau,PI_conc=PI_conc[...,0],emis2conc=emis2conc[...,0],dt=timestep[0])
        RF[...,0] = step_forcing(C=C[...,gas_forc_map,0],PI_conc=PI_conc[...,gas_forc_map,0],f=f)
        S,T[...,0] = step_temperature(S_old=0,F=np.sum(RF[...,0],axis=-1)[...,np.newaxis]+ext_forcing[...,0],q=q,d=d,dt=timestep[0])

        for t in tqdm(np.arange(1,n_year),unit=' timestep'):
            alpha[...,t] = calculate_alpha(G=G[...,t-1],G_A=G_A,T=np.sum(S,axis=-1)[...,np.newaxis],r=r,g0=g0,g1=g1)
            C[...,t],R,G_A = step_concentration(R_old = R,G_A_old=G_A,alpha=alpha[...,t,np.newaxis],E=emissions[...,t,np.newaxis],a=a,tau=tau,PI_conc=PI_conc[...,0],emis2conc=emis2conc[...,0],dt=timestep[t])
            RF[...,t] = step_forcing(C=C[...,gas_forc_map,t],PI_conc=PI_conc[...,gas_forc_map,0],f=f)
            S,T[...,t] = step_temperature(S_old=S,F=np.sum(RF[...,t],axis=-1)[...,np.newaxis]+ext_forcing[...,t],q=q,d=d,dt=timestep[t])

        C_out = pd.DataFrame(np.moveaxis(C,-1,0).reshape(C.shape[-1],-1),index = time_index,columns=pd.MultiIndex.from_product(names_list,names=names_titles))
        E_out = emissions_in

    ext_forcing = np.zeros(np.sum(RF,axis=-2)[...,np.newaxis,:].shape) + ext_forcing
    RF = np.concatenate((RF,ext_forcing),axis=-2)
    RF = np.concatenate((RF,np.sum(RF,axis=-2)[...,np.newaxis,:]),axis=-2)

    alpha_out = pd.DataFrame(np.moveaxis(alpha,-1,0).reshape(alpha.shape[-1],-1),index = time_index,columns=pd.MultiIndex.from_product(names_list,names=names_titles))
    RF_out = pd.DataFrame(np.moveaxis(RF,-1,0).reshape(RF.shape[-1],-1),index = time_index,columns=pd.MultiIndex.from_product([x+['External','Total']*(x==forc_names_list[-1]) for x in forc_names_list],names=forc_names_titles))
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


def prescribed_temps_gas_cycle(emissions_in , gas_parameters , T):

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


def invert_concentrations_prescribed_T( concentrations_in,  gas_parameters , T ):
    
    time_index = concentrations_in.index

    [(dim_scenario,scen_names),(dim_gas_param,gas_set_names)]=[(x.size,list(x)) for x in [concentrations_in.columns.levels[0],gas_parameters.columns.levels[0]]]
    gas_names = [x for x in gas_parameters.columns.levels[1] if '|' not in x]
    n_gas = len(gas_names)
    n_year = time_index.size

    names_list = [scen_names,gas_set_names,gas_names]
    names_titles = ['Scenario','Gas cycle set','Gas name']

    timestep = np.append(np.diff(time_index),np.diff(time_index)[-1])
    
    if set(scen_names) == set(gas_set_names):
        gas_shape, gas_slice = [dim_scenario,1],scen_names
        dim_gas_param = 1
        [x.pop(1) for x in [names_list,names_titles]]
    else:
        gas_shape, gas_slice = [1,dim_gas_param],gas_set_names

    a,tau,r,PI_conc,emis2conc=[gas_parameters.loc[x,(gas_slice,gas_names)].values.T.reshape(gas_shape+[n_gas,-1]) for x in [['a1','a2','a3','a4'],['tau1','tau2','tau3','tau4'],['r0','rC','rT','rA'],'PI_conc','emis2conc']]

    # Dimensions : [scenario, gas params, gas, time, (gas/thermal pools)]

    g1 = np.sum( a * tau * ( 1. - ( 1. + 100/tau ) * np.exp(-100/tau) ), axis=-1 )
    g0 = ( np.sinh( np.sum( a * tau * ( 1. - np.exp(-100/tau) ) , axis=-1) / g1 ) )**(-1.)

    # Create appropriate shape variable arrays / calculate RF if concentration driven

    C = np.zeros((dim_scenario,dim_gas_param,n_gas,n_year))
    T = T.values.flatten().reshape(1,1,-1)
    alpha = np.zeros((dim_scenario,dim_gas_param,n_gas,n_year))
    alpha[...,0] = calculate_alpha(G=np.zeros(C[...,0].shape),G_A=np.zeros(C[...,0].shape),T=np.zeros(C[...,0].shape),r=r,g0=g0,g1=g1)

    diagnosed_emissions = np.zeros((dim_scenario,dim_gas_param,n_gas,n_year))
    C[:] = input_to_numpy(concentrations_in.reindex(scen_names,axis=1,level=0).reindex(gas_names,axis=1,level=1))[:,np.newaxis,...]
    C_end = np.zeros(C.shape)
    C_end[...,0] = C[...,0]*2 - PI_conc[...,0]
    diagnosed_emissions[...,0],R,G_A = unstep_concentration(R_old=np.zeros(a.shape),C=C_end[...,0],alpha=alpha[...,0,np.newaxis],a=a,tau=tau,PI_conc=PI_conc[...,0],emis2conc=emis2conc[...,0],dt=timestep[0])
    for t in np.arange(1,n_year):
        G = np.sum(diagnosed_emissions,axis=-1)
        alpha[...,t] = calculate_alpha(G=G,G_A=G_A,T=T[...,t-1,np.newaxis],r=r,g0=g0,g1=g1)
        C_end[...,t] = C[...,t]*2 - C_end[...,t-1]
        diagnosed_emissions[...,t],R,G_A = unstep_concentration(R_old=R,C=C_end[...,t],alpha=alpha[...,t,np.newaxis],a=a,tau=tau,PI_conc=PI_conc[...,0],emis2conc=emis2conc[...,0],dt=timestep[t])

    C_out = concentrations_in
    E_out = pd.DataFrame(np.moveaxis(diagnosed_emissions,-1,0).reshape(diagnosed_emissions.shape[-1],-1),index = time_index,columns=pd.MultiIndex.from_product(names_list,names=names_titles))

    alpha_out = pd.DataFrame(np.moveaxis(alpha,-1,0).reshape(alpha.shape[-1],-1),index = time_index,columns=pd.MultiIndex.from_product(names_list,names=names_titles))

    out_dict = {'C':C_out, \
                'alpha':alpha_out, \
                'Emissions':E_out , \
                'gas_parameters':gas_parameters , \
                'T':T}

    for axis in [x for x in list(out_dict.keys())[:-2] if type(x)==pd.core.frame.DataFrame]:
        out_dict[axis].index = out_dict[axis].index.rename('Year')

    return out_dict

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


## Fitting the r parameters from Emissions and Concentrations __ WIP ##

def OLSE_NORM(X,Y,add_intercept=True):
    
    ## computes a multiple OLS regression over a field against several indices. First dimension is time, second is features (X), or targets (Y)
    
    if add_intercept:
    
        X_1 = np.concatenate((np.ones(X.shape[0])[:,np.newaxis],X),axis=1)
        
    else:
        
        X_1 = X.copy()
    
    B = np.dot( np.linalg.inv( np.dot( X_1.T , X_1 ) ) , np.dot( X_1.T , Y ) )
    
    e = Y - np.dot(X_1,B)
    
    SSE = np.sum(e**2,axis=0)

    MSE_var = SSE / (X_1.shape[0] - X_1.shape[-1])

    SE_B = np.sqrt( np.diag( np.linalg.inv( np.dot( X_1.T , X_1 ) ) )[:,np.newaxis] * MSE_var[np.newaxis,:] )
    
    return {'coefs':B[1:],'coef_err':SE_B[1:],'res':e,'intercept':B[0],'intercept_err':SE_B[0]}

def alpha_root(alpha,R_old,C,E,a,tau,PI_conc,emis2conc,dt=1):
    
    # computes alpha through a root finding algorithm from emissions and concentrations

    return E - ( C - PI_conc - np.sum(R_old * np.exp( -dt/(alpha*tau) ) , axis=-1 ) ) / ( emis2conc * np.sum( a * alpha * ( tau / dt ) * ( 1. - np.exp( -dt / ( alpha * tau ) ) ) , axis=-1 ) )

def get_alpha_from_E_C(C,E,a,tau,PI_conc,emis2conc,timestep=False):
    
    # returns alpha from concentrations and emissions
    
    if timestep is False:
        timestep = np.ones_like(C)
    C_end = np.zeros_like(C)
    alpha = np.zeros_like(C)
    C_calc = np.zeros_like(C)
    G_A = np.zeros_like(C)
    
    R = np.zeros_like(a)
    C_end[0] = C[0]*2 - PI_conc
    alpha[0] = sp.optimize.root(alpha_root,0.1,args=(R,C_end[0],E[0],a,tau,PI_conc,emis2conc,timestep[0]),method='lm').x
    C_calc[0],R,G_A[0] = step_concentration(R_old=R,alpha=alpha[0],E=E[0],a=a,tau=tau,PI_conc=PI_conc,emis2conc=emis2conc,dt=timestep[0])
    
    for t in np.arange(1,C.size):
        C_end[t] = C[t]*2 - C_end[...,t-1]
        alpha[t] = sp.optimize.root(alpha_root,alpha[t-1],args=(R,C_end[t],E[t],a,tau,PI_conc,emis2conc,timestep[t]),method='lm').x
        C_calc[t],R,G_A[t] = step_concentration(R_old=R,alpha=alpha[t],E=E[t],a=a,tau=tau,PI_conc=PI_conc,emis2conc=emis2conc,dt=timestep[t])
        
    return alpha,C_calc,G_A

def fit_r0_rC_rT_rA(C,E,T,a,tau,PI_conc,emis2conc,timestep=False,coefs=['r_U','r_T','r_C']):
    
    # computes alpha from concentrations/emissions and returns the linear fit to specified r parameters
    # Note this only works on timeseries where the concentration remains significantly different to the pre-industrial value
    
    if timestep==False:
        timestep = np.ones_like(C)
    
    alpha,C_calc,G_A = get_alpha_from_E_C(C,E,a,tau,PI_conc,emis2conc,timestep)
    G = np.cumsum(E)
    
    g1 = np.sum( a * tau * ( 1. - ( 1. + 100/tau ) * np.exp(-100/tau) ), axis=-1 )
    g0 = ( np.sinh( np.sum( a * tau * ( 1. - np.exp(-100/tau) ) , axis=-1) / g1 ) )**(-1.)
    
    X = []
    
    if 'r_U' in coefs:
        X += [G-G_A]
    if 'r_T' in coefs:
        X += [T]
    if 'r_C' in coefs:
        X += [G_A]
    
    X = np.array(X).T
    Y = g1*np.arcsinh(alpha/g0)[:,np.newaxis]
    
    # have to shift the X and Y arrays since alpha starts at the PI value & regressors start at the 1st timestep
    _lr = OLSE_NORM(X[:-1],Y[1:])
    
    return pd.Series(dict(zip(['r_0']+coefs,list(_lr['intercept'])+list(_lr['coefs'].flatten()))))

## Extra definition to easily grab the Tsutsui (2020) parameters ##

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





