import pandas as pd
import numpy as np
import scipy as sp

def get_RCMIP_data():
    
    RCMIP_concs = pd.read_csv('https://drive.google.com/u/0/uc?id=1o5LMu-Tw5lhwnPUb68c08HQHmrEXFBck&export=download').set_index(['Region','Scenario','Variable'])
    RCMIP_emms = pd.read_csv('https://drive.google.com/u/0/uc?id=1krA0lficstXqahlNCko7nbfqgjKrd_Sa&export=download').set_index(['Region','Scenario','Variable'])
    RCMIP_forc = pd.read_csv('https://drive.google.com/u/0/uc?id=1-YP2RbchNdGRpGtZjTr0T6CdKZiEshbx&export=download').set_index(['Region','Scenario','Variable'])
    
    return RCMIP_concs, RCMIP_emms, RCMIP_forc

RCMIP_concs, RCMIP_emms, RCMIP_forc = get_RCMIP_data()

def get_GIR_to_RCMIP_map():
    
    ## map new GIR parameters to old names
    RCMIP_emms_name = ['Emissions|F-Gases|PFC|C2F6', 'Emissions|F-Gases|PFC|C3F8', 'Emissions|F-Gases|PFC|C4F10', 'Emissions|F-Gases|PFC|C5F12', 'Emissions|F-Gases|PFC|C6F14', 'Emissions|F-Gases|PFC|C7F16'\
                      , 'Emissions|F-Gases|PFC|C8F18', 'Emissions|F-Gases|PFC|cC4F8'\
                      ,'Emissions|CO2', 'Emissions|Montreal Gases|CCl4', 'Emissions|F-Gases|PFC|CF4','Emissions|Montreal Gases|CFC|CFC113','Emissions|Montreal Gases|CFC|CFC114','Emissions|Montreal Gases|CFC|CFC115'
                      ,'Emissions|Montreal Gases|CFC|CFC11','Emissions|Montreal Gases|CFC|CFC12','Emissions|Montreal Gases|CH2Cl2','Emissions|Montreal Gases|CH3CCl3'\
                      ,'Emissions|Montreal Gases|CHCl3'\
                      ,'Emissions|Montreal Gases|Halon1211','Emissions|Montreal Gases|Halon1301','Emissions|Montreal Gases|Halon2402','Emissions|Montreal Gases|Halon1202'\
                      ,'Emissions|Montreal Gases|HCFC141b','Emissions|Montreal Gases|HCFC142b','Emissions|Montreal Gases|HCFC22'\
                      ,'Emissions|F-Gases|HFC|HFC125','Emissions|F-Gases|HFC|HFC134a','Emissions|F-Gases|HFC|HFC143a','Emissions|F-Gases|HFC|HFC152a','Emissions|F-Gases|HFC|HFC227ea'\
                      ,'Emissions|F-Gases|HFC|HFC236fa','Emissions|F-Gases|HFC|HFC23','Emissions|F-Gases|HFC|HFC245fa','Emissions|F-Gases|HFC|HFC32','Emissions|F-Gases|HFC|HFC365mfc','Emissions|F-Gases|HFC|HFC4310mee'\
                      ,'Emissions|CH4','Emissions|Montreal Gases|CH3Br','Emissions|Montreal Gases|CH3Cl','Emissions|F-Gases|NF3','Emissions|N2O','Emissions|F-Gases|SF6'\
                      ,'Emissions|F-Gases|SO2F2','Emissions|Sulfur','Emissions|NOx','Emissions|CO','Emissions|VOC','Emissions|BC','Emissions|NH3','Emissions|OC']

    GIR_name = ['c2f6', 'c3f8', 'c4f10', 'c5f12', 'c6f14', 'c7f16', 'c8f18', 'c_c4f8',\
                'carbon_dioxide', 'carbon_tetrachloride', 'cf4', 'cfc113', 'cfc114', 'cfc115', 'cfc11', 'cfc12', 'ch2cl2', 'ch3ccl3', 'chcl3',\
                'halon1211', 'halon1301', 'halon2402', 'halon1202', 'hcfc141b', 'hcfc142b', 'hcfc22',\
                'hfc125', 'hfc134a', 'hfc143a', 'hfc152a', 'hfc227ea', 'hfc236fa', 'hfc23', 'hfc245fa', 'hfc32', 'hfc365mfc', 'hfc4310mee',\
                'methane', 'methyl_bromide', 'methyl_chloride', 'nf3', 'nitrous_oxide', 'sf6', 'so2f2', 'so2', 'nox', 'co', 'nmvoc','bc','nh3','oc']

    RCMIP_to_GIR_map_emms = dict(zip(RCMIP_emms_name,GIR_name))

    RCMIP_to_GIR_map_concs = dict(zip(['Atmospheric Concentrations|'+'|'.join(x.split('|')[1:]) for x in RCMIP_to_GIR_map_emms.keys()],GIR_name))

    GIR_to_RCMIP_map = pd.DataFrame(index=GIR_name)
    GIR_to_RCMIP_map['native_emms_unit'] = 'Mt'
    GIR_to_RCMIP_map.loc['carbon_dioxide','native_emms_unit'] = 'GtC'
    GIR_to_RCMIP_map.loc['nitrous_oxide','native_emms_unit'] = 'MtN2O-N2'
    GIR_to_RCMIP_map.loc['nox','native_emms_unit'] = 'MtNO2'
    
    GIR_to_RCMIP_map['native_concs_unit'] = 'ppb'
    GIR_to_RCMIP_map.loc['carbon_dioxide','native_concs_unit'] = 'ppm'
    GIR_to_RCMIP_map.loc[['so2','nox','co','nmvoc','bc','nh3','oc'],'native_concs_unit'] = 'Mt'
    GIR_to_RCMIP_map.loc['nox','native_concs_unit'] = 'MtNO2'

    GIR_to_RCMIP_map.loc[RCMIP_to_GIR_map_emms.values(),'RCMIP_emms_key'] = list(RCMIP_to_GIR_map_emms.keys())
    GIR_to_RCMIP_map.loc[RCMIP_to_GIR_map_emms.values(),'RCMIP_emms_unit'] = RCMIP_emms.loc[('World','ssp245'),['Unit']].loc[RCMIP_to_GIR_map_emms.keys()].values
    GIR_to_RCMIP_map.loc[RCMIP_to_GIR_map_emms.values(),'RCMIP_emms_scaling'] = 1/1000
    GIR_to_RCMIP_map.loc[['so2','nox','co','nmvoc','bc','nh3','oc'],'RCMIP_emms_scaling'] = 1
    GIR_to_RCMIP_map.loc['nitrous_oxide','RCMIP_emms_scaling'] = 28/(44*1000)
    GIR_to_RCMIP_map.loc['methane','RCMIP_emms_scaling'] = 1
    GIR_to_RCMIP_map.loc['carbon_dioxide','RCMIP_emms_scaling'] = 12/(44.01*1000)

    GIR_to_RCMIP_map.loc[RCMIP_to_GIR_map_concs.values(),'RCMIP_concs_key'] = list(RCMIP_to_GIR_map_concs.keys())
    GIR_to_RCMIP_map.loc[RCMIP_to_GIR_map_concs.values(),'RCMIP_concs_unit'] = RCMIP_concs.loc[('World','ssp245')].reindex(RCMIP_to_GIR_map_concs.keys()).loc[:,'Unit'].values#.loc[('World','ssp245',RCMIP_to_GIR_map_concs.keys()),'Unit'].values
    GIR_to_RCMIP_map.loc[RCMIP_to_GIR_map_concs.values(),'RCMIP_concs_scaling'] = 1/1000
    GIR_to_RCMIP_map.loc['nitrous_oxide','RCMIP_concs_scaling'] = 1
    GIR_to_RCMIP_map.loc['methane','RCMIP_concs_scaling'] = 1
    GIR_to_RCMIP_map.loc['carbon_dioxide','RCMIP_concs_scaling'] = 1
    GIR_to_RCMIP_map.loc[['so2','nox','co','nmvoc','bc','nh3','oc'],'RCMIP_concs_scaling'] = np.nan
    
    return GIR_to_RCMIP_map

GIR_to_RCMIP_map = get_GIR_to_RCMIP_map()

def get_GIR_to_RCMIP_map_forc():
    GIR_to_RCMIP_map_forc = pd.DataFrame(index=GIR_to_RCMIP_map.index)
    GIR_to_RCMIP_map_forc['RCMIP_forc_key'] = GIR_to_RCMIP_map.loc[GIR_to_RCMIP_map_forc.index,'RCMIP_concs_key'].str.replace('Atmospheric Concentrations','Effective Radiative Forcing|Anthropogenic').values
    GIR_to_RCMIP_map_forc.loc['bc','RCMIP_forc_key'] = 'Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Fossil and Industrial|BC and OC|BC'
    GIR_to_RCMIP_map_forc.loc['oc','RCMIP_forc_key'] = 'Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Fossil and Industrial|BC and OC|OC'
    GIR_to_RCMIP_map_forc.loc['nh3','RCMIP_forc_key'] = 'Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|NH3|Fossil and Industrial'
    GIR_to_RCMIP_map_forc.loc['aci','RCMIP_forc_key'] = 'Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-cloud Interactions'
    GIR_to_RCMIP_map_forc.loc['trop_o3','RCMIP_forc_key'] = 'Effective Radiative Forcing|Anthropogenic|Tropospheric Ozone'
    GIR_to_RCMIP_map_forc.loc['strat_o3','RCMIP_forc_key'] = 'Effective Radiative Forcing|Anthropogenic|Stratospheric Ozone'
    GIR_to_RCMIP_map_forc.loc['bc_on_snow','RCMIP_forc_key'] = 'Effective Radiative Forcing|Anthropogenic|Other|BC on Snow'
    GIR_to_RCMIP_map_forc.loc['strat_h2o','RCMIP_forc_key'] = 'Effective Radiative Forcing|Anthropogenic|Other|CH4 Oxidation Stratospheric H2O'
    GIR_to_RCMIP_map_forc.loc['so2','RCMIP_forc_key'] = 'Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Fossil and Industrial|Sulfate'
    GIR_to_RCMIP_map_forc.loc['Total','RCMIP_forc_key'] = 'Effective Radiative Forcing'
    GIR_to_RCMIP_map_forc.loc['External','RCMIP_forc_key'] = 'Effective Radiative Forcing|External'
    return GIR_to_RCMIP_map_forc

GIR_to_RCMIP_map_forc = get_GIR_to_RCMIP_map_forc()

def RCMIP_to_GIR_input_emms(scenario):
    _out = RCMIP_emms.loc[('World',scenario,GIR_to_RCMIP_map.RCMIP_emms_key)].droplevel(level=[0,1]).iloc[:,4:]
    _out_index = _out.index.map({v: k for k, v in GIR_to_RCMIP_map.RCMIP_emms_key.to_dict().items()})
    _out.set_index(_out_index,inplace=True)
    _out = _out.T.mul(GIR_to_RCMIP_map.RCMIP_emms_scaling)
    _out.index = _out.index.astype(int)
    return _out.apply(pd.to_numeric)

def get_RCMIP_forc(scenario,drivers=['Effective Radiative Forcing|Anthropogenic|Albedo Change','Effective Radiative Forcing|Anthropogenic|Other|Contrails and Contrail-induced Cirrus','Effective Radiative Forcing|Natural']):
    
    # returns the sum of specified driving rfs (by default those not included in GIR):
    _out = RCMIP_forc.loc[('World',scenario,drivers)].droplevel(level=[0,1]).iloc[:,4:]
    _out = pd.DataFrame(_out.sum(axis=0,skipna=False).values,index=_out.columns.astype(int),columns=['forcing'])
    return _out

def RCMIP_to_GIR_input_concs(scenario):
    _out = RCMIP_concs.loc[('World',scenario,GIR_to_RCMIP_map.RCMIP_concs_key)].droplevel(level=[0,1]).iloc[:,4:]
    _out_index = _out.index.map({v: k for k, v in GIR_to_RCMIP_map.RCMIP_concs_key.to_dict().items()})
    _out.set_index(_out_index,inplace=True)
    _out = _out.T.mul(GIR_to_RCMIP_map.RCMIP_concs_scaling)
    _out.index = _out.index.astype(int)
    return _out.apply(pd.to_numeric)