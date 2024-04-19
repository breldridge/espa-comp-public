#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script for converting WECC generator spreadsheets to resource_info.xlsx
# (which formatted to be called from scheduler.py)
# This script sits in the root directory of espa-comp

# Creates:
# In market_clearing/offer_data:
#     resource_info.xlsx
# In espa-comp root:
#     scale_factors.json

# Requires several files to run successfully:
# In market_clearing/system_data: (you may rename system_dir for a different path)
#     Generators.xlsx
#     Load_2030_8760.csv
#     Generic_MinUpDn_Info.csv
#     Generic_RampUpDn_Info.csv
#     Generic_StartupCost_Info.csv
#
# In root dir: (for espa-comp)
#     ba_names.json
#     topology_conf.json

# NOTE: a lot of choices on resource attributes and bids are hardcoded here

import pandas as pd
import numpy as np
import json
from os.path import join

np.random.seed(1129)

class ResourceMaker:
    def __init__(self, genfile='Generators.xlsx', gensheet='tabled_aggregation',
                 demfile='Load_2030_8760.csv', system_dir='market_clearing/system_data'):
        # Encoding between names
        self.resource_types = {'battery':'str', 'cc_gas':'gen', 'ct_gas':'gen', 'coal':'gen',
                          'hydro':'ren', 'nuclear':'gen', 'ps':'str', 'solar':'ren',
                          'thermal_other':'gen', 'wind':'ren', 'dr':None}
        self.rtypes_to_use = ['battery', 'cc_gas', 'ct_gas', 'coal', 'wind', 'solar', 'hydro',
                              'nuclear']
        with open("ba_names.json", "r") as f:
            bas = json.load(f)
        self.ba_list = sorted(bas["ba"])
        # Read in the WECC generators
        self.generators = pd.read_excel(join(system_dir,genfile), sheet_name=gensheet)
        self.loads = pd.read_csv(join(system_dir,demfile))
        self.updn = pd.read_csv(join(system_dir,'Generic_MinUpDn_Info.csv'),
                                header=1, index_col=False)
        self.ramp = pd.read_csv(join(system_dir,'Generic_RampUpDn_Info.csv'),
                                header=1, index_col=False)
        self.start = pd.read_csv(join(system_dir,'Generic_StartupCost_Info.csv'),
                                 header=1, index_col=False)
        # Read in the selected topology configuration
        with open("topology_conf.json", "r") as f:
            self.topology = json.load(f)
        self._scale_capacities()
        # Make lists for each sheet
        self.master_list = []
        self.gen_list = []
        self.ren_list = []
        self.str_list = []
        self.dem_list = []

    def write_excel(self, fname, fill_dataframes=True):
        '''Writes all dataframes to an excel file'''
        if fill_dataframes:
            self.fill_dataframes()
        sheets = {'Master': self.master_df, 'Generators': self.gen_df, 'Renewables': self.ren_df,
                  'Storage': self.str_df, 'Demand': self.dem_df}
        with pd.ExcelWriter(fname) as writer:
            for key, df in sheets.items():
                df.to_excel(writer, sheet_name=key, index=False)

    def _scale_capacities(self):
        ''' Computes scaling factors for all resources based on selected topology '''
        capacity = self.topology['capacity']
        scaling = self.topology['scaling']
        resource_mix = self.topology['resource_mix']
        cap_factor = self.topology['cap_factor']

        df = self.generators
        drop_inds = []
        for i in range(df.shape[0]):
            if df["LoadAreaName"].values[i] not in self.ba_list:
                drop_inds += [i]
        df = df.drop(drop_inds, axis=0)
        gp_cap = 0
        gp_cap_dict = {}
        rmix_dict = {}
        for rtype in self.rtypes_to_use:
            power = df[df["Gridpath Classification"]==rtype]["PSSEMaxCap"].values
            gp_cap += sum(power)
            gp_cap_dict[rtype] = sum(power)
            if rtype in resource_mix.keys():
                rmix_dict[rtype] = sum(power)
            elif 'gas' in rtype:
                if 'nat_gas' in rmix_dict.keys():
                    rmix_dict['nat_gas'] += sum(power)
                else:
                    rmix_dict['nat_gas'] = sum(power)
            elif rtype == 'hydro' or rtype == 'nuclear' or 'rtype' == 'other':
                rmix_dict['other_ren'] = sum(power)
        # Rescale to find existing percentages
        for key, val in rmix_dict.items():
            rmix_dict[key] = val/gp_cap
        target_cap = {}
        for key, val in resource_mix.items():
            target_cap[key] = val*capacity/scaling
        scale_fact = {}
        for key, targ in target_cap.items():
            if 'gas' in key:
                gas_sf = targ/(gp_cap_dict['cc_gas']+gp_cap_dict['ct_gas'])
                scale_fact['cc_gas'] = gas_sf
                scale_fact['ct_gas'] = gas_sf
            elif key == 'other_ren':
                other_sf = targ/(gp_cap_dict['hydro'] + gp_cap_dict['nuclear']) # + gp_cap_dict['thermal_other']
                scale_fact[key] = other_sf
            elif key == 'flex_dem':
                scale_fact[key] = None
            elif key == 'wind' or key == 'solar':
                new_targ = gp_cap_dict[key]/scaling + (targ-gp_cap_dict[key]/scaling)/cap_factor[key]
                scale_fact[key] = new_targ/gp_cap_dict[key]
            else:
                scale_fact[key] = targ/gp_cap_dict[key]
        # Now set scale for demand
        peak_demand = self.topology["peak_demand"]
        for col_name in self.loads.columns:
            ba = col_name.split('_')[1]
            if ba == 'TH':
                continue
            if ba not in self.ba_list:
                self.loads = self.loads.drop(col_name, axis=1)
        total_demand = self.loads.sum(axis=1)
        scale_fact['flex_dem'] = peak_demand/np.max(total_demand)/scaling
        self.scale_fact = scale_fact
        with open("scale_factors.json", "w") as f:
            json.dump(self.scale_fact, f, indent=4)
        # {key: val*scaling for key, val in scale_fact.items() if val is not None}
        # print("Subset capacity is:", gp_cap)
        # print("By type:", gp_cap_dict)
        # print("Resource percentages:", rmix_dict)
        # print("Target Capacity", target_cap)
        # print("Scaling Factors", self.scale_fact)

    def fill_dataframes(self):
        '''Loops through all WECC generators then populates the appropriate dataframe.'''
        gen_count = self.generators['Unnamed: 0'].values.flatten()
        for cnt in gen_count:
            gentype = self.generators["Gridpath Classification"].iloc[cnt]
            if gentype not in self.rtypes_to_use: # Ignoring dr, ps, thermal_other for now
                continue
            self.add_resource(cnt, gentype)
        cnt = self.add_storage(cnt)
        for ba in self.ba_list:
            cnt += 1
            self.add_demand_resource(cnt, ba)
        # Generators
        gen_col_names = ['rid', 'cost_rgu', 'cost_rgd', 'cost_spr', 'cost_nsp', 'init_en',
                         'init_status','ramp_dn', 'ramp_up', 'cost_su', 'cost_op','cost_sd',
                         'block_g_mq', 'block_g_mc','pgmin', 'pgmax', 'rgumax', 'rgdmax',
                         'min_uptime', 'min_downtime', 'outage','init_downtime', 'init_uptime']
        self.gen_df = pd.DataFrame(self.gen_list, columns=gen_col_names)
        # Renewables
        ren_col_names = ['rid', 'cost_rgu', 'cost_rgd', 'cost_spr', 'cost_nsp', 'init_en',
                         'init_status', 'ramp_up', 'ramp_dn', 'pvmin', 'pvmax',
                         'block_r_mq', 'block_r_mc']
        self.ren_df = pd.DataFrame(self.ren_list, columns=ren_col_names)
        # Storage
        str_col_names = ['rid', 'cost_rgu', 'cost_rgd', 'cost_spr', 'cost_nsp', 'init_en',
                         'init_status', 'ramp_dn', 'ramp_up', 'block_ch_mc', 'block_dc_mc',
                         'block_soc_mc', 'chmax', 'dcmax', 'block_ch_mq', 'block_dc_mq',
                         'block_soc_mq', 'soc_end', 'soc_begin', 'socmax', 'socmin', 'eff_ch',
                         'eff_dc', 'imax', 'imin', 'vmax', 'vmin', 'ch_cap', 'eff_coul',
                         'eff_inv0', 'eff_inv1', 'eff_inv2', 'voc0', 'voc1', 'voc2', 'voc3',
                         'resis', 'therm_cap', 'temp_max', 'temp_min', 'temp_ref', 'Utherm',
                         'deg_DoD0', 'deg_DoD1', 'deg_DoD2', 'deg_DoD3', 'deg_DoD4', 'deg_soc',
                         'deg_therm', 'deg_time', 'cycle_life', 'cost_EoL', 'socref',
                         'soc_capacity', 'cell_count']
        str_dict = dict(zip(str_col_names, self.str_list[0]))
        # str_dict = {str_col_names[i]:self.str_list[i] for i in range(len(str_col_names))}
        with open('battery_attributes.json', 'w') as f:
            json.dump(str_dict, f, indent=2)
        self.str_df = pd.DataFrame(self.str_list, columns=str_col_names)
        # Demand
        dem_col_names = ['rid', 'cost_rgu', 'cost_rgd', 'cost_spr', 'cost_nsp', 'init_en',
                         'init_status', 'ramp_dn', 'ramp_up', 'pdmin', 'pdmax', 'block_d_mq',
                         'block_d_mv']
        self.dem_df = pd.DataFrame(self.dem_list, columns=dem_col_names)
        # Master
        master_col_names = ['rid', 'ResourceName', 'GridpathType', 'ResourceType', 'ResourceBus']
        self.master_df = pd.DataFrame(self.master_list, columns=master_col_names)

    def add_resource(self, cnt, gentype):
        '''Adds the correct quantities to the appropriate list by resource type'''
        geninfo = self.generators.iloc[cnt]
        rtype = self.resource_types[gentype]
        # S
        if rtype != 'str':
            ba_bus = geninfo['LoadAreaName']
            if ba_bus not in self.ba_list:
                return
            # Update the master list first
            self.master_list += [[f'R{cnt+1:05}', f'{ba_bus}_{gentype}', f'{gentype}', f'{rtype}',
                                 f'{ba_bus}']]
            # Next update the list by resource type
            if rtype == 'gen':
                self.update_gen_list(f'R{cnt+1:05}', geninfo, gentype)
            elif rtype == 'ren':
                self.update_ren_list(f'R{cnt+1:05}', geninfo, gentype)
            # elif rtype == 'str':
            #     self.update_str_list(f'R{cnt+1:05}', geninfo)
            else:
                raise ValueError(f'Resource type {rtype} is not available in this simulation.')

    def add_storage(self, cnt):
        ''' Special handling for manually selected storage locations (from topology.conf) '''
        str_bus = self.topology["storage_loc"]["bus"]
        str_num = self.topology["storage_loc"]["num"]
        tot_str = sum(str_num)
        str_idx = 0
        idx_cnt = 0
        for i in range(tot_str):
            cnt += 1
            ba_bus = str_bus[str_idx]
            rid = f'R{cnt+1:05}'
            self.master_list += [[rid, f'{ba_bus}_battery', 'battery', 'str',
                                 f'{ba_bus}']]
            self.update_str_list(rid, tot_str)
            # Once all units are assigned to a BA, move to the next BA and reset counter
            if idx_cnt + 1 == str_num[str_idx]:
                str_idx += 1
                idx_cnt = 0
            else:
                idx_cnt += 1
        return cnt

    def add_demand_resource(self, cnt, ba_bus):
        '''Adds a flexible demand resource for each balancing authority'''
        column_name = [name for name in self.loads.columns.values if ba_bus in name][0]
        pdmax = self.loads.at[0,column_name]*self.scale_fact['flex_dem']
        # Skip BAs with no demand
        if pdmax == 0:
            return
        rid = f'R{cnt+1:05}'
        self.master_list += [[rid, f'{ba_bus}_flex_dem', 'demand', 'dem', f'{ba_bus}']]
        pdmin = 0
        cost_rgu, cost_rgd, cost_spr, cost_nsp = 4, 4, 8, 6
        init_en, init_status = -pdmax, 1
        ramp_up, ramp_dn = 9999, 9999
        dem_block = self.topology["dem_block"]
        # Create a log-uniform distribution from sc_min to sc_max
        sc_min = 0.5
        sc_max = 5
        mv_scale = 10**(np.random.uniform(np.log10(sc_min), np.log10(sc_max)))
        mv, mq = dem_block["mv"], dem_block["mq"]
        if sum(mq) != 1:
            for i, q in enumerate(mq):
                mq[i] = q/sum(mq)
        block_d_mq = ''
        for i, q in enumerate(mq):
            if i != 0:
                block_d_mq += ','
            power = q*pdmax
            block_d_mq += f'{power:.3f}'
        block_d_mv = ''
        for i, v in enumerate(mv):
            if i != 0:
                block_d_mv += ','
                v *= mv_scale
            block_d_mv += f'{v:.1f}'
        self.dem_list += [[rid, cost_rgu, cost_rgd, cost_spr, cost_nsp, init_en, init_status, ramp_dn,
                         ramp_up, pdmin, pdmax, block_d_mq, block_d_mv]]

    def update_gen_list(self, rid, geninfo, gentype):
        '''Updates the generator list based on available info'''
        #cost_su cost_op block_g_mq block_g_mc
        # TODO: figure out where to get regulations costs by gentype
        cost_rgu, cost_rgd, cost_spr, cost_nsp = 3, 3, 0, 0
        if gentype == 'nuclear':
            sf = self.scale_fact['other_ren']
        else:
            sf = self.scale_fact[gentype]
        pgmax = geninfo['PSSEMaxCap']*sf
        # Pgmin estimated as a ratio from pgmax based on GeneratorList_30.xlsx info
        if gentype == 'coal':
            pgmin = 0.42*pgmax
        elif gentype == 'cc_gas':
            pgmin = 0.47*pgmax
        elif gentype == 'ct_gas':
            pgmin = 0.48*pgmax
        elif gentype == 'nuclear':
            pgmin = 0.9*pgmax
        else:
            pgmin = 0.4*pgmax
        rgumax, rgdmax = pgmax-pgmin, pgmax-pgmin
        A_om = geninfo['vom_weighted_contribution'] # Units are $/MWh
        A_hr = geninfo['hr_weighted_contribution'] # Units are mmBTU/MWh
        # TODO: get cost by month (may need to embed in scheduler after all...)
        A_fuel = geninfo['V1_wc'] # Units are $/mmBTU
        C_en = A_fuel*A_hr
        if 'gas' in gentype:
            if gentype == 'ct_gas':
                rquery = "`Generic Ramp Name`=='CT LMS RR'"
                upquery = "`Generic Min UpDn Name`=='GT LMS UpDn'"
                scquery = "`Generic Startup Name`=='CT LMS SC'"
            elif gentype == 'cc_gas':
                rquery = "`Generic Ramp Name`=='CC F RR'"
                upquery = "`Generic Min UpDn Name`=='CC F UpDn'"
                scquery = "`Generic Startup Name`=='CC F SC'"
            init_en, init_status = pgmax*0.8, 1
            init_downtime, init_uptime, outage, cost_sd = 0, 0, 0, 0 # TODO: check cost_sd
            ramp_up = self.ramp.query(rquery)["RampUp Rate(MWs/minute)"].values[0]*sf
            ramp_dn = self.ramp.query(rquery)["RampDown Rate(MWs/minute)"].values[0]*sf
            min_uptime = self.updn.query(upquery)["Minimum Up Time (hr)"].values[0]
            min_downtime = self.updn.query(upquery)["Minimum Down Time (hr)"].values[0]
            # A_sc = self.start.query(scquery)["Generic Start Fuel (MMBtu/MW)"].values[0]
            A_fc = self.start.query(scquery)["Generic Start Cost ($/MW)"].values[0]
            # A_st = self.start.query(scquery)["Startup Time"].values[0]
        elif gentype == 'coal':
            init_en, init_status = pgmax*0.8, 1
            init_downtime, init_uptime, outage, cost_sd = 0, 0, 0, 0
            ramp_up = 0.01*pgmax
            ramp_dn = 0.01*pgmax
            min_uptime = 8
            min_downtime = 10
            A_fc = 10
        elif gentype == 'nuclear':
            init_en, init_status = 1.0*pgmax, 1
            init_downtime, init_uptime, outage, cost_sd = 0, 0, 0, 0
            ramp_up, ramp_dn = 0.03*pgmax, 0.03*pgmax
            min_uptime = 24
            min_downtime = 24
            A_fc = 5
        else: # Thermal other - no idea on these figures. Perhaps should exclude...
            raise ValueError(f"No characteristics known for generator type {gentype}.")
            # init_en, init_status = 0.8*pgmax, 1
            # init_downtime, init_uptime, outage, cost_sd = 0, 0, 0, 0
            # ramp_up, ramp_dn = 0.03*pgmax, 0.03*pgmax
            # min_uptime = 2
            # min_downtime = 6
            # A_fc = 10
        cost_su = 1.0*A_fc*pgmin
        # TODO: Check; should cost_op use a dynamic MW value or is pgmin okay?
        cost_op = 1.0*A_om*pgmin/60. #Cost_op in $/min
        gen_block = self.topology['gen_block']
        mq, mc = gen_block['mq'], gen_block['mc']
        if gentype == 'nuclear':
            mq = [1.0]
            mc = [0.0]
        if sum(mq) != 1:
            for i, q in enumerate(mq):
                mq[i] = q/sum(mq)
        block_g_mq = ''
        for i, q in enumerate(mq):
            if i == 0:
                power = pgmin+q*rgumax
                add_str = f'{power:.3f}'
            else:
                power = q*rgumax
                add_str = f',{power:.3f}'
            block_g_mq += add_str
        block_g_mc = ''
        for i, c in enumerate(mc):
            if i != 0:
                block_g_mc += ','
            block_g_mc += f'{C_en*c:.3f}'
        self.gen_list += [[rid, cost_rgu, cost_rgd, cost_spr, cost_nsp, init_en, init_status, ramp_dn,
                          ramp_up, cost_su, cost_op, cost_sd, block_g_mq, block_g_mc, pgmin, pgmax,
                          rgumax, rgdmax, min_uptime, min_downtime, outage, init_downtime,
                          init_uptime]]

    def update_ren_list(self, rid, geninfo, gentype):
        '''Updates the renewable list based on available info'''
        cost_rgu, cost_rgd = 10, 10
        cost_spr, cost_nsp = 0, 0
        if gentype == 'hydro':
            sf = self.scale_fact['other_ren']
        else:
            sf = self.scale_fact[gentype]
        pvmax = geninfo['PSSEMaxCap']*sf
        pvmin = 0
        init_en = 0.5*pvmax
        init_status = 1
        ramp_up, ramp_dn = 9999, 9999
        block_r_mq = init_en
        block_r_mc = 0
        self.ren_list += [[rid, cost_rgu, cost_rgd, cost_spr, cost_nsp, init_en, init_status,
                           ramp_up, ramp_dn, pvmin, pvmax, block_r_mq, block_r_mc]]

    def update_str_list(self, rid, tot_num):
        '''Updates the storage list based on available info'''
        # pow_max = geninfo['PSSEMaxCap']
        capacity = self.topology['capacity']
        scaling = self.topology['scaling']
        resource_mix = self.topology['resource_mix']
        pow_max = capacity*resource_mix['battery']/scaling/tot_num
        # Right now assigning a maximum to each battery, then making a stack of these
        # otherwise in our thermal model the batteries heat up too much
        cell_count = 196 # From Rosewater Code Ocean
        cell_count = int(pow_max*1000/500) # Rescales powe_max in MW to 500kW max from Rosewater
        # pow_stack = pow_max/num_stack
        dcmax, chmax = pow_max, pow_max #pow_stack, pow_stack
        soc_pcntmax, soc_pcntmin = 0.95, 0.20
        soc_capacity = 640 #4*pow_max/(soc_pcntmax-soc_pcntmin) # Four hour battery capacity at max discharge (using only 75% of total soc range)
        socmax = soc_pcntmax*soc_capacity
        socmin = soc_pcntmin*soc_capacity
        soc_begin = 0.75*soc_capacity
        soc_end = socmin
        # Block offers
        block_ch_mc = '100' # '21,18'
        block_dc_mc = '0' # '26,34'
        block_soc_mc = '0' # '36,20'
        block_ch_mq = f'{chmax}' # f'{0.6*chmax},{0.4*chmax}'
        block_dc_mq = f'{dcmax}' # f'{0.6*dcmax},{0.4*dcmax}'
        block_soc_mq = f'{socmax}' # f'{0.6*(socmax-socmin)},{0.4*(socmax-socmin)}'
        # Assorted market parameters
        cost_rgu, cost_rgd, cost_spr, cost_nsp = 3, 3, 0, 0
        init_en, init_status = 0, 1
        ramp_up, ramp_dn = 9999, 9999
        eff_ch, eff_dc = 1-2*(1-0.946), 1.0
        # Battery Dispatch Parameters (from Rosewater et al 2019 with minor modifications)
        imax, imin, vmax, vmin = 1000, -1000, 820, 680
        voc_nominal = 750
        ch_cap, eff_coul = soc_capacity*1000000/cell_count/voc_nominal, 0.946
        # eff_inv0, eff_inv1, eff_inv2 = -6.1631, 0.99531, -2.05e-4
        # Using inverter coefficients adjusted to force intercept at (0,0)
        eff_inv0, eff_inv1, eff_inv2 = 0.0, 0.99531, -2.7348e-4
        voc0, voc1, voc2, voc3 = 669.282, 201.004, -368.742, 320.377
        # Adjusted resis and therm_cap from paper to Rosewater et al. 2019 Code Ocean page
        # Further increased CT by a factor of 4 to help keep temperature for a 4 hour discharge
        resis, therm_cap, Utherm = 0.000365333, 36000, 0.2*24
        temp_max, temp_min, temp_ref = 60, -20, 20
        # Battery Degradation Parameters (from Rosewater et al 2019 with minor modifications)
        deg_DoD0, deg_DoD1, deg_DoD2, deg_DoD3, deg_DoD4 = 6.16e-2, 5.37e-1, 3.3209, -6.8292, 5.7905
        # Adjusted deg_time constant to about a 15 year life (was ~75 year)
        # note deg_time is in units of 1/hrs
        deg_soc, deg_therm, deg_time = 1.04, 0.0693, 5.708e-6 # 1.49e-6
        # Adjusted cost based on NREL 2023 mid-cost prediction for 2030 Li ion. Say we recoup 10%
        # of the cost upon sale of old battery (that is an arbitrary choice)
        cost_per_kwh = 325
        cost_EoL = -cost_per_kwh*1000*soc_capacity*0.90
        cycle_life, socref = 2*365*15, 0.5*soc_capacity
        self.str_list += [[rid, cost_rgu, cost_rgd, cost_spr, cost_nsp, init_en, init_status,
                           ramp_dn, ramp_up, block_ch_mc, block_dc_mc, block_soc_mc, chmax, dcmax,
                           block_ch_mq, block_dc_mq, block_soc_mq, soc_end, soc_begin, socmax,
                           socmin, eff_ch, eff_dc, imax, imin, vmax, vmin, ch_cap, eff_coul,
                           eff_inv0, eff_inv1, eff_inv2, voc0, voc1, voc2, voc3, resis, therm_cap,
                           temp_max, temp_min, temp_ref, Utherm, deg_DoD0, deg_DoD1, deg_DoD2,
                           deg_DoD3, deg_DoD4, deg_soc, deg_therm, deg_time, cycle_life, cost_EoL,
                           socref, soc_capacity, cell_count]]

rmaker = ResourceMaker()
rmaker.write_excel('market_clearing/offer_data/resource_info.xlsx')

