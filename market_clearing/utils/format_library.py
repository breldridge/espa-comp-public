import json
import gams
import os
from .data_types import parameter_domains
import numpy as np
import pandas as pd
import datetime

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class OfferType:
    """ Class for formatting and saving market offers """
    def __init__(self, resource_name, participant_name, bus_location, resource_type, resource_df,
                 run_dir='.'):
        self._attributes = {
            "resource_name": resource_name,
            "participant_name": participant_name,
            "bus_location": bus_location,
            "resource_type": resource_type
        }
        self.run_dir = run_dir
        self.system_dir = os.path.join(run_dir,'market_clearing/system_data/')
        self.offer_dir = os.path.join(run_dir,'market_clearing/offer_data/')

        self._offer = {}
        self._default_offer = {}
        self.resource_df = resource_df
        # Set the resources dataframe as the default offer
        if type(resource_df) == dict:
            self._default_offer_dict = resource_df
        else:
            rdf_rid = resource_df.loc[resource_df["rid"]==resource_name]
            keys = rdf_rid.columns.values[1:]
            values = rdf_rid.values.flatten()[1:]
            self._default_offer_dict = dict(zip(keys,values))

    @property
    def attributes(self):
        return self._attributes

    @property
    def offer(self):
        return self._offer

    def update_offer(self, uid, prev_en=None, method='fill_from_default'):
        if method == 'fill_from_default':
            if self._default_offer == {}:
                self._set_default(uid, prev_en)
            self._fill_from_default()
        elif method == 'fill_from_previous':
            self._fill_from_previous(uid)

    def to_json(self, uid):
        data = {
            "attributes": self._attributes,
            "offer": self._offer
        }
        pid = self.attributes['participant_name']
        rtype = self.attributes['resource_type']
        rid = self.attributes['resource_name']
        if pid == 'system':
            file_dir = f'market_clearing/offer_data/fixed_system_resources/{rtype}_{rid}/'
        else:
            file_dir = f'market_clearing/offer_data/participant_{pid}/{rtype}_{rid}/'
        file_dir = os.path.join(self.run_dir, file_dir)
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, f'{uid}.json')
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4, cls=NpEncoder)

    def to_gdx(self, gdx_file):
        ws = gams.GamsWorkspace(debug=gams.DebugLevel.Off)
        db = ws.add_database()
        set_record = db["MySet"].add_record(*self._attributes.values())
        for key, value in self._offer.items():
            db["MyParameter"].add_record(set_record, key).value = value

        db.export(gdx_file)

    def _fill_from_previous(self, uid):
        pass

    def _fill_from_default(self):
        self._offer = self._default_offer

    def fill_default(self, key, value, timeseries):
        # TODO - might change the line below to allow skipping of extra keys instead of exiting
        assert key in parameter_domains.keys(), AssertionError(f"Key {key} not found in parameter_domains keys.")
        value_array = (type(value) == np.ndarray)
        if value is None:
            return None
        elif parameter_domains[key] == ['resource']:
            return {'domain': parameter_domains[key],
                    'values': value,
                    }
        elif parameter_domains[key] == ['resource', 't']:
            if value_array or type(value) == list:
                return {'domain': parameter_domains[key],
                        'keys': [t for t in timeseries],
                        'values': [value[idx] for idx, t in enumerate(timeseries)],
                        }
            else:
                return {'domain': parameter_domains[key],
                        'keys': [t for t in timeseries],
                        'values': [value for t in timeseries],
                        }
        elif parameter_domains[key] == ['resource', 'offer_block']:
            if value_array:
                value = list(value)
            assert type(value) is list
            return {'domain': parameter_domains[key],
                    'keys': [f'{b+1}' for b in range(len(value))],
                    'values': value
                    }
        elif parameter_domains[key] == ['resource', 'offer_block', 't']:
            if value_array:
                return {'domain': parameter_domains[key],
                        'keys': [(f"{b+1}", t) for b in range(value.shape[0]) for t in timeseries],
                        'values': [value[b, idx] for b in range(value.shape[0])
                                                 for idx in range(len(timeseries))]
                        }
            else:
                assert type(value) is list
                return {'domain': parameter_domains[key],
                        'keys': [(f"{b+1}", t) for b in range(len(value)) for t in timeseries],
                        'values': [value[b] for b in range(len(value)) for t in timeseries]
                        }
        else:
            raise ValueError(f"Format of key {key}:{parameter_domains[key]} is not supported.")

    def _get_time_from_system(self, uid):
        # Import system data, most relevantly timeseries (system_data['t'])
        json_file_path = os.path.join(self.system_dir, f'{uid}.json')
        with open(json_file_path, 'r') as f:
            system_data = json.load(f)
        return system_data['t']

    def _set_default(self, uid, prev_en):
        times = self._get_time_from_system(uid)
        for key, offer in self._default_offer_dict.items():
            if key not in parameter_domains.keys():
                continue
            # Unpack comma separated block offers into a list
            if 'block' in key:
                try:
                    offer = [float(val.strip()) for val in offer.split(',')]
                except AttributeError:
                    offer = [offer]
            # Update init_en and init_status based on prev_en
            if 'init_en' in key and prev_en != None:
                offer = prev_en
            if 'init_status' in key and prev_en != None:
                offer = int(prev_en != 0)
            self._default_offer[key] = self.fill_default(key=key, value=offer, timeseries=times)

    # def _forecast_to_arrays(self, forecast, trim_neg=True):
    #     # TODO - this function is slow. Fix the forecasts so they don't have to be parsed this way
    #     header = forecast.columns
    #     time = pd.to_datetime(forecast.loc[:,header[0]].values)
    #     power = forecast.loc[:,header[1]].values
    #     return time, power

    # def _get_value_from_forecast(self, times, use_forecast='newest', ren_type='both'):
    #     if use_forecast == 'newest':
    #         rtype = self._attributes["resource_type"]
    #         forecast_path = os.path.join(self.offer_dir, 'forecast.json')
    #         with open(forecast_path, 'r') as f:
    #             forecast = json.load(f)
    #         if rtype == 'ren':
    #             if ren_type == 'wind':
    #                 power = np.array(forecast['wind'])
    #             elif ren_type == 'solar':
    #                 power = np.array(forecast['solar'])
    #             elif ren_type == 'both':
    #                 power = np.array(forecast['wind'])
    #                 power += np.array(forecast['solar'])
    #             return power
    #         elif rtype == 'dem':
    #             power = np.array(forecast['demand'])
    #             return power
    #     elif use_forecast == 'new':
    #         # Find the index at which the market type ends in the time range
    #         time_idx = [idx for idx, t in enumerate(times[0]) if t == '2'][0]
    #         time0 = datetime.datetime.strptime(times[0][time_idx:], '%Y%m%d%H%M')
    #         rtype = self._attributes["resource_type"]
    #         if rtype == 'ren':
    #             solar_file = 'solar_forecast.csv'
    #             wind_file = 'wind_forecast.csv'
    #             s_forecast = pd.read_csv(os.path.join(self.system_dir,solar_file))
    #             w_forecast = pd.read_csv(os.path.join(self.system_dir,wind_file))
    #             s_time, s_value = self._forecast_to_arrays(s_forecast)
    #             w_time, w_value = self._forecast_to_arrays(w_forecast)
    #             # TODO: Update this later - value scaling shouldn't happen inside this script
    #             s_value *= 6/np.mean(s_value)
    #             w_value *= 4/np.mean(w_value)
    #             # Find the indicies of the appropriate MW values
    #             tw_start, ts_start = None, None
    #             for idx, val in enumerate(w_time):
    #                 if val == time0:
    #                     tw_start = idx
    #                     break
    #             for idx, val in enumerate(s_time):
    #                 if val == time0:
    #                     ts_start = idx
    #                     break
    #             if (tw_start is None) or (ts_start is None):
    #                 raise ValueError(f'Start time {time0} not found in wind/solar forecast')
    #             # The below assumes that forecast time interval and timeseries time interval are the same
    #             wind_mw = w_value[tw_start:tw_start+len(times)]
    #             solar_mw = s_value[ts_start:ts_start+len(times)]
    #             if ren_type == 'both':
    #                 return wind_mw+solar_mw
    #             elif ren_type == 'solar':
    #                 return solar_mw
    #             elif ren_type == 'wind':
    #                 return wind_mw
    #         elif rtype == 'dem':
    #             demand_file = 'demand_forecast.csv'
    #             d_forecast = pd.read_csv(os.path.join(self.system_dir,demand_file))
    #             d_time, d_value = self._forecast_to_arrays(d_forecast)
    #             # TODO: Update this later - value scaling shouldn't happen inside this script
    #             d_value *= 112/np.mean(d_value)
    #             # Find the indicies of the appropriate MW values
    #             td_start = None
    #             for idx, val in enumerate(d_time):
    #                 if pd.to_datetime(val) == time0:
    #                     td_start = idx
    #                     break
    #             if td_start is None:
    #                 raise ValueError(f'Start time {time0} not found in demand forecast')
    #             # The below assumes that forecast time interval and timeseries time interval are the same
    #             demand_mw = d_value[td_start:td_start+len(times)]
    #             return demand_mw
    #         else:
    #             raise ValueError(f'No forecast avaiable for resource type {self.rtype}')
    #     # Delete the 'old' forecast style once the new is fully integrated
    #     elif use_forecast == 'old':
    #         forecast = pd.read_excel(os.path.join(self.system_dir,'forecast.xlsx'), sheet_name=None)
    #         # Find the index at which the market type ends in the time range
    #         time_idx = [idx for idx, t in enumerate(times[0]) if t == '2'][0]
    #         time0 = datetime.datetime.strptime(times[0][time_idx:], '%Y%m%d%H%M')
    #         rtype = self._attributes["resource_type"]
    #         if rtype == 'ren':
    #             # Wind and solar for now - maybe we will merge this all differently later
    #             # Depends on how complex we want to make the model
    #             wind = forecast['Wind']
    #             solar = forecast['Solar']
    #             # The next two lines are only needed if the forecast isn't exactly 5min data
    #             wind['Time'] = wind['Time'].dt.round("min")
    #             solar['Time'] = solar['Time'].dt.round("min")
    #             # Find the indicies of the appropriate MW values
    #             tw_start, ts_start = None, None
    #             for idx, val in enumerate(wind['Time'].values):
    #                 if pd.to_datetime(val) == time0:
    #                     tw_start = idx
    #                     break
    #             for idx, val in enumerate(solar['Time'].values):
    #                 if pd.to_datetime(val) == time0:
    #                     ts_start = idx
    #                     break
    #             if (tw_start is None) or (ts_start is None):
    #                 raise ValueError(f'Start time {time0} not found in wind/solar forecast')
    #             # The below assumes that forecast time interval and timeseries time interval are the same
    #             wind_mw = wind['MW'].values[tw_start:tw_start+len(times)]
    #             solar_mw = solar['MW'].values[ts_start:ts_start+len(times)]
    #             return wind_mw+solar_mw
    #         elif rtype == 'dem':
    #             demand = forecast['Demand']
    #             # The next line is only needed if the forecast isn't exactly 5min data
    #             demand['Time'] = demand['Time'].dt.round("min")
    #             # Find the indicies of the appropriate MW values
    #             td_start = None
    #             for idx, val in enumerate(demand['Time'].values):
    #                 if pd.to_datetime(val) == time0:
    #                     td_start = idx
    #                     break
    #             if td_start is None:
    #                 raise ValueError(f'Start time {time0} not found in demand forecast')
    #             # The below assumes that forecast time interval and timeseries time interval are the same
    #             demand_mw = demand['MW'].values[td_start:td_start+len(times)]
    #             return demand_mw
    #         else:
    #             raise ValueError(f'No forecast avaiable for resource type {self.rtype}')

    # def _mw_to_block(self, p_mw):
    #     ''' Takes a numpy array of power(MW) and converts to a 2D offer block '''
    #     # For renewables we'll always bid in a single block of $0/MWh at the forecasted MW quantity
    #     rtype = self._attributes["resource_type"]
    #     if rtype == 'ren':
    #         block = np.reshape(p_mw, (1,len(p_mw)))
    #     # For demand we'll be two blocks, a base load with high marginal value and a flex load
    #     # with moderate marginal value. The quantities will be a fixed ratio of the overall demand
    #     elif rtype == 'dem':
    #         ratio = 100./112. # Not tied to anything in particular
    #         block1 = ratio*p_mw
    #         block2 = (1-ratio)*p_mw
    #         block = np.vstack((block1, block2))
    #     return block

class GenOffer(OfferType):
    """ Class for submitting Generator offers in correct format """
    def __init__(self, resource_name, participant_name, bus_location, resource_type, resource_df,
                 run_dir='.'):
        super().__init__(resource_name, participant_name, bus_location, resource_type, resource_df,
                         run_dir=run_dir)

    def update_gen_offer(self, uid, prev_en, outages=None):
        ''' The generator offer updates the unit outage offer variable '''
        genid = self._attributes["resource_name"]
        this_gen = self.resource_df.loc[self.resource_df["rid"]==genid]
        pgmax = this_gen['pgmax'].values[0]
        # ramp_dn = this_gen['ramp_dn'].values[0]
        self._set_default(uid, prev_en)
        times = self._get_time_from_system(uid)
        if outages != None:
            for key, offer in self._default_offer_dict.items():
                if key not in parameter_domains.keys():
                    continue
                if key.lower() == 'outage':
                    is_outage = outages[genid]['in_outage']
                    # If outage, set outage flag and put a decreasing bound on pgmax
                    if is_outage == 1:
                        offer = []
                        pg_offer = []
                        tend = outages[genid]['end_time']
                        if type(tend) == str:
                            tend = datetime.datetime.strptime(tend, '%Y%m%d%H%M')
                        for t in times:
                            t = datetime.datetime.strptime(t, '%Y%m%d%H%M')
                            if t > tend:
                                offer += [0]
                                pg_offer += [pgmax]
                            else:
                                offer += [1]
                                # Ramp down logic didn't seem to work with current GAMS formulation
                                # dt = t - datetime.datetime.strptime(times[0], '%Y%m%d%H%M')
                                # dt_min = dt.total_seconds()/60.0 + 5.0
                                # pgmax_t = max(prev_en-dt_min*ramp_dn, 0)
                                pg_offer += [0]
                        self._default_offer[key] = self.fill_default(key=key, value=offer, timeseries=times)
                        self._default_offer['pgmax'] = self.fill_default(key='pgmax', value=pg_offer, timeseries=times)
                    else:
                        offer = [0 for t in times]
                        self._default_offer[key] = self.fill_default(key=key, value=offer, timeseries=times)
        self._offer = self._default_offer

class StrOffer(OfferType):
    """ Class for submitting Storage offers in correct format """
    def __init__(self, resource_name, participant_name, bus_location, resource_type, resource_df,
                 run_dir='.'):
        super().__init__(resource_name, participant_name, bus_location, resource_type, resource_df,
                     run_dir=run_dir)

    def update_str_offer(self, uid, soc=None, prev_en=None):
        ''' Makes sure the next storage offer updates both energy and state of charge '''
        times = self._get_time_from_system(uid)
        # We will loop through keys, with special handling for the block offers
        for key, offer in self._default_offer_dict.items():
            if key not in parameter_domains.keys():
                continue
            # Unpack comma separated block offers into a list
            if 'block' in key:
                try:
                    offer = [float(val.strip()) for val in offer.split(',')]
                except AttributeError:
                    offer = [offer]
            if key == 'soc_begin' and soc != None:
                offer = 1.0*soc
            # Update init_en and init_status based on prev_en
            if 'init_en' in key and prev_en != None:
                offer = prev_en
            if 'init_status' in key and prev_en != None:
                offer = int(prev_en != 0)
            self._default_offer[key] = self.fill_default(key=key, value=offer, timeseries=times)
        self._offer = self._default_offer

class RenOffer(OfferType):
    """ Class for submitting Renewable offers in correct format """
    def __init__(self, resource_name, participant_name, bus_location, resource_type, resource_df,
                 run_dir='.'):
        super().__init__(resource_name, participant_name, bus_location, resource_type, resource_df,
                     run_dir=run_dir)

    def update_ren_offer(self, uid, forecast, prev_en=None):
        ''' Makes a demand offer from forecasted generation with 0 mc '''
        times = self._get_time_from_system(uid)
        # We will loop through keys, with special handling for the block offers
        for key, offer in self._default_offer_dict.items():
            if key not in parameter_domains.keys():
                continue
            if key == 'pvmax':
                offer = forecast*1.0
            if key == 'block_r_mq':
                offer = np.zeros((1, len(forecast)))
                offer[0,:] = forecast*1.0
            if key == 'block_r_mc':
                offer = np.zeros((1,len(forecast)))
            # Update init_en and init_status based on prev_en
            if 'init_en' in key and prev_en != None:
                offer = prev_en
            if 'init_status' in key and prev_en != None:
                offer = int(prev_en != 0)
            self._default_offer[key] = self.fill_default(key=key, value=offer, timeseries=times)
        self._offer = self._default_offer

class DemOffer(OfferType):
    """ Class for submitting Flexible Demand offers in correct format """
    def __init__(self, resource_name, participant_name, bus_location, resource_type, resource_df,
                 run_dir='.'):
        super().__init__(resource_name, participant_name, bus_location, resource_type, resource_df,
                     run_dir=run_dir)

    def update_dem_offer(self, uid, forecast, mv, mq_pcnt, prev_en=None): # source='forecast'):
        ''' Makes a demand offer from forecasted demand and request marginal value and
            marginal cost percentage distribution blocks
            eg. forecast = [2451.4, 2678.9, ..., 3821.6] - power in MW at system times
                mv = [2000, 200, 20, 2]
                mq = [0.85, 0.1, 0.03, 0.02]
        '''
        times = self._get_time_from_system(uid)
        # We will loop through keys, with special handling for the block offers
        for key, offer in self._default_offer_dict.items():
            if key not in parameter_domains.keys():
                continue
            if key == 'pdmax':
                offer = forecast*1.0
            if key == 'block_d_mq':
                offer = np.zeros((len(mq_pcnt), len(forecast)))
                for i, q in enumerate(mq_pcnt):
                    offer[i,:] = forecast*q
            if key == 'block_d_mv':
                offer = np.zeros((len(mv), len(forecast)))
                for i, v in enumerate(mv):
                    offer[i,:] = v*np.ones(len(forecast))
            # Update init_en and init_status based on prev_en
            if 'init_en' in key and prev_en != None:
                offer = prev_en
            if 'init_status' in key and prev_en != None:
                offer = int(prev_en != 0)
            self._default_offer[key] = self.fill_default(key=key, value=offer, timeseries=times)
        self._offer = self._default_offer

class VirOffer(OfferType):
    """ Class for submitting Virtual offers in correct format """
    def __init__(self, resource_name, participant_name, bus_location, resource_type, resource_df,
                 run_dir='.'):
        super().__init__(resource_name, participant_name, bus_location, resource_type, resource_df,
                     run_dir=run_dir)