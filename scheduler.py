import os
from os.path import join
import datetime
import traceback
import json
import logging
import glob
import subprocess
import argparse
import pandas as pd
import numpy as np
# import csv
# import socket
import shutil
import time as tm
from strenum import StrEnum
from enum import Enum, auto
from market_clearing.utils import data_utils as du
from market_clearing.utils import webserver_utils as wu
from physical_dispatch.storage.battery_degradation import Degradation

np.random.seed(1129)

mktSpec = {}
mktSpec['TEST'] = {
    "TEST": {
        'starting_period': [("CT", 0)],
        'market_clearing_period': ("CT", 0),
        'interval_durations': [(3, 5)],
        'interval_types': [(1, "PHYS"), (1, "FWD"), (1, "ADVS")]
    }
}
mktSpec['TS'] = {
    "TSDAM": {
        'starting_period': [("PD", 0)],
        'market_clearing_period': ("CD", 540),
        'interval_durations': [(36, 60)],
        'interval_types': [(24, "FWD"), (12, "ADVS")]
    },
    "TSRTM": {
        'starting_period': [("PH", 0), ("CH", 5), ("CH", 10), ("CH", 15), ("CH", 20), ("CH", 25), ("CH", 30),
                            ("CH", 35), ("CH", 40), ("CH", 45), ("CH", 50), ("CH", 55)],
        'market_clearing_period': ("SP", -5),
        'interval_durations': [(36, 5)],
        'interval_types': [(1, "PHYS"), (35, "ADVS")]
    }
    # "TSRTM": {
    #     'starting_period': [("PH", 0), ("CH", 30)],
    #     'market_clearing_period': ("SP", -5),
    #     'interval_durations': [(4, 60)],
    #     'interval_types': [(1, "PHYS"), (3, "ADVS")]
    # }
}
mktSpec['MS'] = {
    "MSDAM": {
        'starting_period': [("PD", 0)],
        'market_clearing_period': ("CD", 540),
        'interval_durations': [(36, 60)],
        'interval_types': [(24, "FWD"), (12, "ADVS")]
    },
    "MSRTM": {
        'starting_period': [("PH", 0), ("CH", 5), ("CH", 10), ("CH", 15), ("CH", 20), ("CH", 25), ("CH", 30),
                            ("CH", 35), ("CH", 40), ("CH", 45), ("CH", 50), ("CH", 55)],
        'market_clearing_period': ("SP", -5),
        'interval_durations': [(36, 5)],
        'interval_types': [(1, "PHYS"), (23, "FWD"), (12, "ADVS")]
    }
}
mktSpec['RHF'] = {
    "RHF36": {
        'starting_period': [("PH", 0)],
        'market_clearing_period': ("SP", -5),
        'interval_durations': [(24, 5), (40, 15), (24, 60)],
        'interval_types': [(1, "PHYS"), (87, "FWD")]
    },
    "RHF12a": {
        'starting_period': [("CH", 15)],
        'market_clearing_period': ("SP", -5),
        'interval_durations': [(24, 5), (39, 15)],
        'interval_types': [(1, "PHYS"), (62, "ADVS")]
    },
    "RHF12b": {
        'starting_period': [("CH", 30)],
        'market_clearing_period': ("SP", -5),
        'interval_durations': [(24, 5), (38, 15)],
        'interval_types': [(1, "PHYS"), (61, "FWD")]
    },
    "RHF12c": {
        'starting_period': [("CH", 45)],
        'market_clearing_period': ("SP", -5),
        'interval_durations': [(24, 5), (37, 15)],
        'interval_types': [(1, "PHYS"), (60, "FWD")]
    },
    "RHF2a": {
        'starting_period': [("CH", 5), ("CH", 20), ("CH", 35), ("CH", 50)],
        'market_clearing_period': ("SP", -5),
        'interval_durations': [(23, 5)],
        'interval_types': [(1, "PHYS"), (22, "FWD")]
    },
    "RHF2b": {
        'starting_period': [("CH", 10), ("CH", 25), ("CH", 40), ("CH", 55)],
        'market_clearing_period': ("SP", -5),
        'interval_durations': [(22, 5)],
        'interval_types': [(1, "PHYS"), (21, "FWD")]
    }
}

class MarketType(StrEnum):
    TS = "TS"
    MS = "MS"
    RHF = "RHF"
    TEST = "TEST"

class IntervalType(StrEnum):
    PHYS = 'PHYS'
    FWD ='FWD'
    ADVS ='ADVS'

class Timeline(StrEnum):
    current_time = "CT"
    current_day = "CD"
    current_hour = "CH"
    planning_day = "PD"
    planning_hour = "PH"
    horizon_begin = "HB"
    horizon_end = "HE"
    market_clearing = "MC"
    start_period = "SP"

class TimeKeeper:
    '''
    TimeKeeper maintains the simulation clock and facilitates sharing the timeline with other classes.
    Keys:
        Timeline.current_time:  current time on the simulation clock
        Timeline.current_day:   updates to 12:00:00 AM of the current time
        Timeline.current_hour:  updates to HH:00:00 of the current time
        Timeline.planning_day:  updates to one day after Timeline.current_day
        Timeline.planning_hour: updates to one hour after Timeline.current_hour
        Timeline.horizon_begin: first period in the market clearing horizon
        Timeline.horizon_end:   last period in the market clearing horizon
    Methods:
        get_status(): returns current status of clock with respect to simulation horizon.
        set_current_time(timestamp): initializes the clock at the timestamp (datetime), rounded to the next 5 minute interval.
        increment_time(): increments the clock by one interval (default 5 minutes).
        copy(): returns a copy of the dictionary of timeline data.
    '''
    def __init__(self, current_time=None):
        self.data = {}
        # Initialize the keys based on the Timeline class
        for key in Timeline:
            self.data[key] = None
        self.logger = logging.getLogger("TimeKpr")
        self.logger.debug("Initialized TimeKeeper")
        if current_time is not None:
            self.set_current_time(current_time)

    def __getitem__(self, key):
        return self.data[key]
    def __setitem__(self, key, value):
        self.data[key] = value

    class status(Enum):
        #-----PRE_SIMULATION-----|-----NORMAL-----|-----COMPLETE-----
        #                        ^Start           ^End
        PRE_SIMULATION = auto()
        NORMAL = auto()
        COMPLETE = auto()
        HORIZON_NOT_SET = auto()

    def get_status(self):
        current_time = self[Timeline.current_time]
        horizon_begin = self[Timeline.horizon_begin]
        horizon_end = self[Timeline.horizon_end]
        if horizon_begin is None or horizon_end is None:
            status = self.status.HORIZON_NOT_SET
        elif current_time < horizon_begin:
            status = self.status.PRE_SIMULATION
        elif current_time < horizon_end:
            status = self.status.NORMAL
        elif current_time >= horizon_end:
            status = self.status.COMPLETE
        else:
            raise Exception("Unexpected error in timeline status.")
        self.logger.debug(f"{status}")
        return status

    def set_horizon(self, start=None, end=None):
        if start is not None:
            self.data[Timeline.horizon_begin] = self._round_to_5min_interval(start)
        if end is not None:
            self.data[Timeline.horizon_end] = self._round_to_5min_interval(end)

    def set_current_time(self, timestamp):
        self.data[Timeline.current_time] = self._round_to_5min_interval(timestamp)
        self._update_timeline()
        self.logger.debug(f"Current time set to {self.data[Timeline.current_time]}")

    def increment_time(self, minutes=5):
        self.data[Timeline.current_time] += datetime.timedelta(minutes=minutes)
        self._update_timeline()
        self.logger.debug(f"Current time incremented to {self.data[Timeline.current_time]}")

    def _update_timeline(self):
        self.data[Timeline.current_day] = self.data[Timeline.current_time].replace(hour=0, minute=0, second=0, microsecond=0)
        self.data[Timeline.current_hour] = self.data[Timeline.current_time].replace(minute=0, second=0, microsecond=0)
        self.data[Timeline.planning_day] = self.data[Timeline.current_day] + datetime.timedelta(days=1)
        self.data[Timeline.planning_hour] = self.data[Timeline.current_hour] + datetime.timedelta(hours=1)

    def _round_to_5min_interval(self, timestamp):
        minutes_to_add = 5 - timestamp.minute % 5
        rounded_time = timestamp + datetime.timedelta(minutes=minutes_to_add)
        return rounded_time.replace(second=0, microsecond=0)

    def copy(self):
        return self.data.copy()

class BalanceSheet:
    '''
    BalanceSheet keeps track of participant profits/losses
    Each resource is tracked for surplus in the following cost types:
        energy (EN)
        regulation up (RGU)
        regulation down (RGD)
        spinning reserve (SPR)
        non-spinning reserve (NSP)
        degradation (DEG)
    Participant totals are aggegrated across all resources and cost types
    '''
    def __init__(self, participant_res, select_pid='all'): #, time_start, time_end, time_inc=5):
        self.participant_res = participant_res
        # Make reverse of participicant_res where rid is the key, pid is the value
        self.res_participant = {}
        for pid, rlist in participant_res.items():
            if select_pid != 'all':
                if pid != select_pid:
                    continue
            for rid in rlist:
                self.res_participant[rid] = pid
        # Count unique storage resources and set up total profit loss by rid/pid
        self.pid_total = {}
        self.rid_total = {}
        # TODO: add tracker for virtual resources
        num_rids = 0
        for pid in participant_res.keys():
            if select_pid != 'all':
                if pid != select_pid:
                    continue
            self.pid_total[pid] = 0
            num_rids += len(participant_res[pid])
            for rid in participant_res[pid]:
                self.rid_total[rid] = 0
        # time_interval = time_end-time_start
        # num_times = (time_interval.total_seconds()/60/time_inc) + 1 # Number of five minute intervals
        # Set up a tracker for costs by cost_type (these will be cumulative for now)
        self.ledger = {}
        self.timelist = []
        self.degradation = {} # Lists degradation costs by rid as a function of time
        self.rid_cost_types = ['EN', 'RGU', 'RGD', 'SPR', 'NSP', 'DEG']
        self.pid_cost_types = self.rid_cost_types + ['VIR']
        rid_balance = {} # profit/loss by resource
        pid_balance = {} # profit/loss by participant
        for cost in self.rid_cost_types:
            rid_balance[cost] = 0
        for cost in self.pid_cost_types:
            pid_balance[cost] = 0
        self.rid_balance = {}
        for rid in self.rid_total.keys():
            self.rid_balance[rid] = rid_balance
            self.degradation[rid] = {}
        self.pid_balance = {}
        for pid in self.participant_res.keys():
            self.pid_balance[pid] = pid_balance

    def update(self, ledger, settlement, time):
        if time not in self.timelist:
            self.timelist += [time]
        self._update_settlement(settlement)
        self._update_ledger(ledger)

    def _update_settlement(self, settlement, duration=5):
       # Loop through resources, by cost type
       for rid in settlement.keys():
           if rid not in self.rid_balance.keys():
               continue
           pid = self.res_participant[rid]
           if rid[0] == 'p': # Selects for virtual offers
               self.pid_balance[pid]['VIR'] = settlement[rid]
               self.pid_total[pid] += settlement[rid]
           else:
               for cost_type in settlement[rid].keys():
                   if rid not in self.rid_balance.keys():
                       continue
                   self.rid_balance[rid][cost_type] += settlement[rid][cost_type]
                   self.rid_total[rid] += settlement[rid][cost_type]*duration/60.
                   # Also add the value to the ledger by participant
                   self.pid_balance[pid][cost_type] += settlement[rid][cost_type]
                   self.pid_total[pid] += settlement[rid][cost_type]*duration/60.

    def _update_ledger(self, ledger, duration=5):
        # Loop through resources, cost type and timestamp
        for rid in ledger.keys():
            # Skip rid if it is not associate with this participant
            if rid not in self.rid_balance.keys() and rid != 'virtual':
                continue
            # Otherwise add rid key if it is missing
            if rid not in self.ledger.keys():
                self.ledger[rid] = {}
            if rid == 'virtual':
                for vid in ledger[rid].keys():
                    # Add virtual id (pid_bus) if it is missing
                    if vid not in self.ledger[rid].keys():
                        self.ledger[rid][vid] = {}
                    for tstamp in ledger[rid][vid].keys():
                        # Add timestamp key if it is missing
                        if tstamp not in self.ledger[rid][vid].keys():
                            self.ledger[rid][vid][tstamp] = []
                        self.ledger[rid][vid][tstamp] += ledger[rid][vid][tstamp]
            else:
                for cost in self.rid_cost_types:
                    # Skip if no cost is given
                    if cost not in ledger[rid].keys():
                        continue
                    # Add cost key if it is missing
                    if cost not in self.ledger[rid].keys():
                        self.ledger[rid][cost] = {}
                    for tstamp in ledger[rid][cost].keys():
                        # Add timestamp key if it is missing
                        if tstamp not in self.ledger[rid][cost].keys():
                            self.ledger[rid][cost][tstamp] = []
                        self.ledger[rid][cost][tstamp] += ledger[rid][cost][tstamp]

    # def _append_rid_csv(self, newline, cost, newrun, savepath):
    #     ''' Appends the line to the appropriate csv file by rid, making header if initializing.'''
    #     filename = os.path.join(savepath,f'profit_rid_{cost}.csv')
    #     if not os.path.isfile(filename) or newrun:
    #         header = ['time', 'mkt_id']
    #         for rid in self.rid_balance:
    #             header += [f'{rid}_z_{cost}']
    #         with open(filename, 'w') as file:
    #             writer = csv.writer(file)
    #             writer.writerow(header)
    #     with open(filename, 'a') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(newline)

    # def _append_pid_csv(self, newline, cost, newrun, savepath):
    #     ''' Appends the line to the appropriate csv file by pid, making header if initializing.'''
    #     filename = os.path.join(savepath,f'profit_pid_{cost.upper()}.csv')
    #     if not os.path.isfile(filename) or newrun:
    #         header = ['time', 'mkt_id']
    #         for pid in self.pid_balance:
    #             header += [f'{pid}_z_{cost}']
    #         with open(filename, 'w') as file:
    #             writer = csv.writer(file)
    #             writer.writerow(header)
    #     with open(filename, 'a') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(newline)

    def to_csv(self, savedir='saved', run_dir='.', save_ledger=False):
        ''' Saves output to multiple csv files (appending to the last row) '''
        savepath = join(run_dir, savedir)
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        # time, mkt_id = du.uid_to_time(uid, to_datetime=True, return_mkt_spec=True)
        # pidline = [time, mkt_id]
        #pidvir = {key:0 for key in self.pid_total.keys()}
        for pid in self.participant_res.keys():
            pid_ledger, pid_settlement = self.get_participant_ledger(pid, return_set=True)
            # Loop through the settlement, converting to arrays
            r0 = self.participant_res[pid][0]
            # timarray = [datetime.datetime.strptime(t, '%Y%m%d%H%M') for t \
            #             in pid_settlement[r0]['EN'].keys()]
            timearray = np.array(self.timelist)
            cost_times = {}
            for ctype in self.pid_cost_types:
                if ctype in pid_settlement[r0].keys():
                    tlist = [t for t in pid_settlement[r0][ctype].keys()]
                    cost_times[ctype] = np.array(sorted(tlist))
                elif ctype.upper() == 'VIR':
                    if 'virtual' in pid_settlement.keys():
                        tlist = []
                        for vid, tdict in pid_settlement['virtual'].items():
                            for tstamp in tdict.keys():
                                if tstamp not in tlist:
                                    tlist += [tstamp]
                        cost_times['VIR'] = np.array(sorted(tlist))
            extra_times = []
            # Find all the extra times past the last physical horizon:
            for ctype in self.pid_cost_types:
                if ctype not in cost_times.keys():
                    continue
                for i in range(len(cost_times[ctype])):
                    if cost_times[ctype][i] in timearray:
                        continue
                    elif cost_times[ctype][i] not in extra_times:
                            extra_times += [cost_times[ctype][i]]
            extra_times = sorted(extra_times)
            timearray = np.append(timearray, np.array(extra_times))
            # Now create a time mask for each cost product
            smask_dict = {}
            for ctype in self.pid_cost_types:
                if ctype not in cost_times.keys():
                    continue
                smask = []
                for i in range(len(cost_times[ctype])):
                    if cost_times[ctype][i] in timearray:
                        idx = np.where(timearray==cost_times[ctype][i])[0][0]
                        smask += [idx]
                smask_dict[ctype] = smask
            header = ['time', 'total_profit']
            num_keys = len(pid_settlement.keys())
            num_vir = 0
            if 'virtual' in pid_settlement.keys():
                num_keys -= 1
                num_vir = 1
            # Columns (after time) are total, R0_EN...R0_DEG, R1_EN...RN_DEG, VIR
            num_cost = len(self.rid_cost_types)
            num_cols = 1 + num_keys*num_cost + num_vir
            rid_cnt = 0
            profit_array = np.zeros((len(timearray),num_cols))
            for rid in pid_settlement.keys():
                if rid == 'virtual':
                    smask = smask_dict['VIR']
                    profit_dict = {}
                    for vid, tdict in pid_settlement[rid].items():
                        for tstamp, profit in tdict.items():
                            if tstamp in profit_dict.keys():
                                profit_dict[tstamp] += profit
                            else:
                                profit_dict[tstamp] = profit
                    # Sort results by time
                    these_times = np.array([t for t in profit_dict.keys()])
                    these_profits = np.array([p for p in profit_dict.values()])
                    sorted_inds = np.argsort(these_times)
                    profit_array[smask,-1] = these_profits[sorted_inds]
                else:
                    for cost in pid_settlement[rid].keys():
                        smask = smask_dict[cost]
                        colcnt = self.rid_cost_types.index(cost) + 1 + num_cost*rid_cnt
                        # Sort results by time
                        these_times = np.array([t for t in pid_settlement[rid][cost].keys()])
                        these_profits = np.array([p for p in pid_settlement[rid][cost].values()])
                        sorted_inds = np.argsort(these_times)
                        profit_array[smask,colcnt] = these_profits[sorted_inds]
                    for cost in self.rid_cost_types:
                        header += [f'profit_{rid}_{cost}']
                    rid_cnt += 1
            # Include degradation cost:
            rid_cnt = 0
            for rid in pid_settlement.keys():
                if rid == 'virtual':
                    continue
                for time, deg_cost in self.degradation[rid].items():
                    tidx = np.where(timearray==time)[0][0]
                    colcnt = self.rid_cost_types.index('DEG') + 1 + num_cost*rid_cnt
                    profit_array[tidx,colcnt] = deg_cost
                rid_cnt += 1
            if num_vir == 1:
                header += ['profit_VIR']
            # Fill out total_profit column
            profit_array[:,0] = np.sum(profit_array[:,1:], axis=1)
            # Stack on time array
            dtarray = [] # np.zeros(len(timearray), dtype=np.datetime64)
            for i in range(len(timearray)):
                dtarray += [datetime.datetime.strptime(timearray[i], '%Y%m%d%H%M')]
            dtarray = np.array(dtarray)
            sargs = np.argsort(dtarray)
            # dtarray = pd.to_datetime(timearray)
            profit_array = np.hstack((dtarray.reshape(len(dtarray),1),profit_array))
            profit_array = profit_array[sargs,:]
            # Convert to pandas dataframe and save to csv
            pid_df = pd.DataFrame(profit_array, columns=header)
            pid_df.to_csv(join(savepath,f'profit_{pid}.csv'), index=False)
            # Option to save the ledger (as a json file)
            if save_ledger:
                for pid in self.participant_res.keys():
                    pid_ledger = self.get_participant_ledger(pid, return_set=False)
                    with open(join(savepath,f'ledger_{pid}.json'), 'w') as f:
                        json.dump(pid_ledger, f, indent=4)

    def update_deg_cost(self, deg_cost, time):
        # Loop through resources and add degradation cost
        for rid in deg_cost.keys():
            if rid in self.rid_balance.keys():
                self.rid_balance[rid]['DEG'] -= deg_cost[rid]
                self.rid_total[rid] -= deg_cost[rid]
                pid = self.res_participant[rid]
                self.pid_balance[pid]['DEG'] -= deg_cost[rid]
                self.pid_total[pid] -= deg_cost[rid]
                # Also update the degradation object
                self.degradation[rid][time] = -deg_cost[rid]

    def fill_from_file(self, fdir = '.', filename='latest_settlement.json'):
        ''' Loads settlement from a (json) file when resuming a market run'''
        with open(join(fdir,filename), 'r') as f:
            in_dict = json.load(f)
        self.rid_balance = in_dict['rid_balance']
        self.rid_total = in_dict['rid_total']
        self.pid_balance = in_dict['pid_balance']
        self.pid_total = in_dict['pid_total']
        self.ledger = in_dict['ledger']
        self.timelist = in_dict['timelist']
        self.degradation = in_dict['degradation']
        # self.participant_res = in_dict['part_res']
        # self.res_participant = {}
        # for pid, rlist in participant_res.items():
        #     for rid in rlist:
        #         self.res_participant[rid] = pid

    def save_to_file(self, fdir = '.', filename='latest_settlement.json'):
        ''' Saves the balance information to a json file '''
        out_dict = {}
        out_dict['rid_balance'] = self.rid_balance
        out_dict['rid_total'] = self.rid_total
        out_dict['pid_balance'] = self.pid_balance
        out_dict['pid_total'] = self.pid_total
        out_dict['ledger'] = self.ledger
        out_dict['timelist'] = self.timelist
        out_dict['degradation'] = self.degradation
        # out_dict['part_res'] = self.participant_res
        with open(join(fdir,filename), "w") as f:
            json.dump(out_dict, f, indent=4)

    def get_degradation(self, pid='all'):
        if pid == 'all':
            return self.degradation
        else:
            deg_out = {}
            for rid in self.degradation.keys():
                if rid in self.participant_res[pid]:
                    deg_out[rid] = self.degradation[rid]
            return deg_out

    def get_participant_settlement(self, pid, total=True):
        '''can get the settlement for a particular pid or for all pids'''
        if pid == 'all' and total == True:
            return self.pid_total
        elif pid == 'all' and total == False:
            return self.pid_balance
        elif total == True:
            return self.pid_total[pid]
        else:
            return self.pid_balance[pid]

    def get_resource_settlement(self, rid, total=True):
        '''can get the settlement for a particular rid or for all rids'''
        if rid == 'all' and total == True:
            return self.rid_total
        elif rid == 'all' and total == False:
            return self.rid_balance
        elif total == True:
            return self.rid_total[rid]
        else:
            return self.rid_balance[rid]

    def get_participant_ledger(self, pid, return_set=True):
        ''' Retrieves ledger information associated with a given participant pid'''
        part_ledger = {}
        settlement = {}
        for rid in self.ledger.keys():
            virtual = (rid == 'virtual')
            if rid in self.participant_res[pid] and not virtual:
                if rid not in part_ledger.keys():
                    part_ledger[rid] = {}
                    settlement[rid] = {}
                for cost in self.ledger[rid].keys():
                    if cost not in part_ledger[rid].keys():
                        part_ledger[rid][cost] = {}
                        settlement[rid][cost] = {}
                    for tstamp in self.ledger[rid][cost].keys():
                        if tstamp not in part_ledger[rid][cost].keys():
                            part_ledger[rid][cost][tstamp] = []
                            settlement[rid][cost][tstamp] = 0
                        ledger_trunc = du.truncate_float(self.ledger[rid][cost][tstamp],2)
                        part_ledger[rid][cost][tstamp] += ledger_trunc
                        for qp in self.ledger[rid][cost][tstamp]:
                            settlement_trunc = du.truncate_float(qp[0]*qp[1],2)
                            settlement[rid][cost][tstamp] += settlement_trunc
            elif virtual:
                for vid in self.ledger[rid].keys():
                    if pid in vid:
                        if rid not in part_ledger.keys():
                            part_ledger[rid] = {}
                            settlement[rid] = {}
                        if vid not in part_ledger[rid].keys():
                            part_ledger[rid][vid] = {}
                            settlement[rid][vid] = {}
                        for tstamp in self.ledger[rid][vid].keys():
                            if tstamp not in part_ledger[rid][vid].keys():
                                part_ledger[rid][vid][tstamp] = []
                                settlement[rid][vid][tstamp] = 0
                            ledger_trunc = du.truncate_float(self.ledger[rid][vid][tstamp],2)
                            part_ledger[rid][vid][tstamp] += ledger_trunc
                            for qp in self.ledger[rid][vid][tstamp]:
                                settlement_trunc = du.truncate_float(qp[0]*qp[1],2)
                                settlement[rid][vid][tstamp] += settlement_trunc
        if return_set:
            return part_ledger, settlement
        else:
            return part_ledger

class MarketConfiguration:
    '''
    MarketConfiguration holds data that initializes a market model when prompted
    Methods:
        next_market_clearing(TimeKeeperCopy): returns the next timestamp when the market model should be created.
        to_json(): saves the market data to a JSON file.
    '''
    def __init__(self, name, config, network_file='./market_clearing/system_data/Transmission.xlsx'):
        self.name = name
        self.__dict__.update(config)
        self._network_file = network_file
        self._first_t = None
        self._strtimefmt = '%Y%m%d%H%M'
        self.logger = logging.getLogger("MktConfig")
        self.logger.debug("Initialized MarketConfiguration.")

    def next_market_clearing(self, timekeeper):
        # sets the start_period and then the market_clearing_period.
        current_time = timekeeper[Timeline.current_time]
        clear_key, clear_delta = self.market_clearing_period
        mc = None
        sp = None
        reject_time = []
        for start_key, start_delta in self.starting_period:
            start_period = timekeeper[start_key] + datetime.timedelta(minutes=start_delta)
            if start_period < timekeeper[Timeline.horizon_begin]:
                self.logger.debug(f"Market start period {start_period} is before simulation start time {timekeeper[Timeline.horizon_begin]}. Skipped.")
                continue
            if start_period > timekeeper[Timeline.horizon_end]:
                self.logger.debug(f"Market start period {start_period} is after simulation end time {timekeeper[Timeline.horizon_end]}. Skipped.")
                continue
            else:
                pass
                # horizon_flag = False # Remove flag if there is at least one instance within the TimeKeeper's horizon
            timekeeper[Timeline.start_period] = start_period
            new_mc = timekeeper[clear_key] + datetime.timedelta(minutes=clear_delta)
            if current_time <= new_mc:
                if mc is None or new_mc < mc:
                    mc = new_mc
                    sp = start_period
            else:
                reject_time.append(new_mc)
        # if horizon_flag:
        #     raise Exception(f"Data error: all start periods for time {current_time} were outside horizon "
        #                     f"{timekeeper[Timeline.horizon_begin]} to {timekeeper[Timeline.horizon_end]}.")
        # assert mc is not None, f"unable to find a market clearing time before current time {current_time}. Rejected times: {reject_time}"
        self._first_t = sp
        return mc

    def to_json(self, savedir='.', t_dense=None):
        data_out = {}
        data_out = self._generate_interval_data(data_out, t_dense)
        data_out = self._generate_network_data(data_out)

        # save to json file
        uid = f"{self.name}{self._time2str(self._first_t)}"
        uid_json = f"./market_clearing/system_data/{uid}.json"
        uid_json = join(savedir, uid_json)
        with open(uid_json, "w") as file_out:
            json.dump(data_out, file_out, indent=4)
        self.logger.info(f"Saved {uid_json}")
        return uid

    def _time2str(self, timestamp):
        return timestamp.strftime('%Y%m%d%H%M')

    def _generate_interval_data(self, data_out={}, t_dense=None):
        assert self._first_t is not None, "First period of model has not been identified. Please run MarketConfiguration.next_market_clearing() first."
        # populate t and duration data
        interval = self._first_t
        interval_list = []
        duration_list = []
        for duration_tuple in self.interval_durations:
            t_count, t_delta = duration_tuple
            for idx in range(t_count):
                interval_list.append(self._time2str(interval))
                duration_list.append(t_delta)
                interval += datetime.timedelta(minutes=t_delta)
        # populate physical, forward, and advisory data
        type_list = []
        for type_tuple in self. interval_types:
            t_count, t_type = type_tuple
            for idx in range(t_count):
                type_list.append(t_type)
        assert len(interval_list) == len(type_list), AssertionError(f"Input has duration data for {len(interval_list)} "
                                                                    f"intervals but type data for {len(type_list)} intervals.")
        type_dict = dict(zip(interval_list, type_list))
        phys_list = [key for key,value in type_dict.items() if value==IntervalType.PHYS]
        fwd_list = [key for key,value in type_dict.items() if value==IntervalType.FWD]
        advs_list = [key for key,value in type_dict.items() if value==IntervalType.ADVS]
        # populate dictionary
        data_in = {}
        data_in['t'] = interval_list
        data_in['tt'] = interval_list # Copy of t to use in generator startup logic
        # "Dense" time (5 minute intervals) included for forward position calculation
        if t_dense is None:
            data_in['t_dense'] = interval_list
        else:
            # Add in any new regular times
            t_dense += [t for t in interval_list if t not in t_dense]
            data_in['t_dense'] = t_dense
        data_in['physical'] = phys_list
        data_in['forward'] = fwd_list
        data_in['advisory'] = advs_list
        data_in['duration'] = {'dim': 1,
                               'keys': interval_list,
                               'values': duration_list}
        data_out.update(data_in)
        return data_out

    def _generate_network_data(self, data_out={}):
        # TODO: find a cleaner way to implement the network topology
        if NETWORK == 'test':
            with open('market_clearing/system_data/case5.json', 'r') as file:
                data_in = json.load(file)
            data_out.update(data_in)
        elif NETWORK == 'WECC':
            network_df = pd.read_excel(self._network_file, sheet_name='Clean_BA_Format')
            data_in = du.parse_network_data(network_df)
            self._generate_line_status(data_in)
            data_out.update(data_in)
        else:
            raise ValueError(f"No network model for network {NETWORK}. Choose 'test' or 'WECC'.")
        return data_out

    def _generate_line_status(self, network_data:dict):
        for item in ['linestatus', 'monitored']:
            append_dict = {item: {}}
            status = append_dict[item]
            status['__tuple__'] = True
            status['keys'] = []
            status['values'] = []
            for line in network_data['line']:
                status['keys'].append(line)
                status['values'].append(1)
            network_data.update(append_dict)

class MarketQueue:
    '''
    MarketQueue keeps track of the next MarketConfiguration that needs to be solved.
    Methods:
        on_deck(timestamp):
            returns an iterator tuple of (key, MarketConfiguration) that are ready to be generated
            and solved.
        update(timestamp, [key]):
            recomputes the next_market_clearing time given current time at timestamp.
    '''
    def __init__(self, config_dict, timekeeper=None, log_level=logging.DEBUG):
        self.queue = {key: {'market_configuration': MarketConfiguration(name=key, config=config),
                            'next_market_clearing': None}
                      for key, config in config_dict.items()}
        if timekeeper is not None:
            self.update(timekeeper)
        self.logger = logging.getLogger("MktQueue")

    def on_deck(self, timestamp):
        for name, market_data in self.queue.items():
            nmc = market_data['next_market_clearing']
            if nmc is not None and nmc <= timestamp:
                self.logger.info(f"Queueing {name} for {nmc}.")
                yield name, market_data['market_configuration']
            else:
                self.logger.debug(f"{name} next usage is at {nmc}.")

    def update(self, timekeeper, keys=None):
        if keys is None:
            self._update_all(timekeeper)
        else:
            for key in keys:
                self._update(self.queue[key], timekeeper)

    def _update(self, mkt_data, timekeeper):
        mkt_data['next_market_clearing'] = mkt_data['market_configuration'].next_market_clearing(timekeeper)

    def _update_all(self, timekeeper):
        for name, mkt_data in self.queue.items():
            self._update(mkt_data, timekeeper)

class MarketScheduler:
    '''
    MarketScheduler simulates market activities throughout the specified horizon.
    Arguments:
        type:   this specifies the market configuration (valid types are TS, MS, RHF, and TEST)
        start:  The first interval in the simulation horizon that will have a physical market settlement
        end:    The end of the last period in the simulation horizon (note: no physical delivery in this period)
    Methods:
        simulate(): controls the main loop of the market simulation.
        collect_offers(uid): collects all resource offers in JSON format and saves to a single GDX file
        solve_market(uid): executes the GAMS market clearing engine (market_clearing_engine.py)
        physical_dispatch(uid): simulates the physical dispatch of all resources
        settle_positions(uid): calculates the financial settlements of each market participant

    '''
    def __init__(self, mkt_type:MarketType, start:datetime, end:datetime, options_dict=None,
                 last_time=None, time_step=None, run_dir='.', clean=True):
        # Operations differ a bit depending on sandbox or competition mode
        assert options_dict['mode'] in ['sandbox', 'competition']
        self.mode = options_dict['mode']
        # Set participant parameters and market type and directories
        self._parse_options(options_dict)
        self.mkt_type = mkt_type
        self.horizon_type = 'minutes'
        self.horizon = (end-start).total_seconds()/60.
        self.start_date = start.strftime('%Y%m%d%H%M')
        self.run_dir = run_dir
        self.savedir = f'saved/{mkt_type}_market'
        # Ensure all of the necessary subdirectories are made
        du.make_market_dirs(run_dir)
        self.clean = clean
        # Start the logger
        self.logger = logging.getLogger("MktSchdlr")
        self.logger.debug("Initialized MarketScheduler.")
        # Load participant and resource information
        self.resources_df = du.load_resource_info()
        if self.mode == 'competition':
            self.participant_res, self.uname_dict = du.load_participant_info(return_uname_dict=True)
            self.pid = 'all' # Overwrite options, always set to all for competition mode
        else:
            # Select the participant resource (not forced to match json file in sandbox mode)
            if os.path.isfile(join(self.run_dir,'participant_res.json')):
                with open(join(self.run_dir,'participant_res.json'), 'r') as f:
                    trackers = json.load(f)
                self.participant_res = trackers['part_res']
            else:
                self.participant_res = self._select_part_res(options_dict)
                trackers = {'part_res': self.participant_res}
                with open(join(self.run_dir, 'participant_res.json'), "w") as f:
                    json.dump(trackers, f, indent=4)
        # Initiate balance sheet
        self.balance_sheet = BalanceSheet(self.participant_res, select_pid=self.pid)
        if time_step is None:
            self.time_step = 1
        else:
            self.time_step = time_step
        # Load settings (if last_time is None - continues a previous run)
        self._load_settings(last_time, start)
        # Load the forecast files into memory
        self._load_forecasts(options_dict['host'])
        # Create timekeeper and market queue
        self.timekeeper = TimeKeeper(current_time=self.last_time)
        self.timekeeper.set_horizon(start=start, end=end)
        self.queue = MarketQueue(mktSpec[mkt_type], timekeeper=self.timekeeper.copy())

    def _load_forecasts(self, host):
        ''' Opens the forecast files and stores dataframes in a dictionary '''
        if self.mode == 'sandbox' and host == 'VM':
            self.forecast_dict = None # For backward compatibility
        else:
            fkeys = ['forecast', 'actual']
            rkeys = ['demand', 'wind', 'solar','hydro']
            tkeys = ['1hr', '5min']
            # Load all cases into dict
            self.forecast_dict = {}
            for key in fkeys:
                self.forecast_dict[key] = dict()
                for rtype in rkeys:
                    self.forecast_dict[key][rtype] = dict()
                    for ttype in tkeys:
                        forecast_df = du.open_forecast_file(None, rtype, ttype,
                                                            actual=(key=='actual'))
                        self.forecast_dict[key][rtype][ttype] = forecast_df

    def _select_part_res(self, options):
        ''' Picks a random resource from your selected bus '''
        str_bus = options['str_bus']
        # Get storage resource ids and bus from resource info dataframe
        mdf = self.resources_df['Master']
        str_rids = mdf.loc[mdf['ResourceType']=='str','rid'].values
        str_buses = mdf.loc[mdf['ResourceType']=='str','ResourceBus'].values
        # If no str_bus submitted or not storage unit at the bus, pick a random bus
        if str_bus is None or str_bus not in str_buses:
            with open("topology_conf.json", 'r') as f:
                topology = json.load(f)
            bus_options = topology["storage_loc"]["bus"]
            choice = np.random.randint(len(bus_options))
            str_bus = bus_options[choice]
        # selected_rid = self.participant_res[pid][0]
        # selected_idx = list(str_rids).index(selected_rid)
        # if str_bus != str_buses[selected_idx]:

        # Select a random unit and assign to the participant
        str_rid_inds = [i for i in range(len(str_buses)) if str_buses[i] == str_bus]
        ind_choice = np.random.randint(len(str_rid_inds))
        rid = str_rids[ind_choice]
        return {self.pid: [rid]}

    def _parse_options(self, options):
        ''' Takes various optoins and saves, using defaults for missing (None) params
            Sets different characteristics for Sandbox/single-run (one participant at a time)
            and Competition/multi-run (multiple participants simulatanously bidding into market)
        '''
        defaults = {'language':'python', 'alg_name':'market_participant.py', 'pid':'p00002',
                    'str_bus':'CISD', 'part_params':None, 'host':'local',
                    'part_alg_root':'market_clearing/offer_data'}
        for key in defaults.keys():
            if key in options.keys():
                val = options[key]
            else:
                val = defaults[key]
            setattr(self, key, val)

    def _load_settings(self, last_time, start):
        ''' Loads a previous market state or initializes tracking variables '''
        if last_time is None:
            self.last_time = start - datetime.timedelta(days=1)
            # Save the previous cleared market id for each market type (DAM, RTM, 12a, etc.)
            self.prev_mkt_uid = {'all': None}
            for mktid in mktSpec[self.mkt_type].keys():
                self.prev_mkt_uid[mktid] = None
            self.prev_disp_uid = None
            self.prev_deg_time = None
            self.dense_time = None
            self.skip_DAM = False
            self.drw_fact = np.random.rand()*2 - 1
            self.outages = self._init_outages()
            self.start_time = tm.time()
            if self.run_dir == '.' and self.clean:
                du.delete_all_offers(pid=self.pid, run_dir=self.run_dir)
        elif type(last_time) == str:
            self.last_time = datetime.datetime.strptime(last_time, '%Y%m%d%H%M')
            with open(join(self.run_dir,'mkt_trackers.json'), 'r') as f:
                trackers = json.load(f)
            self.prev_mkt_uid = trackers['prev_mkt_uid']
            self.prev_disp_uid = trackers['prev_disp_uid']
            self.dense_time = trackers['dense_time']
            self.prev_deg_time = trackers['prev_deg_time']
            self.start_time = trackers['start_time']
            self.run_time = trackers['run_time']
            self.outages = trackers['outages']
            self.participant_res = trackers['part_res']
            self.outages = du.outage_datetime_to_str(self.outages, reverse=True)
            self.balance_sheet.fill_from_file(fdir=self.run_dir)
            self.skip_DAM = False
            self.drw_fact = trackers['drw_fact']
            if self.prev_disp_uid != None:
                self.last_time -= datetime.timedelta(minutes=5)
                if 'DAM' in self.prev_mkt_uid['all']:
                    self.skip_DAM = True

    # TODO: find a way to refactor simulate_all, simulate, and prep_for_offers to reduce
    #       duplication of code
    def simulate_all(self, offers=None, skip=False):
        ''' Continuous runs the simulation until the end of the time horizon '''
        time = self.timekeeper
        out = {}
        status = {}
        while time.get_status() != time.status.COMPLETE:
            out, _ = self.prep_for_offers(increment=False)
            _, status = self.simulate()
        # Save finance info
        self.balance_sheet.to_csv(savedir=self.savedir, run_dir=self.run_dir, save_ledger=True)
        return out, status

    def prep_for_offers(self, increment=True):
        ''' Prepares for offers, outputting the market info and history dictionaries '''
        time = self.timekeeper
        status = {'done':True, 'time_step':self.time_step, 'success':True, 'disp':True,
                  'mkt_spec': None, 'prev_offer':{}}
        out = {}
        t0 = tm.time()
        if time.get_status() != time.status.COMPLETE:
            prepped_offer = False
            while not prepped_offer:
                current_time = time[Timeline.current_time]
                cleared_mkts = []
                self.queue.update(time.copy())
                cnt = 0
                for mkt_key, mkt_config in self.queue.on_deck(current_time):
                    if self.skip_DAM and cnt == 0:
                        cnt += 1
                        continue
                    # Set and save market configuration
                    t3 = tm.time()
                    uid = mkt_config.to_json(savedir=self.run_dir, t_dense=self.dense_time)
                    junk, mkt_spec = du.uid_to_time(uid, return_mkt_spec=True)
                    status['mkt_spec'] = mkt_spec
                    system_dir = join(self.run_dir,'market_clearing/system_data/')
                    du.json_to_gdx(directory=system_dir, filename=uid)
                    t4 = tm.time()
                    # Add participant info (forecast, history, status, etc.) to output dict
                    out = self._get_participant_info(uid, out, current_time.strftime('%Y%m%d%H%M'))
                    if self.mode == 'sandbox':
                        out['resources'] = out[self.pid]['resources']
                    # First time through (for each mkt type/spec] clear market to get a price
                    # forecast for participants
                    t5 = tm.time()
                    if self.prev_mkt_uid[mkt_spec] is None:
                        out = self.initial_clear(uid, out)
                    t6 = tm.time()
                    if self.run_dir == '.' or self.mode == 'competition':
                        self.run_participant_algorithms(uid, self.participant_res, out)
                    t7 = tm.time()
                    
                    self.logger.info(f"Successfully prepared offer participant info at {uid}.")
                    prepped_offer = True
                    break
                # If offer is prepped, check increment keyword. If not prepped, always increment
                if prepped_offer:
                    if increment:
                        time.increment_time()
                else:
                    time.increment_time()
                if cnt > 288:
                    break
                cnt += 1
                self.queue.update(time.copy(), keys=cleared_mkts)
                if time.get_status() == time.status.COMPLETE:
                    t3, t4, t5, t6, t7 = [tm.time()]*5
                    break
            status['done'] = False
        t1 = tm.time()
        self.logger.info(f"\nPrepping time was {t1-t0:.3f}s")
        # self.logger.info(f"Save system time was {t4-t3:.3f}s")
        # self.logger.info(f"Participant info time was {t5-t4:.3f}s")
        # self.logger.info(f"Initial clear time was {t6-t5:.3f}s")
        # self.logger.info(f"Running participant algorithm time was {t7-t6:.3f}s\n")
        return out, status

    def simulate(self, offers=None, skip=False):
        ''' Simulates the market clearing and physical dispatch, then computes settlements '''
        time = self.timekeeper
        queue = self.queue
        status = {'done':True, 'time_step':self.time_step, 'success': True, 'mkt_spec': None,
                  'disp': False, 'prev_offer':{}}
        t0 = tm.time()
        if time.get_status() != time.status.COMPLETE:
            cleared = False
            while not cleared:
                current_time = time[Timeline.current_time]
                cleared_mkts = []
                queue.update(time.copy())
                cnt = 0
                for mkt_key, mkt_config in queue.on_deck(current_time):
                    if self.skip_DAM and cnt == 0:
                        cnt += 1
                        continue
                    uid = mkt_config.to_json(savedir=self.run_dir, t_dense=self.dense_time)
                    prev_offers = self.collect_offers(uid, offers, skip=skip)
                    t1 = tm.time()
                    prev_offer_pid = {}
                    try:
                        for rid in self.participant_res[self.pid]:
                            prev_offer_pid[rid] = prev_offers[rid]
                    except KeyError:
                        pass
                    status['prev_offer'] = prev_offer_pid
                    # clear market
                    success = self.solve_market(uid)
                    t2 = tm.time()
                    status['success'] = success
                    junk, mkt_spec = du.uid_to_time(uid, return_mkt_spec=True)
                    status['mkt_spec'] = mkt_spec
                    disp = self.physical_dispatch(uid, mkt_config, self.resources_df['Storage'],
                                                  skip=skip)
                    t3 = tm.time()
                    status['disp'] = disp
                    self.settle_positions(uid, mkt_config, skip=skip)
                    cleared_mkts.append(mkt_key)
                    self.logger.info(f"Cleared {uid} market at {current_time}")
                    print(f"Cleared {uid} market at {current_time}")
                    cleared = True
                    break
                # prepare for next time interval (but don't increment if skip=True)
                if cleared:
                    if not skip:
                        time.increment_time()
                        if self.host != 'VM':
                            self.time_step += 1
                else:
                    time.increment_time()
                queue.update(time.copy(), keys=cleared_mkts)
                if cnt > 288:
                    break
                cnt += 1
            if not skip:
                self.save_state()
            if self.run_dir == '.':
                self._delete_old_records(uid, keep_count=12)
            else:
                self._delete_old_records(uid)
            status['done'] = False
        # On completion, save all finance info
        if time.get_status() == time.status.COMPLETE and self.host == 'VM':
            status['done'] = True #VM won't do a final call after the last simulation
        if status['done']:
            self.balance_sheet.to_csv(savedir=self.savedir, run_dir=self.run_dir, save_ledger=True)
        else:
            t4 = tm.time()
            # self.logger.info(f"Collect offer time was {t1-t0:.3f}s")
            # self.logger.info(f"Solve market time was {t2-t1:.3f}s")
            # self.logger.info(f"Physical dispatch time was {t3-t2:.3f}s")
            # self.logger.info(f"Settlement time was {t4-t3:.3f}s")
        # Send a copy of the latest score
        try:
            resources = du.get_res_for_participants(self.pid, uid, self.prev_disp_uid,
                                        self.resources_df, self.participant_res,
                                        self.balance_sheet.get_degradation(), run_dir=self.run_dir)
            # Get participant pricing info for their resource settlements
            for pid in self.participant_res.keys():
                ledger, settlement = self.balance_sheet.get_participant_ledger(pid, return_set=True)
                schedule, score = du.convert_ledger_to_score(uid, ledger, settlement,
                                                             self.balance_sheet.get_degradation())
                summary = {"score":score['current']}
        except UnboundLocalError:
            summary = {}
        return summary, status

    def initial_clear(self, uid, out):
        ''' Clears the market then deletes the market gdx and offer files created '''
        # Delete any existing gdx files
        # self._delete_old_records(uid, keep_count=0)
        # self._delete_old_records(uid, dispatch=True, keep_count=0)
        # Set skip = True to use system offers by default and to skip the settlement
        self.simulate(skip=True)
        this_time, mkt_spec = du.uid_to_time(uid, return_mkt_spec=True)
        prev_mkt_dict = du.create_prev_mkt_dict(self.prev_mkt_uid, mkt_spec)
        prev_cleared = du.get_prev_cleared(prev_mkt_dict, mktSpec, run_dir=self.run_dir)
        out['market']['previous'] = prev_cleared
        print(f"Ran an initial clearing for market {mkt_spec}")
        # Delete the gdx files (these aren't used for settlements, just for initial price forecasts)
        try:
            os.remove(join(self.run_dir,f'market_clearing/offer_data/{uid}.gdx'))
        except FileNotFoundError:
            pass
        try:
            os.remove(join(self.run_dir,f'market_clearing/results/results_{uid}.gdx'))
        except FileNotFoundError:
            pass
        return out

    def _get_participant_info(self, uid, out, current_time):
        ''' Retrevies dictionaries with forecast, history, market info, settlement info, etc.
            to be sent to participant algorithms.
        '''
        this_time, mkt_spec = du.uid_to_time(uid, return_mkt_spec=True)
        intervals = mktSpec[self.mkt_type][mkt_spec]['interval_durations']
        interval_types = mktSpec[self.mkt_type][mkt_spec]['interval_types']
        # Get information related to the market forecast and history
        market = du.get_mkt_for_participants(self.resources_df, self.participant_res, uid,
                                        self.prev_mkt_uid['all'], intervals, current_time,
                                        interval_types, self.forecast_dict, run_dir=self.run_dir,
                                        drw=self.drw_fact)
        out['market'] = market
        # Add previous pricing info for price forecasts
        prev_mkt_dict = du.create_prev_mkt_dict(self.prev_mkt_uid, mkt_spec)
        if prev_mkt_dict != None:
            prev_cleared = du.get_prev_cleared(prev_mkt_dict, mktSpec, run_dir=self.run_dir)
            out['market']['previous'] = prev_cleared

        # Get resource specific dispatch information
        resources = du.get_res_for_participants(self.pid, uid, self.prev_disp_uid,
                                        self.resources_df, self.participant_res,
                                        self.balance_sheet.get_degradation(), run_dir=self.run_dir)
        # Get participant pricing info for their resource settlements
        for pid in self.participant_res.keys():
            out[pid] = dict()
            ledger, settlement = self.balance_sheet.get_participant_ledger(pid, return_set=True)
            schedule, score = du.convert_ledger_to_score(uid, ledger, settlement,
                                                         self.balance_sheet.get_degradation())

            out[pid]['resources'] = resources[pid]
            out[pid]['resources']['ledger'] = ledger
            out[pid]['resources']['settlement'] = settlement
            out[pid]['resources']['schedule'] = schedule
            out[pid]['resources']['score'] = score
        return out

    def run_participant_algorithms(self, uid, participant_res, info_dicts, save_mkt_res=True):
        ''' Used for local testing of participant algorithms.
            Invokes python market_participant.py (or alternative alg. name) with inputs:
                                                    {time_step}, {market}, and {resources}
            If in competition mode, uses webserver_utils.py to request offers and wait
            for a response (offers are computed on the cluster in the competition)
        '''
        market = info_dicts['market']
        market_json = json.dumps(market)
        # Now loop through participants and launch their jobs
        # Doing in series for now - can parallelize later if desired
        for pid in participant_res.keys():
            resources = info_dicts[pid]['resources']
            resource_json = json.dumps(resources)
            prev_uid = self.prev_mkt_uid['all']
            if prev_uid is None:
                prev_uid = 'none'
            if save_mkt_res: # and 'DAM' in prev_uid:
                if self.host == 'local' and self.mode == 'sandbox':
                    if not os.path.isdir('saved'):
                        os.makedirs('saved')
                    # prev_dam = market['previous']['TSDAM']
                    # with open(f"saved/market_{uid}.json", "w") as f:
                        # json.dump(prev_dam, f, indent=4)
                    with open(f"saved/market_{uid}.json", "w") as f:
                        json.dump(market, f, indent=4)
                    with open(f"saved/resources_{uid}.json", "w") as f:
                        json.dump(resources, f, indent=4)
            if self.host == 'local':
                try:
                    part_alg_path = join(self.part_alg_root,f'participant_{pid}')
                    pdir = join(os.getcwd(), part_alg_path)
                    mkt = join(pdir,"market.json")
                    with open(mkt, "w") as f:
                        json.dump(market, f, indent=4)
                    res = join(pdir,"resources.json")
                    with open(res, "w") as f:
                        json.dump(resources, f, indent=4)
                    algorithm = subprocess.run(['python',f'{self.alg_name}',f'{self.time_step}',
                                                f'{mkt}', f'{res}'],
                                                capture_output=True, text=True, check=True,
                                                cwd=os.path.join(self.run_dir,pdir))
                    # algorithm = subprocess.run(['./market_participant',f'{self.time_step}',
                    #                             f'{market_json}', f'{resource_json}'],
                    #                             capture_output=True, text=True, check=True,
                    #                             cwd=os.path.join(self.run_dir,pdir))
                    # Readout messages from algorithm
                    self.logger.debug(algorithm.stdout)
                    self.logger.debug(algorithm.stderr)
                    print(algorithm.stdout.strip())
                    print(algorithm.stderr.strip())
                except subprocess.CalledProcessError as e:
                    print(f"Call to market_participant.py failed for participant {pid}. Error message:")
                    print(e.output)
                    print(e.stderr)
                    exit(0)
            elif self.host == 'VM':
                print("running offer request code")
                # Track whether offers are requested - check for saves in case of interruption
                if os.path.isfile(join(self.run_dir, 'offer_request.json')):
                    with open(join(self.run_dir, 'offer_request.json'),'r') as f:
                        offers_requested = json.load(f)
                    try:
                        offer_requested = offers_requested[uid]
                    except KeyError:
                        offer_requested = {} # Child dictionary with request at this uid
                        for pid in self.participant_res.keys():
                            offer_requested[pid] = False
                else:
                    offers_requested = {} # Parent dictionary with uid as keys
                    offer_requested = {} # Child dictionary with request at this uid
                    for pid in self.participant_res.keys():
                        offer_requested[pid] = False
                # Make uuid-to-pid dict inside here for retrieval
                if not hasattr(self, 'uuid_to_pid'):
                    self.uuid_to_pid = dict()
                while sum([int(v) for v in offer_requested.values()]) \
                    < len(self.participant_res.keys()):
                    print("Offer requested:", offer_requested)
                    print("Participant res:", self.participant_res)
                    tm.sleep(0.1) # Short sleep to help prevent conflicts when sending files
                    participant_details = wu.get_next_participant()
                    details = participant_details
                    print("found details:", details)
                    if participant_details == {}:
                        continue
                    # If an offer already exists for this participant, don't request another
                    # TODO: May need to have market_webserver.py sort by order, or accept uuid
                    # as an input to get_next_offer to ensure no double-offers
                    # Should only be an issue if a run gets interrupted
                    try:
                        username = details['team']
                    except KeyError:
                        username = details['deception_path'].split('/')[-2]
                    pid = self.uname_dict[username]
                    print(f"Retrieved pid {pid} for user {username}")
                    # offer_details = wu.get_next_offer()
                    # print("Got offer details:", offer_details)
                    # if offer_details is not  None:
                    #    part_uuid = details["uuid"]
                    #    offer_uuid = offer_details["uuid"]
                    #    if part_uuid == offer_uuid:
                    #        offer_requested[pid] == True
                    #        continue
                    uuid = details["uuid"]
                    division = details["division"]
                    time_step = details["time_step"]
                    offer_request_details = {
                        "uuid": uuid,
                        "division": division,
                        "time_step": time_step,
                        "market": market,
                        "resource": info_dicts[pid]['resources'],
                        "time_limit": info_dicts[pid]['resources']['time_limit']
                        }
                    print(f'\nRequesting offer for {username}/{uuid} at time_step {time_step}')
                    logger.info(f'Requested offer for {username}/{uuid} at time_step {time_step}')

                    wu.request_offer(offer_request_details)
                    # Update tracking for while loop and uuid to pid conversion (for offer
                    # retrieval)
                    if details["uuid"] not in self.uuid_to_pid.keys():
                        self.uuid_to_pid[details["uuid"]] = pid
                        print(f"setting dict uuid with ({details['uuid']}:{pid})")
                    offer_requested[pid] = True
                    print(f"Set offer request True for pid {pid}", offer_requested)
                # Save offer requests at this uid
                offers_requested.update({uid:offer_requested})
                # Add uuid to pid conversion dict
                offers_requested.update({'uuid_to_pid':self.uuid_to_pid})
                with open(join(self.run_dir,'offer_request.json'),'w') as f:
                    json.dump(offers_requested, f, indent=2)

    def _load_local_offer(self, uid):
        ''' Loads offer_{timestep}.json file from a locally run offer '''
        # Check participant directories for offers
        globpath = join(self.run_dir,'market_clearing/offer_data/participant_p*')
        part_dirs = [entry for entry in glob.glob(globpath) \
                     if os.path.isdir(entry)]
        offers = {}
        for offer_dir in part_dirs:
            pid = offer_dir.split('/')[-1].split('_')[-1]
            if pid not in self.participant_res.keys():
                continue
            files = [f for f in os.listdir(offer_dir) if f'offer_{self.time_step}.json' == f]
            if len(files) == 0:
                return 'empty'
                # raise FileNotFoundError(f"Offer file not found for {pid} at {uid}" + \
                                        # f" (time step {self.time_step}")
            # Open the JSON file and load its contents
            with open(os.path.join(offer_dir,files[0]), 'r') as file:
                offer_data = json.load(file)
            offers.update(offer_data)
            self.logger.debug(f"Loaded data from {file}")
        return offers

    def _load_webserver_offer(self, uid):
        ''' Requests completed offers using the market_webserver interface '''
        # Track whether offers have been found
        offer_found = {}
        for pid in self.participant_res.keys():
            offer_found[pid] = False
        all_offers = {}
        # Load uuid_to_pid conversion dictionary if it hasn't been made
        if not hasattr(self, 'uuid_to_pid'):
            with open(join(self.run_dir,'offer_request.json'),'r') as f:
                offers_requested = json.load(f)
                self.uuid_to_pid = offers_requested['uuid_to_pid']
        while sum([int(v) for v in offer_found.values()]) < len(self.participant_res.keys()):
            tm.sleep(0.1) # Short sleep to help prevent conflicts when sending files
            # Look for offer
            print("running offer request code in load offer")
            offer_details = wu.get_next_offer()
            if offer_details is None:
                continue
            details = offer_details
            uuid = details["uuid"]
            division = details["division"]
            time_step = details["time_step"]
            try:
                offers = json.loads(details['offer'])
            except json.decoder.JSONDecodeError:
                self.logger.debug(f'Warning: Found empty offer for {uuid} at time_step {time_step}')
                self.logger.debug("Requesting previous offer")
                prev_offers = wu.get_recent_offer(uuid, division, self.mkt_type)
                # TODO: add in error handling if there is no previous offer
                offers = json.loads(prev_offers['offer'])
                # Update the timestamps, then update the database with the recent offer
                rid = list(offers.keys())[0]
                prev_offer = du.fill_from_previous_offer(rid, uid, {rid:offers[rid]}, run_dir=self.run_dir)
                self.logger.debug("Setting offer at with:", uuid, division, time_step, prev_offer)
                adjusted_offer = json.dumps(prev_offer)
                adjusted_offer = json.loads(adjusted_offer)
                self.logger.debug("Adjusted Offer:\n", adjusted_offer)
                wu.set_offer(uuid, division, time_step, adjusted_offer)
            # Send score over time
            resources = du.get_res_for_participants(self.pid, uid, self.prev_disp_uid,
                                        self.resources_df, self.participant_res,
                                        self.balance_sheet.get_degradation(), run_dir=self.run_dir)
            ledger, settlement = self.balance_sheet.get_participant_ledger(self.uuid_to_pid[uuid],
                    return_set=True)
            schedule, score = du.convert_ledger_to_score(uid, ledger, settlement,
                                                         self.balance_sheet.get_degradation())
            summary = {"score":score['current']}
            mkt_params = {'mktSpec':self.mkt_type, 'horizon_type':self.horizon_type, 'horizon':self.horizon, 'start_date':self.start_date, 'str_bus':None}
            wu.set_offer_settlement_summary(uuid, division, self.mkt_type, self.mode, mkt_params, time_step, summary)
            # Add the offer to the output and 'found' list
            all_offers.update(offers)
            offer_found[self.uuid_to_pid[uuid]] = True
            print(f"Found an offer for participant {self.uuid_to_pid[uuid]} at time step {time_step}")
        return all_offers

    def collect_offers(self, uid, offers, skip=False):
        '''
        looks at all of the resource offer JSON files and compiles them into a single GDX named by the market UID.
        :param uid:
        :return:
        '''
        # Update system offers, send status and forecast to competitors, then collect all offers
        t0 = tm.time()
        this_time, mkt_spec = du.uid_to_time(uid, return_mkt_spec=True)
        intervals = mktSpec[self.mkt_type][mkt_spec]['interval_durations']
        system_offers = du.update_system_offers(self.resources_df, self.participant_res, uid,
                                                self.prev_disp_uid, intervals, self.forecast_dict,
                                                run_dir=self.run_dir, drw=self.drw_fact,
                                                outages=self.outages)
        self._offer_data = system_offers
        t1 = tm.time()
        times = du.get_time_from_system(uid, run_dir=self.run_dir)
        # Now get the competitor offers, either from input dictionary or loaded from json files
        prev_offers = {}
        # TODO: clean up this if/else - most sections aren't used anymore
        if offers is None and not skip:
            if self.host == 'local':
                offers = self._load_local_offer(uid)
            elif self.host == 'VM' and self.mode == 'competition':
                offers = self._load_webserver_offer(uid)
        if skip:
            pass
        elif offers == 'zero':
            for rid in self.balance_sheet.res_participant.keys():
                pid = self.balance_sheet.res_participant[rid]
                zero_offer = du.make_zero_offer(rid, times, self.resources_df)
                prepped_offer = du.prep_offer_for_gdx(pid, zero_offer, self.resources_df,
                                                      self.participant_res)
                self._append_offer_data(offer_data=prepped_offer, offer_key=None)
            self.logger.debug("Made a zero offer")
        elif offers == 'empty':
            for rid in self.balance_sheet.res_participant.keys():
                rid_offer = du.fill_from_previous_offer(rid, uid, run_dir=self.run_dir)
                if rid_offer is None:
                    print("On empty offer couldn't load previous. Reverting to system default.")
                    continue
                pid = self.balance_sheet.res_participant[rid]
                prepped_offer = du.prep_offer_for_gdx(pid, rid_offer, self.resources_df,
                                                      self.participant_res)
                self._append_offer_data(offer_data=prepped_offer, offer_key=None)
            self.logger.debug("Used previous offer data for empty offer.")
        elif type(offers) != dict:
            # type(list(offers.values())[0]) != dict:
            print("No valid offer found")
            self.logger.debug("Input offer appears to have an invalid format.")
            pass
        else:
            for rid in self.balance_sheet.res_participant.keys():
                time_in_offer = None
                # Check if the offer is available. If not skip (will use system default offer)
                try:
                    _ = offers[rid]
                except KeyError:
                    print(f"No valid offer found for rid {rid}")
                    continue
                for key, val in offers[rid].items():
                    if type(val) == dict:
                        time_in_offer = list(val.keys())[0]
                        break
                    else:
                        continue
                # If offer timestamps match, use, otherwise this is a previous offer...
                if times[0] == time_in_offer:
                    du.save_previous_offer(offers[rid], rid, uid, run_dir=self.run_dir)
                    rid_offer = {rid: offers[rid]}
                else:
                    rid_offer = du.fill_from_previous_offer(rid, uid, {rid:offers[rid]}, run_dir=self.run_dir)
                    if rid_offer is None:
                        continue
                    prev_offers[rid] = rid_offer[rid]
                    #rid_offer = du.make_zero_offer(rid, times, self.resources_df)
                pid = self.balance_sheet.res_participant[rid]
                prepped_offer = du.prep_offer_for_gdx(pid, rid_offer, self.resources_df,
                                                      self.participant_res, validate=True,
                                                      run_dir=self.run_dir)
                self._append_offer_data(offer_data=prepped_offer, offer_key=None)
            self.logger.debug("Added received offer data")
        t2 = tm.time()
        self._offer_data_to_gdx(uid)
        t3 = tm.time()
        # print(f"  System offer time {t1-t0:.3f}s")
        # print(f"  Participant offer time {t2-t1:.3f}s")
        # print(f"  GDX write time {t3-t2:.3f}s")
        return prev_offers

    def _append_offer_data(self, offer_data, offer_key):
        for key, value in offer_data.items():
            if offer_key is None:
                insert = value
            else:
                insert = {offer_key: value}
            if key in self._offer_data.keys():
                self._offer_data[key].update(insert)
            else:
                self._offer_data[key] = insert

    def _offer_data_to_gdx(self, uid):
        directory = join(self.run_dir,'market_clearing/offer_data/')
        du.offer_to_gdx(data_dict=self._offer_data, filename=uid, directory=directory)

    def solve_market(self, uid, update_uid=True):
        directory = join(self.run_dir,'market_clearing/system_data/')
        du.json_to_gdx(directory=directory, filename=uid)
        success = self._run_market(uid, update_uid=update_uid)
        # Update drw factor for forecast interpolation
        self.drw_fact = du.drw_mr(self.drw_fact)
        return success

    def _run_market(self, uid, update_uid=True, results_dir=None, save_gams_lst=False):

        if self.prev_mkt_uid['all'] is None:
            prev_uid = 'pre_start'
            if results_dir is None:
                results_dir = 'market_clearing/results'
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
        else:
            prev_uid = self.prev_mkt_uid['all']
        # resource_filepath = os.path.join(os.getcwd(),f"market_clearing/offer_data/{uid}")
        # print("Resources at:", resource_filepath)
        if self.run_dir == '.':
            mkt_path = self.run_dir
        else:
            mkt_path = join(self.run_dir, 'market_clearing')
        try:
            mkt_result = subprocess.run(['gams','market_clearing_engine.gms',"--market_uid", f'{uid}',
                                     "--previous_uid", f'{prev_uid}', "--run_path",
                                     f"{mkt_path}"],
                                    capture_output=True, text=True, check=True,
                                    cwd=os.path.join(os.getcwd(),'market_clearing'))
            if save_gams_lst:
                shutil.copy(os.path.join(os.getcwd(),'market_clearing','market_clearing_engine.lst'),
                            os.path.join(os.getcwd(),'market_clearing',f'market_clearing_engine_{uid}.lst'))
            self.logger.debug(mkt_result.stdout)
            self.logger.debug(mkt_result.stderr)
            self.logger.info(f'Cleared market at the interval {uid}')
            success = True
        except subprocess.CalledProcessError as e:
            print("Call to market_clearing_engine.gms failed. Error message:")
            print(e.output)
            self.logger.error(f'Failed to clear market at the interval {uid}. Infeasible solution found by GAMS.')
            success = False
        # Once done with market, update the previous uid (both overall and for this mkt_spec)
        tjunk, mkt_spec = du.uid_to_time(uid, return_mkt_spec=True)
        self.prev_mkt_uid['all'] = uid
        self.prev_mkt_uid[mkt_spec] = uid
        return success

    def physical_dispatch(self, uid, mkt_config, storage_df, skip=False):
        '''
        For physical intervals, does the following:
            preps dispatch (writing gdx files as needed)
            runs the battery dispatch model (written in gams)
            saves historic state for competitors and power, soc, and lmp for analysis
            computes battery degradation cost (daily)
        '''
        # Only run dispatch on a physical interval
        if 'PHYS' != mkt_config.interval_types[0][1].upper() or skip:
            return True
        # First update all necessary gdx files (attributes only need to be updated on the first run)
        if self.prev_disp_uid == None:
            update_attr = True
            prev_uid = 'pre_start'
        else:
            update_attr = False
            prev_uid = self.prev_disp_uid
        this_time, mkt_spec = du.uid_to_time(uid, return_mkt_spec=True)
        duration = mktSpec[self.mkt_type][mkt_spec]['interval_durations'][0][1]
        du.prep_dispatch(uid, self.prev_disp_uid, storage_df, duration=duration,
                         run_dir=self.run_dir, update_attr=update_attr)
        # Now run the battery_dispatch.gms script
        if self.run_dir == '.':
            str_path = self.run_dir
        else:
            str_path = join(self.run_dir, 'physical_dispatch/storage')
        try:
            disp_result = subprocess.run(['gams','battery_dispatch.gms',"--market_uid", f'{uid}',
                                      "--previous_uid", f'{prev_uid}', "--run_path",
                                      f"{str_path}"],
                                    capture_output=True, text=True, check=True,
                                    cwd=os.path.join(os.getcwd(),'physical_dispatch/storage/'))
            self.logger.debug(disp_result.stdout)
            self.logger.debug(disp_result.stderr)
            self.logger.info(f'Dispatched battery at the interval {uid}')
            success = True
        except subprocess.CalledProcessError as e:
            print(e.output)
            print(e.output)
            self.logger.error(f'Failed to execute physical dispatch at the interval {uid}. Infeasible solution found by GAMS.')
            success = False
        # Update the forecast history and pricing (physical dispatch intervals only)
        newrun = (self.prev_disp_uid == None)
        # Now save the power, soc, and LMP for use in later analysis
        # Note, surplus is saved in self.settle_positions inside the balance sheet object
        du.save_pow_soc(uid, self.resources_df['Master'], newrun, savedir=self.savedir,
                        run_dir=self.run_dir, save_sys=True)
        du.save_lmp(uid, newrun, savedir=self.savedir, run_dir=self.run_dir)
        # Once a day, compute the degradation cost
        if self.prev_deg_time == None:
            self.prev_deg_time = du.uid_to_time(uid)
        else:
            t_now, mkt_key = du.uid_to_time(uid, return_mkt_spec=True)
            t_delta = datetime.datetime.strptime(t_now, '%Y%m%d%H%M') - \
                datetime.datetime.strptime(self.prev_deg_time, '%Y%m%d%H%M')
            if t_delta.days >= 1:
                t_start = datetime.datetime.strptime(self.prev_deg_time, '%Y%m%d%H%M')
                # Load saved soc and temp for each battery
                times, socs, rids = du.load_saved_csv('soc', savedir=self.savedir,
                                                     run_dir=self.run_dir)
                times, temps, rids = du.load_saved_csv('temp', savedir=self.savedir,
                                                     run_dir=self.run_dir)
                times = pd.to_datetime(times)
                self.logger.info(f"Computing degradation costs at time {t_now}")
                battery_deg = Degradation(socs, temps, times, t_start, storage_df, rids=rids)
                deg_costs = battery_deg.compute_degradation_cost()
                self.balance_sheet.update_deg_cost(deg_costs, t_now)
                self.prev_deg_time = t_now
                if self.run_dir != '.':
                    self._delete_old_records(uid, dispatch=True)
        self.outages = du.check_for_outages(self.outages, uid)
        # Once done with dispatch, update the previous uid
        self.prev_disp_uid = uid
        return success

    def _init_outages(self):
        ''' Initializes an outage dictionary for all generators (none start in outage) '''
        genids = self.resources_df['Generators']['rid'].values
        outages = {}
        for genid in genids:
            outages[genid] = {'num_outages':0, 'in_outage':0, 'start_time':None, 'end_time':None}
        return outages

    def settle_positions(self, uid, mkt_config, skip=False):
        '''Collects settlement information from market clearing results and updates balance'''
        # Option to skip, used for getting an initial price forecasts without saving settlement
        if skip:
            return
        # First load that latest settlement data, optionally saving the results to file
        dense_time = du.update_settlement_gdx(uid, mkt_config, run_dir=self.run_dir)
        ledger, settlements = du.load_settlement_data(uid, self.resources_df, run_dir=self.run_dir)
        # Update the 5-minute time interval list for the next cycle
        # newrun = False
        if self.dense_time is None:
            self.dense_time = dense_time
            # newrun = True
        else:
            # Remove first time if it is a physical interval
            if mkt_config.interval_types[0][1].upper() == 'PHYS':
                self.dense_time = self.dense_time[1:]
            # Add in any new times from the settlement
            self.dense_time += [t for t in dense_time if t not in dense_time]
        this_time = du.uid_to_time(uid)
        # Now update competitor balance sheets
        self.balance_sheet.update(ledger, settlements, this_time)

    def save_state(self):
        '''
        Saves files relevant to the current state of the market clearing engine, allowing code to
        resume if interrupted.
        '''
        # First save the various trackers in the system that increment with each run
        run_time = tm.time() - self.start_time
        self.balance_sheet.save_to_file(fdir = self.run_dir)
        trackers = {}
        trackers['prev_mkt_uid'] = self.prev_mkt_uid
        trackers['prev_disp_uid'] = self.prev_disp_uid
        trackers['dense_time'] = self.dense_time
        trackers['prev_deg_time'] = self.prev_deg_time
        trackers['start_time'] = self.start_time
        trackers['run_time'] = run_time
        trackers['drw_fact'] = self.drw_fact
        self.outages = du.outage_datetime_to_str(self.outages)
        trackers['outages'] = self.outages
        trackers['part_res'] = self.participant_res
        with open(join(self.run_dir,'mkt_trackers.json'), 'w') as f:
            trackers = json.dump(trackers, f, indent=4)

    def _delete_old_records(self, uid, dispatch=False, keep_count=2):
        ''' Deletes old market or dispatch gdx and json files to reduce storage space '''
        record_dirs = ['market_clearing/system_data', 'market_clearing/offer_data',
                       'market_clearing/results']
        if dispatch:
            record_dirs = ['physical_dispatch/storage/data', 'physical_dispatch/storage/results']
        # Will delete all but the last 'keep_count' of this market type
        tjunk, mkt_spec = du.uid_to_time(uid, return_mkt_spec=True)
        for rd in record_dirs:
            gdx_files = [f for f in glob.glob(join(self.run_dir,rd,f'*{mkt_spec}*.gdx')) if \
                         (uid not in f)]
            # Also keep prev disp uid
            gdx_files = [f for f in gdx_files if str(self.prev_disp_uid) not in f]
            # Sort and don't delete the latest 'keep_count'
            gdx_files.sort()
            # If you don't have enough records, then don't delete anything
            if keep_count < len(gdx_files):
                gdx_files = gdx_files[:-keep_count]
            else:
                continue
            for gf in gdx_files:
                os.remove(gf)
                # If it exists, also remove a json file with the same name
                base, gdx_name = '/'.join(gf.split('/')[:-1]), gf.split('/')[-1].split('.')[:-1][0]
                json_file = join(base, f'{gdx_name}.json')
                try:
                    os.remove(json_file)
                except:
                    pass

def setup_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s/%(name)s - %(message)s')
    # Set up Handlers (clear any existing to avoid multiple instances)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(filename='_scheduler.log')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def launch_scheduler(args, params=None, mode='all', offers=None, run_root=None, newrun=False):
    modelist = ['all', 'prep_for_offers', 'clear_market']
    if mode not in modelist:
        raise ValueError(f"Run mode {mode} not an avaiable option. Choose from {modelist}")
    # Unpack input arguments
    # TODO: eventually get rid of params keyword altogether
    if newrun == False:
        newrun = args.new_run

    if params is None:
        mkt_type = args.mkt_type
        horizon = float(args.horizon)
        horizon_units = args.horizon_units
        start_date = args.start_date
        time_step = None
    else:
        mkt_params = params['market_params']
        mkt_type = mkt_params['mktSpec']
        horizon, horizon_units = float(mkt_params['horizon']), mkt_params['horizon_type']
        start_date = mkt_params['start_date']
        time_step = params['time_step']
    global NETWORK
    NETWORK = args.network # options are 'WECC', 'test'
    now = datetime.datetime.strptime(start_date, '%Y%m%d%H%M') - datetime.timedelta(minutes=5)
    # Remove any quotes and convert horizon to minutes and ensure within bounds
    if "'" in horizon_units:
        horizon_units = horizon_units.replace("'", "")
    if '"' in horizon_units:
        horizon_units = horizon_units.replace('"', '')
    if horizon_units == 'minutes':
        minutes = horizon
    elif horizon_units == 'hours':
        minutes = horizon*60
    elif horizon_units == 'days':
        minutes = horizon*24*60
    else:
        print(f'Unsupported horizon_units ({horizon_units}. Choose from "minutes",\
                         "hours", or "days".')
        info = {'market':{}, 'resources':{'time_limit':10}}
        status = {'done':True, 'success':False, 'disp':False}
        return info, status
        # raise ValueError(f'Unsupported horizon_units ({horizon_units}. Choose from "minutes",\
        #                 "hours", or "days".')
    # Truncate to last full 5 minutes
    minutes -= minutes % 5
    # Cap at one month, minimum of 5 minutes
    max_days = 365
    if minutes > 60*24*max_days:
        minutes = 60*24*max_days
    if minutes < 5:
        minutes = 5
    minutes = int(minutes)
    then = now + datetime.timedelta(minutes=minutes)
    # Set a directory for this run
    if args.host == 'local':
        run_dir = '.'
    elif args.mode == 'competition':
        # Make a unique competition save directory for each market/instance
        cnt = 0
        cdir_exists = True
        while cdir_exists:
            wroot = '/'.join(os.getcwd().split('/')[:-1])
            run_dir = f'{wroot}/submissions/competition_{args.mkt_type}_{cnt}'
            if not os.path.isdir(run_dir):
                cdir_exists = False
                os.makedirs(run_dir)
            else:
                if args.new_run:
                    break
                cnt += 1
    else:
        # Make a unique directory for each run
        user_dir = os.path.join(params['username'], params['uuid'])
        if run_root == None:
            run_dir = os.path.join(os.getcwd(),user_dir)
        else:
            run_dir = os.path.join(run_root,user_dir)
        print(f"Running test scheduler in mode {mode} with:\n\n", params) # , "\n\n", offers)
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    # Check for the latest gdx files to determine the last run.
    # gdx_dir = join(run_dir,'market_clearing/results')
    # last_time, last_mkt = du.uid_to_time(last_gdx.split('_')[-1].split('.')[0],
    #                                      to_datetime=True, return_mkt_spec=True)
    # cleared_gdx = [f for f in glob.glob(os.path.join(gdx_dir,'*.gdx'))]
    # else:
    #     cleared_gdx.sort(key=lambda x: os.path.getmtime(x))
    #     last_gdx = (cleared_gdx[-1]).split('/')[-1]
        
    # TODO: if mkt_trackers method works, eliminate the gdx searching algorithm
    mkt_track_file = join(run_dir, 'mkt_trackers.json')
    if os.path.isfile(mkt_track_file) and not newrun:
        with open(mkt_track_file, 'r') as f:
            mkt_trackers = json.load(f)
        prev_uid = mkt_trackers['prev_mkt_uid']['all']
        last_time, last_mkt = du.uid_to_time(prev_uid, to_datetime=True, return_mkt_spec=True)
        # If we clear the day-ahead-market, reset timer to 9am (15hrs back)
        if 'DAM' in last_mkt:
            last_time -= datetime.timedelta(minutes=900)
        last_time = datetime.datetime.strftime(last_time, '%Y%m%d%H%M')
    else:
        last_time = None
 
    options_dict = args.options
    scheduler = MarketScheduler(mkt_type, start=now, end=then, options_dict=options_dict,
                                last_time=last_time, run_dir=run_dir, time_step=time_step)
    if mode == 'prep_for_offers':
        json_out, status = scheduler.prep_for_offers()
    elif mode == 'clear_market':
        if mode == 'all': # Need to re-initialize scheduler/queue if doing all in sequence
            scheduler = MarketScheduler(mkt_type, start=now, end=then, options_dict=options_dict,
                                        last_time=last_time, run_dir=run_dir,
                                        time_step=time_step, clean=False)
        json_out, status = scheduler.simulate(offers)
    elif mode == 'all':
        json_out, status = scheduler.simulate_all()
    return json_out, status

class ArgOptions:
    ''' Class to handle arguments, including merge options text file and command line arguments '''
    def select_arguments(self, parser):
        ''' Combines input arguments and options file, saved to this object
            Command line arguments will override options file choices if there is a conflict
        '''
        cline_args = parser.parse_args()
        try:
            with open(cline_args.optfile, 'r') as f:
                opt_args = json.load(f)
        except FileNotFoundError:
            print(f"Options file {cline_args.optfile} not found in {os.getcwd()}")
            exit(0)
        loaded_args = list()
        self.options = dict() # Also save arguments into a dictionary
        for arg in opt_args.keys():
            # If this argument is included from the command line, use that choice
            if hasattr(cline_args, arg) and getattr(cline_args,arg) is not None:
                    setattr(self, arg, getattr(cline_args,arg))
                    self.options[arg] = getattr(cline_args,arg)
                    loaded_args.append(arg)
            # Otherwise use the value from the options file
            else:
                setattr(self, arg, opt_args[arg])
                self.options[arg] = opt_args[arg]
        # Add in extra command line arguments if they aren't alread loaded (and aren't empty)
        all_cline_args = [a[0] for a in cline_args._get_kwargs()]
        for arg in all_cline_args:
            if arg in loaded_args:
                continue
            elif getattr(cline_args, arg) is not None:
                setattr(self, arg, getattr(cline_args,arg))
                self.options[arg] = getattr(cline_args,arg)

    def add_argument(self, new_arg, new_val):
        # TODO: make so it doesn't silently overwrite existing args
        setattr(self, new_arg, new_val)
        self.options[new_arg] = new_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mkt_type', type=str,
                        help="Market types, can choose TS (Two-Settlement), MS (Multi-Settlement), or RHF (Rolling Horizon Forward).", choices=['TS', 'MS', 'RHF'])
    parser.add_argument('-d', '--horizon', type=int,
                        help="Choose the integer horizon (units of horizon are set in horizon_units).")
    parser.add_argument('-u', '--horizon_units', type=str,
                        help="Set the units of the horizon. May be 'minutes', 'hours', or 'days'. (Minumum of 5 minutes, maximum of 30 days).", choices=['days', 'hours', 'minutes'])
    parser.add_argument('-s', '--start_date', type=str,
                        help="Date/Time to start simulation. Format is YYYYmmddHHMM. Must be within the time stamp of the supplied forecast.")
    parser.add_argument('-l', '--language', type=str,
                        help="Code language desired for competitor algorithm.")
    parser.add_argument('-a', '--algorithm_name', type=str,
                        help="The name of the competitor algorithm to run.")
    # parser.add_argument('-t', '--last_time', type=str, default=None,
    #                     help="If resuming a stalled scheduler you can enter the last time cleared.")
    parser.add_argument('-n', '--new_run', action='store_true', help="Boolean flag to force a new run (overwrites previous saved files)")
    parser.add_argument('-o', '--optfile', type=str, help="Name of options file to use for custom options", default='options.json')
    args = ArgOptions()
    args.select_arguments(parser)
    setup_logger(log_level=logging.WARNING)
    logger = logging.getLogger("Main")
    if args.host == 'local' or args.mode == 'competition':
        launch_scheduler(args, mode='all')
        # Check total run time
        with open('mkt_trackers.json', 'r') as f:
            trackers = json.load(f)
        run_time = trackers['run_time']
        print(f"Total run time is {run_time:.3f}s")
        # Don't try to launch VM code (below)
        exit(0)
    # TODO: The below can probably be broken out into a separate script for running on the VM
    # Market Webserver Inputs
    market_params = {'horizon':args.horizon, 'horizon_type':args.horizon_units, 'mktSpec':
                     args.mkt_type, 'start_date':args.start_date, 'str_bus': None}
    params = {'uuid':'93458912', 'username':'mcornach', 'local_run':True, 'market_params':
              market_params, 'time_step':None}
    # params = {'uuid':'93458912', 'username':'liping', 'local_run':True, 'market_params':
    #             market_params, 'time_step':None}
    if args.host == 'VM':
        params['local_run'] = False
        try:
            sub_base = os.environ['SUB_DIR']
        except KeyError:
            print("Environmental varaiable 'SUB_DIR' must be set for the saved submission directory")
    cnt, newrun = 0, False
    if args.new_run:
        newrun = True
    isdone = False
    last_time_requested = {}
    last_time_cleared = {}
    # request = True
    while not isdone:
        if params['local_run']:
            t0 = tm.time()
            info, status = launch_scheduler(args, params, mode='all', newrun=newrun)
            t1 = tm.time()
            print(f"Total simulation time for step {params['time_step']} was {t1-t0:.3f}s")
            if status['done']:
                isdone = True
            elif not status['success'] or not status['disp']:
                print("Market success?", status['success'], "Dispatch?", status['disp'])
                break
            cnt += 1
            if args.new_run and cnt >=1:
                newrun = False
            params['time_step'] = status['time_step'] + 1
        else:
            tm.sleep(1)
            request = False
            participant_details = wu.get_next_participant()
            details = participant_details
            if participant_details == {}:
                print("Scanning for participants (None found)")
                offer_details = wu.get_next_offer()
                if offer_details is None:
                    print("Checking for offers (None found)")
                    continue
                print("Found offer:", offer_details.keys())
                details = offer_details
            # If found, parse participant details
            if 'market_params' in details.keys():
                request = True
                if details['market_params'] is not None:
                    params['market_params'] = details['market_params']
            else:
                market_params = wu.get_market_params(details['uuid'], details['division'])
                params['market_params'] = market_params
                details['market_params'] = market_params
                print("loaded market params:", market_params)
            params['uuid'] = f'{details["uuid"]}_{details["division"]}'
            try:
                params['username'] = details['team']
            except KeyError:
                username = details['deception_path'].split('/')[-2]
            params['time_step'] = details['time_step']
            uuid = details["uuid"]
            division = details["division"]
            # print(f"Send message test {uuid}_{division}")
            # wu.send_message(uuid, division, 'Update', f'Offer {details["time_step"]} failed to clear market. Terminating submission', '')
            # wu.terminate_submission(uuid, division)
            event = details["event"]
            time_step = details["time_step"]
            if time_step == 1:
                newrun = True
            # Logic to only request an offer once per time step
            # request = False
            # if uuid not in last_time_requested.keys():
            #     request = True
            # elif uuid in last_time_cleared.keys():
            #     if last_time_cleared[uuid] == last_time_requested[uuid]:
            #         request = True
            #     else:
            #         time_step -= 1
            # else:
            #     time_step -= 1
            if request:
                info, status = launch_scheduler(args, params, mode='prep_for_offers', run_root=sub_base, newrun=newrun)
                offer_request_details = {
                    "uuid": uuid,
                    "division": division,
                    "time_step": time_step,
                    "market": info['market'],
                    "resource": info['resources'],
                    "time_limit": info['resources']['time_limit']
                    }
                print(f'\nRequesting offer for {params["username"]}/{uuid} at time_step {time_step}')
                logger.info(f'Requested offer for {params["username"]}/{uuid} at time_step {time_step}')

                wu.request_offer(offer_request_details)
                last_time_requested[uuid] = time_step
            # Now we have to wait until the offer is ready
            if 'offer' in details.keys():
                try:
                    offers = json.loads(details['offer'])
                    update_offer_db = False
                except json.decoder.JSONDecodeError:
                    print(f'Warning: Found empty offer for {params["username"]}/{uuid} at time_step {time_step}')
                    print("Requesting previous offer")
                    try:
                        prev_offers = wu.get_recent_offer(uuid, division, details['market_params']['mktSpec'])
                        update_offer_db = True
                        print("Previous offer:\n", prev_offers.keys())
                        offers = json.loads(prev_offers['offer'])
                    except:
                        wu.terminate_submission(uuid, division)
                try:
                    settlement_details, status = launch_scheduler(args, params, mode='clear_market',
                                            offers=offers, run_root=sub_base, newrun=newrun)
                except:
                    # print(traceback.format_exc())
                    settlement_details = {}
                    status['prev_offer'] = ''
                    status['success'] = False
                    status['disp'] = False
                if update_offer_db:
                    print("Setting a previous offer for:", uuid, division, time_step)
                    # Make sure dictionary is in correct json format
                    adjusted_offer = json.dumps(status['prev_offer'])
                    adjusted_offer = json.loads(adjusted_offer)
                    wu.set_offer(uuid, division, time_step, adjusted_offer)
                if status['success'] == False or status['disp'] == False:
                    print("GAMS market engine/battery dispatch failed to find a valid solution")
                    wu.terminate_submission(uuid, division)
                summary = settlement_details
                mkt_type = params["market_params"]["mktSpec"]
                wu.set_offer_settlement_summary(uuid, division, mkt_type, 'sandbox', None, time_step, summary)
                last_time_cleared[uuid] = time_step
                newrun = False # Only do a new run on the first pass through
                print("Request/clear last times:", last_time_requested, last_time_cleared)
                print(f'Processed offer for {params["username"]}/{uuid} at time_step {time_step}')
                logger.info(f'Processed offer for {params["username"]}/{uuid} at time_step {time_step}')
    # Print out total run time at the end for local run
    if params['local_run']:
        with open('mkt_trackers.json', 'r') as f:
            trackers = json.load(f)
        run_time = trackers['run_time']
        print(f"Total run time is {run_time:.3f}s")
