#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ESPA Webserver Utils: For interacting with market_webserver.py on the WEASLE virtual machine
import requests

url_base = "http://localhost:9001/"
request_offer_url = f"{url_base}/request_offer"
get_next_participant_url = f"{url_base}/get_next_participant"
get_market_params_url = f"{url_base}/get_conf_market_parameters"
get_next_offer_url = f"{url_base}/get_next_offer"
set_offer_settlement_summary_url = f"{url_base}/set_offer_settlement_summary"
get_recent_offer_url = f"{url_base}/get_recent_offer"
set_offer_url = f"{url_base}/set_offer"
terminate_submission_url = f"{url_base}/terminate_submission"
send_message_url = f"{url_base}/remote_message"

def clear_market():
    pass

def request_offer(offer_request_details):
    response = requests.get(request_offer_url, json=offer_request_details)

def get_next_participant():
    response = requests.get(get_next_participant_url)
    try:
        participant_details = response.json()
    except:
        participant_details = {}
    if participant_details == {}:
        return {}
    # print('participant_details', participant_details)
    return participant_details

def get_market_params(uuid, division):
    dict_out = {"uuid": uuid, "division": division}
    response = requests.get(get_market_params_url, json=dict_out)
    market_params = response.json()
    return market_params

def get_updated_forecast(participant_details):
    forecast =  { "wind": {"1": 22, "2": 33}, "solar": {"1":22, "2":33} }
    return forecast


def get_next_offer():
    response = requests.get(get_next_offer_url)
    try:
        offer_details = response.json()
        if offer_details == {}:
            return None
        # print('\nreceived offer_details', offer_details)
        return offer_details
    except:
        return None
        # = {uuid, div, mkt_params, time_step, status}

def get_recent_offer(uuid, division, mkt_spec):
    dict_out = {"uuid": uuid, "division": division, "uid": mkt_spec}
    response = requests.get(get_recent_offer_url, json=dict_out)
    try:
        offer_details = response.json()
        offer = offer_details["offer"]
    except: #Maybe JSONDecodeError?
        offer = {'offer':'empty'}
    return offer

def set_offer(uuid, division, time_step, offer):
    if offer == {}:
        offer == '' # Convert empty dict to empty str
    dict_out = {"uuid": uuid, "division": division, "time_step": time_step, "offer": offer}
    requests.get(set_offer_url, json=dict_out)
    
def get_settlement(offer_details):
    #clear the market here
    settlement_summary = {}
    return settlement_summary

def set_offer_settlement_summary(uuid, division, mkt_type, event, mkt_params, time_step, summary):
    dict_out = {"uuid": uuid, "division": division, "mkt_type":mkt_type, "event":event,
            "mkt_params":mkt_params, "time_step": time_step, "summary": summary}
    response = requests.get(set_offer_settlement_summary_url, json=dict_out)
    # print('response from send_settlement_summary', response)

def get_updated_status(participant_details):
    return participant_details["status"]

def send_message(uuid, division, status, status_note, value):
    dict_out = {"uuid": uuid, "division": division, "status": status, "status_note":status_note, "value": value}
    requests.get(send_message_url, json=dict_out) 

def terminate_submission(uuid, division):
    dict_out = {"uuid": uuid, "division": division}
    response = requests.get(terminate_submission_url, json=dict_out)
