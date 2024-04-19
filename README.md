# espa-comp
This is the root directory for the ESPA-Comp simulation code.
Welcome!

Below you will find instructions on how to setup the
WEASLE code for ESPA-Comp, how to run the scheduler.py with 
different options, and how to view the results of a simulation.

The directory 'market_clearing' and subdirectories hold the supporting
scripts in Python and GAMS to compute all parts of the market clearing.

The directory 'physical_dispatch' and subdirectories hold scripts to
simulate the physics-based dispatch (using target power from the
market clearing unit committment schedules).

# Getting Started
## Requirements
ESPA-Comp requires Python 3 and GAMS (tested with version 44).
We recommend installing Python through Anaconda (https://www.anaconda.com/download).
GAMS can be downloaded from (https://www.gams.com/download/) and 
requires a license for full functionality.

You may wish to create an ESPA-Comp environment:
```
conda create --name espa-comp
```
ESPA-Comp will require the following Python packages (mostly available
through `conda install <package>` or `pip install <package>`):
- strenum (pip)
- pandas (conda)
- numpy (conda)
- scipy (conda)
- matplotlib(conda)
- pyarrow (conda)
- fastparquet (conda)
- socket (conda)
- requests (conda)

To install the GAMS python API, follow the instructions on the GAMS
website (https://www.gams.com/latest/docs/API_PY_GETTING_STARTED.html).
If you already have conda installed, you should be able to install the
GAMS python API with
```
pip install gamsapi[all]
```
## Setup
Once you have your environment set, clone the repo and nagivate to the cloned
directory. To load generator information, you must have access to the shared
folder for WEASLE on Microsoft Teams (we can change the location later if we
want to expand access to a larger group). Make sure you have a local copy of
the WEASLE General folder (likely through OneDrive). You may set an environmental
variable `WEASLE_DIR` that points to this directory.

Now run
```
python setup.py
```
This will copy over forecast and system information to market_clearing/system_data
as well as some files used for making new forecasts in the forecasting directory.

Once these files are copied, the setup file will invoke `load_gen_info.py` which
will make the file market_clearing/offer_data/resource_info.xlsx and the file
scale_factors.json. If there is an error or you want to make changes, you may
directly run `python load_gen_info.py`.

You may want to verify that the appropriate files have been transferred. If they
haven't, the scheduler will raise an exception. Assuming all is well, you are
now ready to run the market simulation.

# Running a Simulation

To run a simulation, you can simply enter
```
python scheduler.py [-options]
```
The scheduler options (viewable with `python scheduler.py -h`) are:
[Note, we can move this to a config file, or make the option to use
a config file or command line arguments]
* -m: Market type - can be TS, MS, RHF, or TEST (TS default)
* -d: Horizon - the time frame of the simulation horizon, start to finish (5 default)
* -u: Horizon Units - time units can be minutes, hours, or days (minutes default)
* -s: Start Date - String formatted as YYYYmmddHHMM. Year is ignored, but month, day, etc. will determine the part of the year the simulation begins (201801280000 default)
* -n: Newrun - A boolean on weather to do a new run or not. Default behavior is to continue a previous run.

For example, to initiate a Multi-Settlement market for 30 days starting at midnight
on June 1st and forcing a newrun, you would enter
```
python scheduler.py -m MS -d 30 -u days -s 202306010000 -n
```
The scheduler will display progress information as it proceeds.
On a 3.49GHz CPU with 16GB RAM, each market clearing cycle takes
on the order of 10s. Thus a 30 day simulation will require approximately
24 hours of CPU time to complete.

Scheduler will save its state. Therefore, if you need to cancel a run
partway through, you may resume it later, as long as you do not include
the -n flag. To continue the run from above, you would enter
```
python scheduler.py -m MS -d 30 -u days -s 202306010000
```
WARNING: Right now scheduler.py does not support multiple simultaneous
market runs. If you start a new scheduler while a previous instance is 
running, it will invalidate the data and likely raise an error. [This 
functionality can be added later if desired]

Results from the run are saved to `saved\<Market Type>_market`. The
.gdx and .json files created during the run are also saved by default.
On a Linux/Mac, unwanted .gdx and .json files can be cleared by running
`./clear_test.sh` (no Windows version written at present).

# Modifying Characteristics
## Adding Participants/Changing Resources
The default participant is 'mcornach' with a participant id of p00002.
The default storage unit for bidding has the resource id R00229 and is
located at the CISD bus.

To modify these details, edit `market_clearing\system_data\participant_info.json`.
You need may add a dictionary with a new username, participant id (must be unique),
and resource (not required to be unique unless running multiple users at once). You
will also need to edit the username in the `params` dictionary in the main function
inside scheduler [improvements to this are planned...]

You must make a new directory inside `market_clearing\system_data` with the
name `participant_<participant_id>` where participant_id (ex. p00001) must
match the id in participant_info.json. Storage bidding algorithms can now be
placed in this directory (see the following section).

## Storage Unit Bidding

You may add a storage unit bidding algorithm to any participant directory. The
algorithm must have the name `market_participant.py` (additional code is allowed,
but scheduler will only call market_participant.py). [Alternative code languages
can be used, but none are supported at present, so scheduler.py would also need
modifications].

For details on properly configuring a storage unit bidding algorithm, see Section 2
of the Participation Guide on the ESPC-Comp Website (https://espa-competition.pnnl.gov/)
(See References -> Participation Guide).

# Viewing Results
Saved files can be viewed directly, or seen by running (note the -i)
```
python -i plot_results.py
```
This script has the directory hardcoded to `saved\TS_market` so if you want
to view other market results, you must edit the code directly.

Saved files include:
- lmp.csv: The system LMPs for every physical interval at each bus (reserve MCPs also included)
- power/soc/temp.csv: The power, state-of-charge, and temperature over time for all 12 storage units included in the default simulation.
- profit_[participant_id].csv: Each participant's profit, broken down by type
- ledger_[participant_id].json: A detailed ledger with every forward position cleared over the course of the simulation 
