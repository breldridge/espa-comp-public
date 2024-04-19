* ESPA-Comp: Market Data Loading Utility


Parameters
* Network
x(i,j,c)            Reactance of line from i to j on c
Bsus(i,j,c)         Susceptance of line from i to j on c
tmax(i,j,c)         Transmission limit of line from i to j on c
type(bus)           Type of bus (1=reference bus)
area(bus)           Load area of bus
baseMVA             Scaling parameter to have units in pu
linestatus(i,j,c)   Availabilty of line i j c    
monitored(i,j,c)    Indicator of whether to monitor the line limit of line i j c
* Intervals
pd(i,t)             Fixed load at bus i and period t
duration(t)         Duration of period t (in minutes)
duration_h(t)       Duration of period t (in hours)
prev_duration(t)    Duration of period t from previous market
;

* Load market data
$gdxin '%run_path%/system_data/%market_uid%.gdx'
$load bus,line,circuit,
$load x,tmax                                                 
$load type,area,baseMVA
$load linestatus,monitored
$load t_dense,t,physical,forward,advisory
$load duration
$gdxin
* Load previous market data
display '%run_path%/results/results_%previous_uid%.gdx';
$ifthen exist '%run_path%/results/results_%previous_uid%.gdx'
$gdxin '%run_path%/results/results_%previous_uid%.gdx'
$load lmp,mcp
$load prev_duration
$gdxin
$else
prev_duration(t)=duration(t);
$endIf

* derived line data
Bsus(i,j,c)$line(i,j,c) = -1/x(i,j,c);

* population duration_h variable for computations that need time in units of hours
duration_h(t) = duration(t) / 60;
