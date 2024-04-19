* ESPA-Comp: Resource Data Loading Utility

Parameters
* Generic
cost_rgu(n,t)       Cost incurred per MWh of regulation up reserve
cost_rgd(n,t)       Cost incurred per MWh of regulation down
cost_spr(n,t)       Cost incurred per MWh of spinning reserve
cost_nsp(n,t)       Cost incurred per MWh of non-spinning reserve
init_en(n)          Initial dispatch level of resource n
init_status(n)      Operating status of resource n in the previous interval
ramp_dn(n)          Maximum decrease in dispatch level
ramp_up(n)          Maximum increase in dispatch level
* Storage
chmax(n,t)          Maximum dispatch level while charging
dcmax(n,t)          Maximum dispatch level while discharging
socmax(n)           Maximum state-of-charge of the storage resource
socmin(n)           Minimum state-of-charge of the storage resource
block_ch_mc(n,b,t)  Cost incurred per MWh dispatched in charging quantity block b in interval t
block_dc_mc(n,b,t)  Cost incurred per MWh dispatched in discharging quantity block b in interval t
block_soc_mc(n,b,t) Cost incurred per MWh in state-of-charge at the end of interval t
block_ch_mq(n,b,t)  Incremental battery charge quantity (negative dispatch) in block b
block_dc_mq(n,b,t)  Incremental battery discharge quantity (positive dispatch) in block b
block_soc_mq(n,b,t) State-of-charge offer block b
soc_end(n)          Desired minimum state-of-charge at the end of the dispatch horizon
soc_begin(n)        State-of-charge at the beginning of the dispatch horizon
eff_ch(n)           Battery efficiency during charging mode
eff_dc(n)           Battery efficiency during discharging mode
* Generators
cost_su(n,t)        Fixed cost of generator n to start up in interval t
cost_op(n,t)        Fixed cost of generator n to operate in interval t
cost_sd(n,t)        Fixed cost of generator n to shut down in interval t
pgmin(n)            Minimum production if generator n is online
pgmax(n,t)          Maximum production of generator n
rgumax(n)           Maximium regulation up from generator n
rgdmax(n)           Maximium regulation down from generator n
block_g_mq(n,b)     Maximum production of generator n in block b
block_g_mc(n,b)     Marginal cost of energy production of generator n in block b
min_uptime(n)       Minimum amount of time online after a start-up by generator n
min_downtime(n)     Minimum amount of time offline after a shut-down by generator n
init_downtime(n)    Accumulated downtime for resource in the initial period
init_uptime(n)      Accumulated uptime for resource n in the initial period
outage(n,t)         Outage status in period t
* Renewables
pvmin(n,t)          Minimum production of renewable resource n in interval t
pvmax(n,t)          Maximum production of renewable resource n in interval t
block_r_mq(n,b,t)   Renewable resource n's quantity block b in interval t
block_r_mc(n,b,t)   Renewable resource n's cost block b in interval t
* Flexible Demand
pdmin(n,t)          Minimum energy consumption of load n in interval t
pdmax(n,t)          Maximum energy consumption of load n in interval t
block_d_mq(n,b,t)   Quantity of energy consumption in block b in interval t
block_d_mv(n,b,t)   Marginal value or benefit of load n if consuming energy in block b in interval t
* Virtual
block_dec_mq(n,b,t) Quantity of virtual DEC block b in interval t
block_inc_mq(n,b,t) Quantity of virtual INC block b in interval t
block_v_mv(n,b,t)   Bid price of virtual DEC block b in interval t
block_v_mc(n,b,t)   Offer price of virtual INC block b in interval t
* Settlement
fwd_en(n,td)         Forward energy position of resource n in interval t
fwd_rgu(n,td)        Forward regulation up reserve position of resource n in interval t 
fwd_rgd(n,td)        Forward regulation down reserve position of resource n in interval t 
fwd_spr(n,td)        Forward spinning reserve position of resource n in interval t 
fwd_nsp(n,td)        Forward non-spinning reserve position of resource n in interval t 
* Commitment status
commit_status(n,t)      Unit commitment status of resource n in interval t
;  

* Load offer data
$gdxin '%run_path%/offer_data/%market_uid%.gdx'
$load resource,dem,gen,ren,str,vir
$load dem_block, gen_block, ren_block, str_en_block, str_soc_block, vir_block, resource_at_bus
$load chmax,dcmax,block_ch_mc,block_dc_mc,block_soc_mc,block_ch_mq,block_dc_mq,block_soc_mq,soc_end,soc_begin,socmax,socmin,eff_ch,eff_dc
$load cost_rgu,cost_rgd,cost_spr,cost_nsp,init_en,init_status,ramp_dn,ramp_up,rgumax,rgdmax
$load cost_su,cost_op,cost_sd,block_g_mq,block_g_mc,pgmin,pgmax,min_uptime,min_downtime,init_downtime,init_uptime,outage
$load pvmin,pvmax,block_r_mq,block_r_mc
$load pdmin,pdmax,block_d_mq,block_d_mv
$load block_dec_mq,block_inc_mq,block_v_mv,block_v_mc
$gdxin
* Load previous forward positions
display '%run_path%/results/results_%previous_uid%.gdx';
$ifthen exist '%run_path%/results/results_%previous_uid%.gdx'
$gdxin '%run_path%/results/results_%previous_uid%.gdx'
$load fwd_en,fwd_rgu,fwd_rgd,fwd_spr,fwd_nsp
$load commit_status
$gdxin
fwd_en(n,td)$(fwd_en(n,td)=0) = 0;
$else
fwd_en(n,td) = 0;
fwd_rgu(n,td) = 0;
fwd_rgd(n,td) = 0;
fwd_spr(n,td) = 0;
fwd_nsp(n,td) = 0;
commit_status(n,t) = 0;
$endIf

* Populate commit_status based on previous market durations.
Scalars cumulative,end_t;
loop(t$prev_duration(t),
  cumulative = 0;
  end_t = prev_duration(t);
  loop(tt$(ord(tt)>=ord(t)),
    cumulative = cumulative + duration(tt);
    commit_status(n,tt)$(cumulative <= end_t) = yes$commit_status(n,t);
  );
);
*option commit_status:0:0:2;
*display commit_status;
