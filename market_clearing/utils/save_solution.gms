
*** Save solution
$if not set outfile $set outfile '%run_path%/results/results_%market_uid%.gdx'

*----- price data --------------------*
lmp(i,t) = -c_balance.m(i,t) / duration_h(t);
lw_lmp(t)$phys(t) = sum(i,lmp(i,t) * sum(n$dem(n), p_en.l(n,t)$resource_at_bus(n,i))) / sum(i, sum(n$dem(n), p_en.l(n,t)$resource_at_bus(n,i)));
lw_lmp_all(t) = sum(i,lmp(i,t) * sum(n$dem(n), p_en.l(n,t)$resource_at_bus(n,i))) / sum(i, sum(n$dem(n), p_en.l(n,t)$resource_at_bus(n,i)));
mcp(t,'RGU') = c_rgu_supply.m(t);
mcp(t,'RGD') = c_rgd_supply.m(t);
mcp(t,'SPR') = c_spr_supply.m(t);
mcp(t,'NSP') = c_nsp_supply.m(t);


*----- schedule and dispatch ---------*
schedule(n,t,'EN')$fwd(t) = p_en.l(n,t);
schedule(n,t,'RGU')$fwd(t) = r_rgu.l(n,t);
schedule(n,t,'RGD')$fwd(t) = r_rgd.l(n,t);
schedule(n,t,'SPR')$fwd(t) = r_spr.l(n,t);
schedule(n,t,'NSP')$fwd(t) = r_nsp.l(n,t);
actual(n,t,'EN')$phys(t) = p_en.l(n,t);
actual(n,t,'RGU')$phys(t) = r_rgu.l(n,t);
actual(n,t,'RGD')$phys(t) = r_rgd.l(n,t);
actual(n,t,'SPR')$phys(t) = r_spr.l(n,t);
actual(n,t,'NSP')$phys(t) = r_nsp.l(n,t);
all_power(n,t,'EN') = p_en.l(n,t);
all_power(n,t,'RGU') = r_rgu.l(n,t);
all_power(n,t,'RGD') = r_rgd.l(n,t);
all_power(n,t,'SPR') = r_spr.l(n,t);
all_power(n,t,'NSP') = r_nsp.l(n,t);

*----- settlement data ----------*
delta(n,t,'EN')$(phys(t) or fwd(t)) = p_en_delta.l(n,t);
delta(n,t,'RGU')$(phys(t) or fwd(t)) = r_rgu_delta.l(n,t);
delta(n,t,'RGD')$(phys(t) or fwd(t)) = r_rgd_delta.l(n,t);  
delta(n,t,'SPR')$(phys(t) or fwd(t)) = r_spr_delta.l(n,t); 
delta(n,t,'NSP')$(phys(t) or fwd(t)) = r_nsp_delta.l(n,t);
settlement(n,'EN') =  sum(t$(phys(t) or fwd(t)), delta(n,t,'EN') * sum(i$resource_at_bus(n,i),lmp(i,t)));  
settlement(n,'RGU') = sum(t$(phys(t) or fwd(t)), delta(n,t,'RGU') * mcp(t,'RGU'));
settlement(n,'RGD') = sum(t$(phys(t) or fwd(t)), delta(n,t,'RGD') * mcp(t,'RGD'));
settlement(n,'SPR') = sum(t$(phys(t) or fwd(t)), delta(n,t,'SPR') * mcp(t,'SPR'));
settlement(n,'NSP') = sum(t$(phys(t) or fwd(t)), delta(n,t,'NSP') * mcp(t,'NSP'));
fwd_en(n,t(td))$fwd(t) = fwd_en(n,t) + delta(n,t,'EN');
fwd_rgu(n,t(td))$fwd(t) = fwd_rgu(n,t) + delta(n,t,'RGU');
fwd_rgd(n,t(td))$fwd(t) = fwd_rgd(n,t) + delta(n,t,'RGD');
fwd_spr(n,t(td))$fwd(t) = fwd_spr(n,t) + delta(n,t,'SPR');
fwd_nsp(n,t(td))$fwd(t) = fwd_nsp(n,t) + delta(n,t,'NSP');

*----- subsequent market data --------*
commit_status(n,t) = u_status.l(n,t);
prev_duration(t) = duration(t);

*----- market performance data -------*
market_summary('surplus_total') = z_mktsurplus.l;
market_summary('surplus_phys') = z_total_phys.l;
market_summary('surplus_dem') = sum(n$dem(n), z_phys.l(n));
market_summary('surplus_vir') = sum(n$vir(n), z_phys.l(n));
market_summary('surplus_gen') = sum(n$gen(n), z_phys.l(n));
market_summary('surplus_ren') = sum(n$ren(n), z_phys.l(n));
market_summary('surplus_str') = sum(n$str(n), z_phys.l(n));
market_summary('surplus_minus_str') = market_summary('surplus_phys') - market_summary('surplus_str');
market_summary('load_curtailed') = sum((i,t)$phys(t), max(sum(n$dem(n), p_en.l(n,t)$resource_at_bus(n,i)) - v_pd.l(i,t), 0));
market_summary('load_added') =     sum((i,t)$phys(t), max(v_pd.l(i,t) - sum(n$dem(n), p_en.l(n,t)$resource_at_bus(n,i)), 0));
market_summary('transmision_violation_cost') = sum((i,j,c,t)$(monitored(i,j,c) and phys(t)), z_pflow_vio.l(i,j,c,t));
market_summary('energy_imbalance_cost') = sum((i,t), z_en_excess.l(i,t) + z_en_shortage.l(i,t));
market_summary('rgu_excess_value')  = sum(t$phys(t), z_rgu_excess.l(t));
market_summary('rgu_shortage_cost') = sum(t$phys(t), z_rgu_shortage.l(t));
market_summary('rgd_excess_value')  = sum(t$phys(t), z_rgd_excess.l(t));
market_summary('rgd_shortage_cost') = sum(t$phys(t), z_rgd_shortage.l(t));
market_summary('spr_excess_value')  = sum(t$phys(t), z_spr_excess.l(t));
market_summary('spr_shortage_cost') = sum(t$phys(t), z_spr_shortage.l(t));
market_summary('nsp_excess_value')  = sum(t$phys(t), z_nsp_excess.l(t));
market_summary('nsp_shortage_cost') = sum(t$phys(t), z_nsp_shortage.l(t));
market_summary('reserve_excess_value')  = market_summary('rgu_excess_value') + market_summary('rgd_excess_value')
                                        + market_summary('spr_excess_value') + market_summary('nsp_excess_value');
market_summary('reserve_shortage_cost') = market_summary('rgu_shortage_cost') + market_summary('rgd_shortage_cost')
                                        + market_summary('spr_shortage_cost') + market_summary('nsp_shortage_cost');



execute_unload '%outfile%',lmp,lw_lmp,lw_lmp_all,mcp,schedule,actual,delta,settlement,fwd_en,fwd_rgu,fwd_rgd,fwd_spr,fwd_nsp,commit_status,prev_duration,p_flow.l,market_summary;
