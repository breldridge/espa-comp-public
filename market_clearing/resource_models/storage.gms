*** ESPA-Comp: Storage Participation Model

Equations
c_str_surplus(n)        Total surplus (value - cost) of storage resource n's dispatch
c_str_phys(n)           Total surplus of storage resource n during a physical dispatch interval
c_str_en(n)             Total cost of energy dispatch provided by storage resource n
c_str_soc(n)            Total cost of stored energy
c_str_rgu(n)            Total cost of regulation up reserve provided by storage resource n
c_str_rgd(n)            Total cost of regulation down reserve provided by storage resource n
c_str_spr(n)            Total cost of spinning reserve provided by storage resource n
c_str_nsp(n)            Total cost of non-spinning reserve provided by storage resource n
c_str_ch_block(n,t)     Charge block conversion to total charge quantity for storage resource n
c_str_dc_block(n,t)     Discharge block conversion to total discharge quantity for storage resource n
c_str_soc_block(n,t)    State-of-charge block conversion to total state-of-charge for storage resource n
c_str_net_en(n,t)       Net energy injection of storage resource n in interval t
c_str_ramp_up(n,t)      Storage ramp-up constraint
c_str_ramp_down(n,t)    Storage ramp-down constraint
c_str_ch_max(n,t)       Charging status bounds for storage resource n in interval t
c_str_dc_max(n,t)       Discharging status bounds for storage resource n in interval t
c_str_dispatch_ub(n,t)  Dispatch upper bound capacity with reserves
c_str_disaptch_lb(n,t)  Dispatch lower bound capacity with reserves
c_str_energy_ub(n,t)    State-of-charge upper bound for reserves
c_str_energy_lb(n,t)    State-of-charge lower bound for reserves
c_str_spr_ramp(n,t)     Ramp capacity to provide spinning reserves
c_str_nsp_ramp(n,t)     Ramp capacity to provide non-spinning reserves
c_str_progression(n,t)  ISO-managed energy conservation constraint
c_str_end(n,t)          Ending state-of-charge constraint
;
* Variable upper/lower bounds
p_ch.lo(n,t)$str(n) = 0;
p_dc.lo(n,t)$str(n) = 0;
p_ch_block.lo(n,b,t)$str(n) = 0;
p_dc_block.lo(n,b,t)$str(n) = 0;
e_soc_block.lo(n,b,t)$str(n) = 0;
p_ch_block.up(n,b,t)$str(n) = block_ch_mq(n,b,t);
p_dc_block.up(n,b,t)$str(n) = block_dc_mq(n,b,t);
e_soc_block.up(n,b,t)$str(n) = block_soc_mq(n,b,t);
e_soc.lo(n,t)$str(n) = socmin(n);
e_soc.up(n,t)$str(n) = socmax(n);
r_rgu.lo(n,t)$str(n) = 0;
r_rgd.lo(n,t)$str(n) = 0;
r_spr.lo(n,t)$str(n) = 0;
r_nsp.lo(n,t)$str(n) = 0;

*** Storage Constraints
c_str_surplus(n)$str(n)..       z_surplus(n) =e= z_en(n) + z_soc(n) + z_rgu(n) + z_rgd(n) + z_spr(n) + z_nsp(n);
c_str_phys(n)$str(n)..          z_phys(n) =e= -sum((b,t)$(str_en_block(n,b) and phys(t)), duration_h(t) * (block_dc_mc(n,b,t) * p_dc_block(n,b,t) - block_ch_mc(n,b,t) * p_ch_block(n,b,t)))
                                              -sum((b,t)$(str_soc_block(n,b) and phys(t)), block_soc_mc(n,b,t) * e_soc_block(n,b,t));
c_str_en(n)$str(n)..            z_en(n) =e= -sum((b,t)$str_en_block(n,b), duration_h(t) * (block_dc_mc(n,b,t) * p_dc_block(n,b,t) - block_ch_mc(n,b,t) * p_ch_block(n,b,t)));
c_str_soc(n)$str(n)..           z_soc(n) =e= -sum((b,t)$str_soc_block(n,b), block_soc_mc(n,b,t) * e_soc_block(n,b,t));
c_str_rgu(n)$str(n)..           z_rgu(n) =e= -sum(t, cost_rgu(n,t) * r_rgu(n,t));
c_str_rgd(n)$str(n)..           z_rgd(n) =e= -sum(t, cost_rgd(n,t) * r_rgd(n,t));
c_str_spr(n)$str(n)..           z_spr(n) =e= -sum(t, cost_spr(n,t) * r_spr(n,t));
c_str_nsp(n)$str(n)..           z_nsp(n) =e= -sum(t, cost_nsp(n,t) * r_nsp(n,t));
c_str_ch_block(n,t)$str(n)..    sum(b$str_en_block(n,b), p_ch_block(n,b,t)) - p_ch(n,t) =e= 0;
c_str_dc_block(n,t)$str(n)..    sum(b$str_en_block(n,b), p_dc_block(n,b,t)) - p_dc(n,t) =e= 0;
c_str_soc_block(n,t)$str(n)..   sum(b$str_soc_block(n,b), e_soc_block(n,b,t)) - e_soc(n,t) =e= 0;
c_str_net_en(n,t)$str(n)..      p_en(n,t) =e= p_dc(n,t) - p_ch(n,t);
c_str_ramp_up(n,t)$ren(n)..     p_en(n,t) - p_en(n,t-1)$(ord(t)>1) - init_en(n)$(ord(t)=1) =l= duration(t) * ramp_up(n);
c_str_ramp_down(n,t)$ren(n)..   p_en(n,t-1)$(ord(t)>1) + init_en(n)$(ord(t)=1) - p_en(n,t) =l= duration(t) * ramp_dn(n);
c_str_ch_max(n,t)$str(n)..      p_ch(n,t) =l= dcmax(n,t) * u_status(n,t);
c_str_dc_max(n,t)$str(n)..      p_dc(n,t) =l= chmax(n,t) * (1 - u_status(n,t));
c_str_dispatch_ub(n,t)$str(n).. p_en(n,t) + r_rgu(n,t) + r_spr(n,t) + r_nsp(n,t) =l= dcmax(n,t);
c_str_disaptch_lb(n,t)$str(n).. -p_en(n,t) + r_rgd(n,t) =l= chmax(n,t);
c_str_energy_ub(n,t)$str(n)..   e_soc(n,t) - duration_rgu / 60 * r_rgu(n,t) - duration_spr / 60 * r_spr(n,t) - duration_nsp / 60 * r_nsp(n,t) =g= socmin(n);
c_str_energy_lb(n,t)$str(n)..   e_soc(n,t) + duration_rgd / 60 * r_rgd(n,t) =l= socmax(n);
c_str_spr_ramp(n,t)$str(n)..    r_rgu(n,t) + r_spr(n,t) =l= response_spr * ramp_up(n);
c_str_nsp_ramp(n,t)$str(n)..    r_rgu(n,t) + r_spr(n,t) + r_nsp(n,t) =l= response_nsp * ramp_up(n);
c_str_progression(n,t)$str(n).. e_soc(n,t) =e= e_soc(n,t-1)$(ord(t)>1) + soc_begin(n)$(ord(t)=1) + duration_h(t) * (eff_ch(n) * p_ch(n,t) - (1/eff_dc(n)) * p_dc(n,t));
c_str_end(n,t)$(str(n) and ord(t)=card(t))..    e_soc(n,t) =g= soc_end(n);    
