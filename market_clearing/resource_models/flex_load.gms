*** ESPA-Comp: Flexible Load Participation Model

Equations
c_dem_surplus(n)        Total surplus (value - cost) of flexible demand resource n's consumption
c_dem_phys(n)           Total surplus of flexible demand resource n during a physical dispatch interval
c_dem_en(n)             Value of energy consumption by flexible demand resource n
c_dem_rgu(n)            Cost of regulation up reserve provided by flexible demand resource n
c_dem_rgd(n)            Cost of regulation down reserve provided by flexible demand resource n
c_dem_spr(n)            Cost of spinning reserve provided by flexible demand resource n
c_dem_nsp(n)            Cost of non-spinning reserve provided by flexible demand resource n
c_dem_en_block(n,t)     Net energy consumed by flexible demand resource n
c_dem_res_up_cap(n,t)   Capacity to provide upward reserve products from flexible demand resource n
c_dem_res_dn_cap(n,t)   Capacity to provide downward reserve products from flexible demand resource n
c_dem_ramp_up(n,t)      Ramp-up constraint for flexible demand resource n
c_dem_ramp_down(n,t)    Ramp-down constraint for flexible demand resource n
;

* Variable upper/lower bounds
p_en.up(n,t)$dem(n) = -pdmin(n,t);
p_en.lo(n,t)$dem(n) = -pdmax(n,t);
p_en_block.up(n,b,t)$dem(n) = block_d_mq(n,b,t);
p_en_block.lo(n,b,t)$dem(n) = 0;
r_rgu.lo(n,t)$dem(n) = 0;
r_rgd.lo(n,t)$dem(n) = 0;
r_spr.lo(n,t)$dem(n) = 0;
r_nsp.fx(n,t)$dem(n) = 0;

*** Flexible Load Constraints
c_dem_surplus(n)$dem(n)..       z_surplus(n) =e= z_en(n) + z_rgu(n) + z_rgd(n) + z_spr(n) + z_nsp(n);
c_dem_phys(n)$dem(n)..          z_phys(n) =e= sum((b,t)$(dem_block(n,b) and phys(t)), duration_h(t) * block_d_mv(n,b,t) * p_en_block(n,b,t));
c_dem_en(n)$dem(n)..            z_en(n) =e= sum((b,t)$dem_block(n,b), duration_h(t) * block_d_mv(n,b,t) * p_en_block(n,b,t));
c_dem_rgu(n)$dem(n)..           z_rgu(n) =e= -sum(t, cost_rgu(n,t) * r_rgu(n,t));
c_dem_rgd(n)$dem(n)..           z_rgd(n) =e= -sum(t, cost_rgd(n,t) * r_rgd(n,t));
c_dem_spr(n)$dem(n)..           z_spr(n) =e= -sum(t, cost_spr(n,t) * r_spr(n,t));
c_dem_nsp(n)$dem(n)..           z_nsp(n) =e= -sum(t, cost_nsp(n,t) * r_nsp(n,t));
c_dem_en_block(n,t)$dem(n)..    p_en(n,t) =e= -sum(b$dem_block(n,b), p_en_block(n,b,t));
c_dem_res_up_cap(n,t)$dem(n)..  p_en(n,t) + r_rgu(n,t) + r_spr(n,t) =l= -pdmin(n,t);
c_dem_res_dn_cap(n,t)$dem(n)..  p_en(n,t) - r_rgd(n,t) =g= -pdmax(n,t);
c_dem_ramp_up(n,t)$dem(n)..     p_en(n,t) - p_en(n,t-1)$(ord(t)>1) - init_en(n)$(ord(t)=1) =l= duration(t)*ramp_up(n);
c_dem_ramp_down(n,t)$dem(n)..   p_en(n,t-1)$(ord(t)>1) + init_en(n)$(ord(t)=1) - p_en(n,t) =l= duration(t)*ramp_dn(n);
