*** ESPA-Comp: Renewable Participation Model

Equations
c_ren_surplus(n)        Total surplus (value - cost) of renewables resource n's dispatch
c_ren_phys(n)           Total surplus of renewables resource n during a physical dispatch interval
c_ren_en_block(n,t)     Net energy provided by renewable resource n
c_ren_rgd_cap(n,t)      Capacity to provide regulation down reserve from generator n
c_ren_spr_cap(n,t)      Capacity to provide spinning reserve from generator n
c_ren_ramp_up(n,t)      Ramp-up constraint for renewable resource n
c_ren_ramp_down(n,t)    Ramp-down constraint for renewable resource n
;

* Variable upper/lower bounds
p_en.up(n,t)$ren(n) = pvmax(n,t);
p_en.lo(n,t)$ren(n) = pvmin(n,t);
p_en_block.up(n,b,t)$ren(n) = block_r_mq(n,b,t);
p_en_block.lo(n,b,t)$ren(n) = 0;
r_rgu.lo(n,t)$ren(n) = 0;
r_rgd.lo(n,t)$ren(n) = 0;
r_spr.lo(n,t)$ren(n) = 0;
r_nsp.fx(n,t)$ren(n) = 0;

*** Renewable Constraints
c_ren_surplus(n)$ren(n)..       z_surplus(n) =e= -sum((b,t)$ren_block(n,b), duration_h(t) * block_r_mc(n,b,t) * p_en_block(n,b,t));
c_ren_phys(n)$ren(n)..          z_phys(n) =e= -sum((b,t)$(ren_block(n,b) and phys(t)), duration_h(t) * block_r_mc(n,b,t) * p_en_block(n,b,t));
c_ren_en_block(n,t)$ren(n)..    p_en(n,t) =e= sum(b$ren_block(n,b), p_en_block(n,b,t));
c_ren_rgd_cap(n,t)$ren(n)..     p_en(n,t) - r_rgd(n,t) =g= pvmin(n,t);
c_ren_spr_cap(n,t)$ren(n)..     p_en(n,t) + r_rgu(n,t) + r_spr(n,t) =l= pvmax(n,t);
c_ren_ramp_up(n,t)$ren(n)..     p_en(n,t) - p_en(n,t-1)$(ord(t)>1) - init_en(n)$(ord(t)=1) =l= duration(t)*ramp_up(n);
c_ren_ramp_down(n,t)$ren(n)..   p_en(n,t-1)$(ord(t)>1) + init_en(n)$(ord(t)=1) - p_en(n,t) =l= duration(t)*ramp_dn(n);
  