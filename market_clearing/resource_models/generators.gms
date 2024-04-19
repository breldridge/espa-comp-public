*** ESPA-Comp: Generator Participation Model

Equations
c_gen_surplus(n)        Total surplus (value - cost) of generation resource n's dispatch
c_gen_phys(n)           Total surpls of generation resource n during a physical dispatch interval
c_gen_mc(n)             Total cost of generator n's energy dispatch
c_gen_su(n)             Total cost of generator n's start-ups
c_gen_op(n)             Total operating (no-load) cost of generator n
c_gen_sd(n)             Total cost of generator n's shut-downs
c_gen_rgu(n)            Total cost of spinning reserve provided by generator n
c_gen_rgd(n)            Total cost of non-spinning reserve provided by generator n
c_gen_en_block(n,t)     Net energy provided by generation resource n
c_gen_pmax(n,t)         Maximum dispatch level with commitment status for generation resource n in interval t
c_gen_pmin(n,t)         Minimum dispatch level with commitment status for generation resource n in interval t
c_gen_rgu_cap(n,t)      Capacity to provide regulation up reserve from genrerator n
c_gen_rgd_cap(n,t)      Capacity to provide regulation down reserve from generator n
c_gen_spr_cap(n,t)      Capacity to provide spinning reserve from generator n
c_gen_res_cap(n,t)      Total capacity to provide reserve products from generator n
c_gen_spr_ramp(n,t)     Ramp capacity to provide spinning reserves from generator n
c_gen_nsp_ramp(n,t)     Ramp capacity to prove non-spinning reserves from generator n
c_gen_min_uptime(n,t)   Minimum up-time constraint for generator n
c_gen_min_downtime(n,t) Minimum down-time constraint for generator n
c_gen_ramp_up(n,t)      Ramp-up constraint for generator n
c_gen_ramp_down(n,t)    Ramp-down constraint for generator n
c_gen_commit_logic(n,t) Commitment logic for generator n
;

* Variable upper/lower bounds
p_en.up(n,t)$gen(n) = pgmax(n,t);
p_en.lo(n,t)$gen(n) = 0;
p_en_block.up(n,b,t)$gen(n) = block_g_mq(n,b);
p_en_block.lo(n,b,t)$gen(n) = 0;
r_rgu.lo(n,t)$gen(n) = 0;
r_rgd.lo(n,t)$gen(n) = 0;
r_rgu.up(n,t)$gen(n) = rgumax(n);
r_rgd.up(n,t)$gen(n) = rgdmax(n);
r_spr.lo(n,t)$gen(n) = 0;
r_nsp.lo(n,t)$gen(n) = 0;


*** Generator Constraints
c_gen_surplus(n)$gen(n)..       z_surplus(n) =e= z_en(n) + z_su(n) + z_op(n) + z_sd(n) + z_rgu(n) + z_rgd(n);
c_gen_phys(n)$gen(n)..          z_phys(n) =e= -sum((b,t)$(gen_block(n,b) and phys(t)), duration_h(t) * block_g_mc(n,b) * p_en_block(n,b,t));
c_gen_mc(n)$gen(n)..            z_en(n) =e= -sum((b,t)$gen_block(n,b), duration_h(t) * block_g_mc(n,b) * p_en_block(n,b,t));
c_gen_su(n)$gen(n)..            z_su(n) =e= -sum(t, cost_su(n,t) * u_startup(n,t));
c_gen_op(n)$gen(n)..            z_op(n) =e= -sum(t, cost_op(n,t) * u_status(n,t));
c_gen_sd(n)$gen(n)..            z_sd(n) =e= -sum(t, cost_sd(n,t) * u_shutdown(n,t));
c_gen_rgu(n)$gen(n)..           z_rgu(n) =e= -sum(t, cost_rgu(n,t) * r_rgu(n,t));
c_gen_rgd(n)$gen(n)..           z_rgd(n) =e= -sum(t, cost_rgd(n,t) * r_rgd(n,t));
c_gen_en_block(n,t)$gen(n)..    p_en(n,t) =e= sum(b$gen_block(n,b), p_en_block(n,b,t));
c_gen_pmax(n,t)$gen(n)..        p_en(n,t) =l= pgmax(n,t) * u_status(n,t);
c_gen_pmin(n,t)$gen(n)..        p_en(n,t) =g= pgmin(n) * u_status(n,t);
c_gen_rgu_cap(n,t)$gen(n)..     p_en(n,t) + r_rgu(n,t) =l= pgmax(n,t) * u_status(n,t);
c_gen_rgd_cap(n,t)$gen(n)..     p_en(n,t) - r_rgu(n,t) =g= pgmin(n) * u_status(n,t);
c_gen_spr_cap(n,t)$gen(n)..     p_en(n,t) + r_rgu(n,t) + r_spr(n,t) =l= pgmax(n,t) * u_status(n,t);
c_gen_res_cap(n,t)$gen(n)..     p_en(n,t) + r_rgu(n,t) + r_spr(n,t) + r_nsp(n,t) =l= pgmax(n,t);
c_gen_spr_ramp(n,t)$gen(n)..    r_rgu(n,t) + r_spr(n,t) =l= response_spr * ramp_up(n);
c_gen_nsp_ramp(n,t)$gen(n)..    r_rgu(n,t) + r_spr(n,t) + r_nsp(n,t) =l= response_nsp / 60 * ramp_up(n);
c_gen_min_uptime(n,t)$gen(n)..  u_status(n,t) =g= sum(tt$((ord(tt) >= ord(t) - min_uptime(n) + 1) and (ord(tt) <= ord(t))), u_startup(n,t));
c_gen_min_downtime(n,t)$gen(n)..u_status(n,t) =l= 1- sum(tt$((ord(tt) >= ord(t) - min_downtime(n) + 1) and (ord(tt) <= ord(t))), u_shutdown(n,t));
c_gen_ramp_up(n,t)$gen(n)..     p_en(n,t) - p_en(n,t-1)$(ord(t)>1) - init_en(n)$(ord(t)=1) =l=
                                    (pgmin(n) + duration(t)*ramp_up(n)) * u_status(n,t)
                                    - pgmin(n)*(u_status(n,t-1)$(ord(t)>1) + init_status(n)$(ord(t)=1))
                                    - duration(t) * ramp_up(n) * u_startup(n,t);
c_gen_ramp_down(n,t)$gen(n)..     p_en(n,t-1)$(ord(t)>1) + init_en(n)$(ord(t)=1) - p_en(n,t) =l=
                                    (pgmin(n) + duration(t)*ramp_dn(n)) * (u_status(n,t-1)$(ord(t)>1) + init_status(n)$(ord(t)=1))
                                    - pgmin(n)*u_status(n,t)
                                    - duration(t) * ramp_dn(n) * u_shutdown(n,t) + outage(n,t)*init_en(n);
c_gen_commit_logic(n,t)$gen(n)..u_startup(n,t) - u_shutdown(n,t) =e= u_status(n,t) - u_status(n,t-1)$(ord(t)>1) - init_status(n)$(ord(t)=1);