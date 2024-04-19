*** ESPA-Comp: Flexible Load Participation Model

Equations
c_vir_surplus(n)        Total surplus (value - cost) of virtual resource n
c_vir_en_block(n,t)     Net energy consumed by virtual resource n in interval t
;

* Variable upper/lower bounds
p_en.fx(n,t)$(vir(n) and phys(t)) = 0;
p_dec_block.up(n,b,t)$vir(n) = block_dec_mq(n,b,t);
p_inc_block.up(n,b,t)$vir(n) = block_inc_mq(n,b,t);
p_dec_block.lo(n,b,t)$vir(n) = 0;
p_inc_block.lo(n,b,t)$vir(n) = 0;
p_en_block.lo(n,b,t)$vir(n) = 0;
r_rgu.fx(n,t)$vir(n) = 0;
r_rgd.fx(n,t)$vir(n) = 0;
r_spr.fx(n,t)$vir(n) = 0;
r_nsp.fx(n,t)$vir(n) = 0;
z_phys.fx(n)$vir(n) = 0;

*** Virtual Resource Constraints
c_vir_surplus(n)$vir(n)..       z_surplus(n) =e= sum((b,t)$vir_block(n,b), duration_h(t)*(block_v_mv(n,b,t) * p_dec_block(n,b,t) - block_v_mc(n,b,t) * p_inc_block(n,b,t)));
c_vir_en_block(n,t)$vir(n)..    p_en(n,t) =e= sum(b$vir_block(n,b), p_inc_block(n,b,t) - p_dec_block(n,b,t));
