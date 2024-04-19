gen_offer = {}
gen_offer['cost_rgu'] = 5
gen_offer['cost_rgd'] = 5
gen_offer['cost_spr'] = 0
gen_offer['cost_nsp'] = 0
gen_offer['init_en'] = 100
gen_offer['init_status'] = 1
gen_offer['ramp_dn'] = 100
gen_offer['ramp_up'] = 100
gen_offer['cost_su'] = 1000
gen_offer['cost_op'] = 10
gen_offer['cost_sd'] = 0
gen_offer['block_g_mq'] = [100,50,50]
gen_offer['block_g_mc'] = [20,25,30]
gen_offer['pgmin'] = 100
gen_offer['pgmax'] = 200
gen_offer['rgumax'] = 20
gen_offer['rgdmax'] = 20
gen_offer['min_uptime'] = 1
gen_offer['min_downtime'] = 1
gen_offer['outage'] = 0
gen_offer['init_downtime'] = 0
gen_offer['init_uptime'] = 0

str_offer = {}
str_offer['cost_ch'] = 3
str_offer['cost_dc'] = 3
str_offer['cost_soc'] = 0
str_offer['cost_rgu'] = 3
str_offer['cost_rgd'] = 3
str_offer['cost_spr'] = 0
str_offer['cost_nsp'] = 0
str_offer['init_en'] = 0
str_offer['init_status'] = 1
str_offer['ramp_dn'] = 9999
str_offer['ramp_up'] = 9999
str_offer['block_ch_mc'] = [21,18]
str_offer['block_dc_mc'] = [26,34]
str_offer['block_soc_mc'] = [36,34] # None
str_offer['pmax_ch'] = 25
str_offer['pmax_dc'] = 25
str_offer['block_ch_mq'] = [15,10]
str_offer['block_dc_mq'] = [15,10]
str_offer['block_soc_mq'] = [30,20] # None
str_offer['soc_end'] = 0
str_offer['soc_begin'] = 0
str_offer['socmax'] = 50
str_offer['socmin'] = 0
str_offer['eff_ch'] = 0.9
str_offer['eff_dc'] = 1.0

ren_offer = {}
ren_offer['cost_rgu'] = 10
ren_offer['cost_rgd'] = 10
ren_offer['cost_spr'] = 0
ren_offer['cost_nsp'] = 0
ren_offer['init_en'] = 10
ren_offer['init_status'] = 1
ren_offer['ramp_dn'] = 9999
ren_offer['ramp_up'] = 9999
ren_offer['pvmin'] = 0
ren_offer['pvmax'] = 10
ren_offer['block_r_mq'] = [10]
ren_offer['block_r_mc'] = [0]

dem_offer = {}
dem_offer['cost_rgu'] = 4
dem_offer['cost_rgd'] = 4
dem_offer['cost_spr'] = 8
dem_offer['cost_nsp'] = 6
dem_offer['init_en'] = -112
dem_offer['init_status'] = 1
dem_offer['ramp_dn'] = 9999
dem_offer['ramp_up'] = 9999
dem_offer['pdmin'] = 0
dem_offer['pdmax'] = 112
dem_offer['block_d_mq'] = [100, 12]
dem_offer['block_d_mv'] = [2000, 200]

vir_offer = {}
vir_offer['block_dec_mq'] = 100 # None
vir_offer['block_inc_mq'] = 100 # None
vir_offer['block_v_mv'] = 30 # None
vir_offer['block_v_mc'] = 40 # None