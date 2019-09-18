# set JULIA_NUM_THREADS=8
using FileIO, DataFrames, CSV, Statistics, StatsBase, LinearAlgebra, GLM
DTDIR = "E:/NutStore/Research/mig_leftbh_enrollment"
WKDIR = "E:/Dropbox/GitHub/Migration_Left_Behind_BLP/msl_indept"
# WKDIR = "/Users/probmetrics/Dropbox/GitHub/Migration_Left_Behind_BLP/msl_indept"
# DTDIR = "/Users/probmetrics/NutStore/Research/mig_leftbh_enrollment"

include("$WKDIR/obj_helper_funs.jl")
include("$WKDIR/data_prepare_msl.jl")
# include("msl_llk.jl")
include("$WKDIR/msl_llk_loop.jl")
include("$WKDIR/squarem_helper_funs.jl")
include("$WKDIR/blp_squarem.jl")
include("$WKDIR/msl_est_iter.jl")


##
## 1. load choice data from MPS
##

LeftbhData = CSV.read("$DTDIR/mig_leftbh_enroll_fit.csv"; type = Float64)
# LeftbhData = sort(LeftbhData, (:year, :hhtype, :ID, :cline, :city_alts))
LeftbhData[:cage9] = Float64.(LeftbhData[:cagey] .> 8)
LeftbhData[:cagey] = LeftbhData[:, :cagey] / 10
LeftbhData[:cageysq] = LeftbhData[:, :cagey].^2
LeftbhData[:nchild_lnhp] = LeftbhData[:, :lnhprice].* LeftbhData[:, :nchild]
YL = Vector{Float64}(LeftbhData[:, :chosen])
YM = Vector{Float64}(LeftbhData[:, :child_leftbh])

XMnames = [:lndist, :cross_prov, :cross_regn, :lndist_crsprov, :lndist_crsregn,
           :amenity_pca_flowdur, :migdist_flowdur, :amenity_pca_highsch_f,
           :migdist_highsch_f, :amenity_pca_highsch_m, :migdist_highsch_m,
           :amenity_pca_age_f, :migdist_age_f, :amenity_pca_age_m, :migdist_age_m]
XTnames = [:highsch_f, :highsch_m, :age_f, :han]
XFnames = [:htreat, :migscore_fcvx_city, :lnhprice, :migscore_treat, :lnhp_treat,
		   :nchild_lnhp]
XLnames = [:cfemale, :nchild, :cagey, :cageysq, :age_m]
XQnames = [:cfemale, :cagey, :nchild, :highsch_f, :highsch_m, :age_f]
XQJnames = [:tstu2_ratio, :sschool_per]

lnDataShare, Delta_init, lnW, lnP, XQJ_mig, wgt, sgwgt,
XT, XM, XL, XF, XQ, nalt, nind, dgvec = data_prepare(LeftbhData, XQJnames,
			 XTnames, XLnames, XFnames, XMnames, XQnames; trs = true)

##
## 2. get random draw
##

MigBootData = CSV.read("$DTDIR/mig_leftbh_indboot.csv")

# --- bootstrap random shock ---
nsim = 20
ndraw = nind * nsim
alpha = 0.12

# bootstrap observed preference vars.
USHK = dropdims(draw_shock(ndraw; dims = 1); dims = 2) # draw iid standard normal random shock

##
## 3. search for initial values
##

lftvar = [XTnames; XFnames; XLnames; XQJnames]
lft_form = @eval @formula(child_leftbh ~ $(Meta.parse(join(lftvar, " + "))) + lnhinc_alts)
lft_fit = glm(lft_form, view(LeftbhData, YL .== 1, :),
			  Binomial(), LogitLink(), wts = wgt)
lft_init = coef(lft_fit)

lnq_form = @eval @formula(cog_adj ~ leftbh + clnhinc + $(Meta.parse(join(XQnames, " + "))) +
						  tstu2_ratio + sschool_per + leftbh&clnhinc + leftbh&tstu2_ratio +
						  leftbh&sschool_per)
lnq_fit = lm(lnq_form, MigBootData)
lnq_init = coef(lnq_fit)

XbQ = XQ' * [lnq_init[1]; lnq_init[4:end-5]]

##
## 4. Evaluate the likelihood
##

initset = load("$DTDIR/msl_indept_results/msl_indept_est_20190807.jld2")
initpar = initset["coefx"]
initdel = initset["deltas"]
xm_init = initpar[11:25]
xf_init = [initpar[26:31]; initpar[33]; -0.4]

nparm = size(XT, 1) + size(XL, 1) + size(XM, 1) + size(XF, 1)  +
		2*size(XQJ_mig, 1) + 1

xt_init = [3.0; lft_init[2:size(XT, 1)]]
xl_init = lft_init[(size(XT, 1) + size(XF, 1) - 1):(size(XT, 1) + size(XF, 1) + size(XL, 1) - 2)]
xqj_init = lft_init[(end-size(XQJ_mig, 1)):end-1]

bw = 0.194
blft = -0.148
bitr = -0.167

initval = [xt_init; xl_init; xm_init; xf_init; zeros(length(xqj_init)); xqj_init; -1.0]

parm_names = [:XT_cons; Symbol.("XT_" .* string.(XTnames)); Symbol.("XL_" .* string.(XLnames));
			  Symbol.("XM_" .* string.(XMnames)); :XF_cons; Symbol.("XF_" .* string.(XFnames));
			  :XF_lnW; Symbol.("XQJ_Mig_" .* string.(XQJnames));
			  Symbol.("XQJ_Dif_" .* string.(XQJnames)); :lnsigu]

# --- iterative maximization ---
ret_msl = msl_est_iter(initval, lnDataShare, initdel, YL, YM, lnW, lnP, XbQ,
			 			XQJ_mig, XT, XL, XM, XF, USHK, wgt, sgwgt, nind,
			 			nalt, nsim, dgvec; biter = 1)

# --- evaluate log-likelihood ---
@time mig_leftbh_llk(initval, initdel, YL, YM, lnW, lnP, XbQ, XQJ_mig,
			 		 XT, XL, XM, XF, USHK, wgt, nind, nalt, nsim, dgvec,
					 0.12, 1)
# 444022.5091; 5.1s

@time mig_leftbh_llk_thread(initval, initdel, YL, YM, lnW, lnP, XbQ, XQJ_mig,
			 		 		XT, XL, XM, XF, USHK, wgt, nind, nalt, nsim, dgvec,
							0.12, 1)
# 444022.5091; 0.9s for 8 threads
#
# --- test for ForwardDiff ---
# using Optim, ForwardDiff
# llk_opt = parm -> mig_leftbh_llk(parm, Delta_init, YL, YM, lnW, lnP, XQJ_mig,
# 			 		 			 XT, XL, XM, XF, XQ, ZSHK, USHK, wgt,
# 			   					 nind, nalt, nsim, dgvec, 0.12, 1)
# @time llk_opt(initval)
# @time gr = ForwardDiff.gradient(llk_opt, initval)

# llk_opt_thread = parm -> mig_leftbh_llk_thread(parm, Delta_init, YL, YM, lnW, lnP,
# 										XQJ_mig, XT, XL, XM, XF, XQ, ZSHK,
# 										USHK, wgt, nind, nalt, nsim, dgvec, 0.12, 1)
# @time llk_opt_thread(initval)
# @time gr = ForwardDiff.gradient(llk_opt_thread, initval)

#
# # --- predicted location choice probabilities ---
# ngrp = length(sgwgt)
# mktshare = zeros(nalt, ngrp)
# @time locpr_serial!(mktshare, initval, Delta_init, lnW, lnP, XQJ_mig, XT,
# 					XL, XM,	XF, XQ, ZSHK, USHK, wgt, sgwgt, nind, nalt,
# 					nsim, dgvec)
# @time locpr_thread!(mktshare, initval, Delta_init, lnW, lnP, XQJ_mig, XT,
# 					XL, XM,	XF, XQ, ZSHK, USHK, wgt, sgwgt, nind, nalt,
# 					nsim, dgvec)

# # --- BLP Contraction Mapping ---
# ngrp = length(sgwgt)
# delta_fpt = zeros(nalt, ngrp)
# delta_new = zeros(nalt, ngrp)
# delta_old = copy(Delta_init)
# delta_q1 = zeros(nalt, ngrp)
# delta_q2 = zeros(nalt, ngrp)
# @time fpt_squarem!(delta_fpt, delta_new, delta_old, delta_q1, delta_q2, lnDataShare,
# 				   initval, lnW, lnP, XQJ_mig, XT, XL, XM, XF, XQ, ZSHK, USHK,
# 				   wgt, sgwgt, nind, nalt, nsim, dgvec)
