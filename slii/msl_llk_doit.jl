# set JULIA_NUM_THREADS=8
using FileIO, DataFrames, CSV, Statistics, StatsBase, LinearAlgebra, GLM
WKDIR = "D:/XuShu/mig_leftbh_gmm"
# WKDIR = "/Users/probmetrics/Dropbox/GitHub/Migration_Left_Behind_BLP/msl_indept"
# DTDIR = "/Users/probmetrics/NutStore/Research/mig_leftbh_enrollment"

include("$WKDIR/msl_indept/obj_helper_funs.jl")
include("$WKDIR/msl_indept/data_prepare_msl.jl")
# include("msl_llk.jl")
include("$WKDIR/msl_indept/msl_llk_loop.jl")
include("$WKDIR/msl_indept/squarem_helper_funs.jl")
include("$WKDIR/msl_indept/blp_squarem.jl")
include("$WKDIR/msl_indept/msl_est_iter.jl")


##
## 1. load choice data from MPS
##

LeftbhData = CSV.read("$WKDIR/mig_leftbh_enroll_fit.csv"; type = Float64)
LeftbhData = sort(LeftbhData, (:year, :hhtype, :ID, :cline, :city_alts))
LeftbhData[:cagey] = LeftbhData[:, :cagey] / 10
LeftbhData[:cageysq] = LeftbhData[:, :cagey].^2
LeftbhData[:nchild_lnhp] = LeftbhData[:, :lnhprice].* LeftbhData[:, :nchild]
YL = Vector{Float64}(LeftbhData[:, :chosen])
YM = Vector{Float64}(LeftbhData[:, :child_leftbh])

lnDataShare, Delta_init, lnW, lnP, XQJ_mig, wgt, sgwgt,
XT, XM, XL, XF, XQ, nalt, nind, dgvec = data_prepare(LeftbhData; trs = true)

##
## 2. get random draw
##

MigBootData = CSV.read("$WKDIR/mig_leftbh_indboot.csv")

# --- bootstrap random shock ---
nsim = 50
ndraw = nind * nsim
alpha = 0.12

# bootstrap observed preference vars.
DF_master = LeftbhData[LeftbhData[:, :chosen] .== 1,
            [:year, :ID, :cline, :child_leftbh,
            :highsch_f, :highsch_m]]
rename!(DF_master, :child_leftbh => :leftbh)
Znames = [:caring_study, :college_expect]
match_vars = [:leftbh, :highsch_f, :highsch_m]

Random.seed!(20190610);
ZSHK = map(1:nrow(DF_master)) do i
    bdf = filter(df -> df[match_vars] == DF_master[i, match_vars], MigBootData)
    Matrix{Float64}(boot_df(bdf, Znames; nboot = nsim))
end
ZSHK = vcat(ZSHK...)
ZSHK = copy(ZSHK')
USHK = dropdims(draw_shock(ndraw; dims = 1); dims = 2) # draw iid standard normal random shock

##
## 3. search for initial values
##

XTnames = [:highsch_f, :highsch_m, :age_f, :age_m, :han]
XLnames = [:cfemale, :nchild, :cagey, :cageysq]
XMnames = [:lndist, :cross_prov, :cross_regn, :lndist_crsprov, :lndist_crsregn,
		   :amenity_pca_flowdur, :migdist_flowdur, :amenity_pca_highsch_f,
		   :migdist_highsch_f, :amenity_pca_highsch_m, :migdist_highsch_m,
		   :amenity_pca_age_f, :migdist_age_f, :amenity_pca_age_m, :migdist_age_m]
XFnames = [:htreat, :migscore_fcvx_city, :lnhprice, :migscore_treat, :lnhp_treat,
		   :nchild_lnhp]
XQnames = [:cfemale, :cagey, :nchild]
XQJnames = [:tstu2_ratio, :pschool_per, :sschool_per]

# lftvar = [XTnames; XFnames; XLnames; XQJnames]
# lft_form = @eval @formula(child_leftbh ~ $(Meta.parse(join(lftvar, " + "))) + lnhinc_alts)
# lft_fit = glm(lft_form, view(LeftbhData, YL .== 1, :),
# 			  Binomial(), LogitLink(), wts = wgt)
# lft_init = coef(lft_fit)

##
## 4. Evaluate the likelihood
##

initset = load("$WKDIR/msl_indept_results/msl_indept_est_20190807.jld2")
Delta_init = initset["deltas"]
initparm = initset["coefx"]

nparm = size(XT, 1) + size(XL, 1) + size(XM, 1) + size(XF, 1) + 3 +
		size(XQ, 1) + 2*size(XQJ_mig, 1) + size(ZSHK, 1) + 1

initval = [initparm[1:31]; initparm[33]; -0.12; 0.2; -0.2; zeros(9); initparm[end-2:end-1]; -2]

parm_names = [:XT_cons; Symbol.("XT_" .* string.(XTnames)); Symbol.("XL_" .* string.(XLnames));
			  Symbol.("XM_" .* string.(XMnames)); :XF_cons; Symbol.("XF_" .* string.(XFnames));
			  :blft; :bw; :bitr; Symbol.("XQ_" .* string.(XQnames));
			  Symbol.("XQJ_Mig_" .* string.(XQJnames)); Symbol.("XQJ_Dif_" .* string.(XQJnames));
			  Znames; :lnsigu]

# --- iterative maximization ---
# 355388.7264999869
ret_msl = msl_est_iter(initval, lnDataShare, Delta_init, YL, YM, lnW, lnP, XQJ_mig,
			 			XT, XL, XM, XF, XQ, ZSHK, USHK, wgt, sgwgt, nind,
			 			nalt, nsim, dgvec)
save("$WKDIR/msl_indept_results/msl_indept_est_20190821.jld2", Dict("parm_names" => parm_names,
	 "coefx" => ret_msl[1], "deltas" => ret_msl[2], "loglik" => ret_msl[3]))


# --- evaluate log-likelihood ---
@time mig_leftbh_llk(initval, Delta_init, YL, YM, lnW, lnP, XQJ_mig,
			 		 XT, XL, XM, XF, XQ, ZSHK, USHK, wgt,
			   		 nind, nalt, nsim, dgvec, 0.12, 1)
# 444022.5091; 5.1s

@time mig_leftbh_llk_thread(initval, Delta_init, YL, YM, lnW, lnP, XQJ_mig,
			 		 		XT, XL, XM, XF, XQ, ZSHK, USHK, wgt,
			   		 		nind, nalt, nsim, dgvec, 0.12, 1)
# 444022.5091; 0.9s for 8 threads
#
# --- test for ForwardDiff ---
# using Optim, ForwardDiff
# llk_opt = parm -> mig_leftbh_llk(parm, Delta_init, YL, YM, lnW, lnP, XQJ_mig,
# 			 		 			 XT, XL, XM, XF, XQ, ZSHK, USHK, wgt,
# 			   					 nind, nalt, nsim, dgvec, 0.12, 1)
# @time llk_opt(initval)
# @time gr = ForwardDiff.gradient(llk_opt, initval)

ret_msl_final = load("$WKDIR/msl_indept_results/msl_indept_est_20190814.jld2")
coefx_final = ret_msl_final["coefx"]
delta_final = ret_msl_final["deltas"]
llk_opt_thread = parm -> mig_leftbh_llk_thread(parm, delta_final, YL, YM, lnW, lnP,
										XQJ_mig, XT, XL, XM, XF, XQ, ZSHK,
										USHK, wgt, nind, nalt, nsim, dgvec, 0.12, 1)
@time llk_opt_thread(coefx_final)
@time grad = ForwardDiff.gradient(llk_opt_thread, coefx_final)
@time hess = ForwardDiff.hessian(llk_opt_thread, coefx_final)
std = sqrt.(diag(inv(hess)))
tval = coefx_final ./ std
par_est_df = DataFrame(par_names = parm_names, coefx = coefx_final, std = std, tval = tval)
CSV.write("$WKDIR/msl_indept_results_20190814.csv", par_est_df)

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