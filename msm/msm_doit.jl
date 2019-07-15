using FileIO, DataFrames, CSV, Statistics, StatsBase, LinearAlgebra, GLM
DTDIR = "E:/NutStore/Research/mig_leftbh_enrollment"
WKDIR = "E:/Dropbox/GitHub/Migration_Left_Behind_BLP"

include("$WKDIR/utility_funs/obj_helper_funs.jl")
include("$WKDIR/msm/data_prepare_msm.jl")
include("$WKDIR/msm/get_msm_moments.jl")
# include("$WKDIR/msl/msl_llk_loop.jl")
# include("$WKDIR/utility_funs/squarem_helper_funs.jl")
# include("$WKDIR/utility_funs/blp_squarem.jl")
# include("$WKDIR/msl/msl_est_iter.jl")

##
## 1. load choice data from MPS
##

LeftbhData = CSV.read("$DTDIR/mig_leftbh_enroll_fit.csv"; type = Float64)
LeftbhData[:cage9] = Float64.(LeftbhData[:cagey] .> 8)
LeftbhData[:cagey] = LeftbhData[:cagey] / 10
LeftbhData[:cageysq] = LeftbhData[:cagey].^2
LeftbhData[:nchild_lnmw] = LeftbhData[:lnmnw_city].* LeftbhData[:nchild]
LeftbhData[:nchild_lnhp] = LeftbhData[:lnhprice].* LeftbhData[:nchild]
LeftbhData[:hhtype] = LeftbhData[:hhtype] .- 1
YL = Vector{Float64}(LeftbhData[:chosen])
# YM = Vector{Float64}(LeftbhData[:child_leftbh])

XMnames = [:lndist; names(LeftbhData)[(end - 22):(end - 9)]]
XTnames = [:highsch_f, :highsch_m, :age_f, :age_m, :han]
XFnames = [:htreat, :migscore_fcvx_city, :lnhprice, :migscore_treat, :lnhp_treat,
		   :lnmnw_city, :nchild_lnmw, :nchild_lnhp]
XLnames = [:cfemale, :nchild, :cagey, :cageysq]
XQnames = [:cfemale, :cagey, :nchild, :highsch_f, :highsch_m, :pub_eduexp_fac]

lnDataShare, Delta_init, lnW, lnP,
lnQX, wgt, sgwgt, swgt9, XT, XM,
XL, XF, XQ, pr_lft,	pr_lft_alt,
nalt, nind, dgvec, htvec, dage9vec = data_prepare(LeftbhData, :lnhinc_alts, :pub_eduexp_fac,
	             						XTnames, XLnames, XFnames, XMnames,
										XQnames; trs = true)

##
## 2. get random draw
##

MigBootData = CSV.read("$DTDIR/mig_leftbh_indboot.csv")

# --- bootstrap random shock ---
nsim = 10
ndraw = nind * nsim
alpha = 0.12

# bootstrap observed preference vars.
DF_master = LeftbhData[LeftbhData[:chosen] .== 1,
            [:year, :ID, :cline, :child_leftbh,
            :highsch_f, :highsch_m]]
rename!(DF_master, :child_leftbh => :leftbh)
zshk_vars = [:caring_study, :college_expect]
match_vars = [:leftbh, :highsch_f, :highsch_m]

Random.seed!(20190610);
ZSHK = map(1:nrow(DF_master)) do i
    bdf = filter(df -> df[match_vars] == DF_master[i, match_vars], MigBootData)
    Matrix{Float64}(boot_df(bdf, zshk_vars; nboot = nsim))
end
ZSHK = vcat(ZSHK...)
ZSHK = copy(ZSHK')

USHK = draw_shock(ndraw; dims = 2) # draw iid standard normal random shock
QSHK = USHK[:, 2]
USHK = USHK[:, 1]

##
## 3. Calculate Data Moments
##
include("calc_data_moments.jl")
xq_vars = [:female, :cage, :nchild, :highsch_f, :highsch_m, :pub_eduexp_fac]

# --- the data moments ---
leftbh_mnt = data_moments_leftbh(view(LeftbhData, LeftbhData[:chosen] .== 1, :),
							 	 :lnhinc_alts, :cage9, XTnames, XLnames, XFnames, XMnames)
zcog_mnt = data_moments_zcog(MigBootData, :clnhinc, :cognitive, :pub_eduexp_fac,
						     zshk_vars, xq_vars)
data_mnts_all = vcat(leftbh_mnt, zcog_mnt)

# --- bootstrap to calculate the variance of data moments ---
var_leftbh_mnt = mnt_var_leftbh(view(LeftbhData, LeftbhData[:chosen] .== 1, :),
							 	 :lnhinc_alts, :cage9, XTnames, XLnames, XFnames,
								 XMnames, length(leftbh_mnt))
var_zcog_mnt = mnt_var_zcog(MigBootData, :clnhinc, :cognitive, :pub_eduexp_fac,
						     zshk_vars, xq_vars, length(zcog_mnt))
dwt = 1.0 ./ vcat(var_leftbh_mnt, var_zcog_mnt)

##
## 4. search for initial values
##

lftvar = [XTnames; XFnames; XLnames]
lft_form = @eval @formula(child_leftbh ~ $(Meta.parse(join(lftvar, " + "))))
lft_fit = glm(lft_form, view(LeftbhData, YL .== 1, :),
			  Binomial(), LogitLink(), wts = wgt)
lft_init = coef(lft_fit)

##
## 5. Evaluate the moment conditions
##

nparm = size(XT, 1) + size(XL, 1) + size(XM, 1) + size(XF, 1) + 3 +
		size(XQ, 1) + size(ZSHK, 1) + 3
xt_init = lft_init[1:6]
xf_init = lft_init[7:14]
xl_init = lft_init[15:end]
xm_init = zeros(size(XM, 1))
xq_init = zeros(size(XQ, 1))
bw = 0.194
blft = -0.148
bitr = -0.167
initval = [xt_init; 0.0; xl_init; xm_init; 0.0; xf_init; blft; bw; bitr;
		   xq_init; zeros(2); -1.5; 0.0; -1.0]

# --- iterative maximization ---
# msl_est_iter(initval, lnDataShare, Delta_init, YL, YM, lnW, lnP, XT, XL, XM, XF,
# 			 XQ, ZSt, USHK, wgt, sgwgt, nind, nalt, nsim, dgvec; biter = 3)

# --- evaluate moment conditions ---
# 6.5s
@time mnt_serial = get_moments(initval, alpha, lnW, lnP, lnQX, XT, XL, XM, XF, XQ, ZSHK,
				  USHK, QSHK, pr_lft, pr_lft_alt, Delta_init, dgvec, htvec, dage9vec, wgt,
				  sgwgt, swgt9, nind, nalt, nsim; xdim = 1)

# 1.2s for 8 threads
@time mnt_thread = get_moments_thread(initval, alpha, lnW, lnP, lnQX, XT, XL, XM, XF, XQ, ZSHK,
				  USHK, QSHK, pr_lft, pr_lft_alt, Delta_init, dgvec, htvec, dage9vec, wgt,
				  sgwgt, swgt9, nind, nalt, nsim; xdim = 1)

# # --- test for ForwardDiff ---
# using Optim, ForwardDiff
# msm_opt = parm -> get_moments(parm, alpha, lnW, lnP, lnQX, XT, XL, XM, XF, XQ,
# 				  	ZSHK, USHK, QSHK, pr_lft, pr_lft_alt, Delta_init, dgvec, htvec,
# 				  	wgt, sgwgt, nind, nalt, nsim; xdim = 1)
# @time msm_opt(initval)
# @time gr = ForwardDiff.gradient(msm_opt, initval)
#
# msm_opt_thread = parm -> get_moments_thread(parm, alpha, lnW, lnP, lnQX, XT, XL,
# 				  	XM, XF, XQ, ZSHK, USHK, QSHK, pr_lft, pr_lft_alt, Delta_init,
# 				  	dgvec, htvec, wgt, sgwgt, nind, nalt, nsim; xdim = 1)
# @time msm_opt_thread(initval)
# @time gr = ForwardDiff.gradient(msm_opt_thread, initval)

#
# # --- predicted location choice probabilities ---
# ngrp = length(sgwgt)
# mktshare = zeros(nalt, ngrp)
# @time locpr_serial!(mktshare, initval, Delta_init, lnW, lnP, XT, XL, XM,
# 			   		XF, XQ, ZSt, USHK, wgt, sgwgt, nind, nalt, nsim, dgvec)
# @time locpr_thread!(mktshare, initval, Delta_init, lnW, lnP, XT, XL, XM,
# 			   		XF, XQ, ZSt, USHK, wgt, sgwgt, nind, nalt, nsim, dgvec)
#
# # --- BLP Contraction Mapping ---
# delta_fpt = zeros(nalt, ngrp)
# delta_new = zeros(nalt, ngrp)
# delta_old = copy(Delta_init)
# delta_q1 = zeros(nalt, ngrp)
# delta_q2 = zeros(nalt, ngrp)
# @time fpt_squarem!(delta_fpt, delta_new, delta_old, delta_q1, delta_q2, lnDataShare,
# 				   initval, lnW, lnP, XT, XL, XM, XF, XQ, ZSt, USHK, wgt, sgwgt,
# 				   nind, nalt, nsim, dgvec)
