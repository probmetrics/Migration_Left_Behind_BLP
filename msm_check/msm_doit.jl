using FileIO, DataFrames, CSV, Statistics, StatsBase, LinearAlgebra, GLM
DTDIR = "E:/NutStore/Research/mig_leftbh_enrollment"
WKDIR = "E:/Dropbox/GitHub/Migration_Left_Behind_BLP"
# WKDIR = "/Users/probmetrics/Dropbox/GitHub/Migration_Left_Behind_BLP"
# DTDIR = "/Users/probmetrics/NutStore/Research/mig_leftbh_enrollment"

include("$WKDIR/utility_funs/obj_helper_funs.jl")
include("$WKDIR/msm_check/data_prepare_msm.jl")
include("$WKDIR/msm_check/get_msm_moments_rev2.jl")
# include("$WKDIR/msl/msl_llk_loop.jl")
include("$WKDIR/utility_funs/squarem_helper_funs.jl")
include("$WKDIR/utility_funs/blp_squarem.jl")
include("$WKDIR/msm_check/msm_est_iter.jl")

##
## 1. load choice data from MPS
##

LeftbhData = CSV.read("$DTDIR/mig_leftbh_enroll_fit.csv"; type = Float64)
# LeftbhData = sort(LeftbhData, (:year, :hhtype, :ID, :cline, :city_alts))
LeftbhData[:cage9] = Float64.(LeftbhData[:cagey] .> 8)
LeftbhData[:cagey] = LeftbhData[:cagey] / 10
LeftbhData[:cageysq] = LeftbhData[:cagey].^2
LeftbhData[:nchild_lnhp] = LeftbhData[:lnhprice].* LeftbhData[:nchild]
LeftbhData[:hhtype] = LeftbhData[:hhtype] .- 1
YL = Vector{Float64}(LeftbhData[:chosen])
# YM = Vector{Float64}(LeftbhData[:child_leftbh])

XMnames = [:lndist, :cross_prov, :cross_regn, :lndist_crsprov, :lndist_crsregn,
           :amenity_pca_flowdur, :migdist_flowdur, :amenity_pca_highsch_f,
           :migdist_highsch_f, :amenity_pca_highsch_m, :migdist_highsch_m,
           :amenity_pca_age_f, :migdist_age_f, :amenity_pca_age_m, :migdist_age_m]
XTnames = [:highsch_f, :highsch_m, :age_f, :age_m, :han]
XFnames = [:htreat, :migscore_fcvx_city, :lnhprice, :migscore_treat, :lnhp_treat,
		   :nchild_lnhp]
XLnames = [:cfemale, :nchild, :cagey, :cageysq]
XQnames = [:cfemale, :cagey, :nchild, :highsch_f, :highsch_m, :age_f]
XQJMnames = [:tstu2_ratio, :sschool_per]

lnDataShare, Delta_init, lnW, lnP,
XQJ_mig, wgt, sgwgt, shtwgt, swgt9,
XT, XM, XL, XF, XQ, pr_lft,	pr_lft_alt,
nalt, nind, dgvec, htvec, dage9vec =
	data_prepare(LeftbhData, :lnhinc_alts, XQJMnames, XTnames,
	             XLnames, XFnames, XMnames, XQnames; trs = true)
XFW = vcat(XF, lnW')

##
## 2. get random draw
##

MigBootData = CSV.read("$DTDIR/mig_leftbh_indboot.csv")

# --- bootstrap random shock ---
nsim = 20
ndraw = nind * nsim
alpha = 0.12

# bootstrap observed preference vars.
# DF_master = LeftbhData[LeftbhData[:chosen] .== 1,
#             [:year, :ID, :cline, :child_leftbh,
#             :highsch_f, :highsch_m]]
# rename!(DF_master, :child_leftbh => :leftbh)
zshk_vars = [:caring_study, :college_expect]

Random.seed!(20190610);
ZSHK = Matrix{Float64}(boot_df(MigBootData, zshk_vars; nboot = ndraw))
ZSHK = copy(ZSHK')

USHK = draw_shock(ndraw; dims = 2) # draw iid standard normal random shock
QSHK = USHK[:, 2]
USHK = USHK[:, 1]

##
## 3. Calculate Data Moments
##
include("$WKDIR/msm_check/calc_data_moments_rev2.jl")

# --- the data moments ---
leftbh_mnt = data_moments_leftbh(view(LeftbhData, LeftbhData[:chosen] .== 1, :),
							 	 :lnhinc_alts, :cage9, XTnames, XLnames, XFnames,
								 XMnames, XQJMnames)
zcog_mnt = data_moments_zcog(MigBootData, :clnhinc, :cog_adj, XQJMnames,
						     zshk_vars, XQnames)
data_mnts_all = vcat(leftbh_mnt, zcog_mnt)

# --- bootstrap to calculate the variance of data moments ---
var_leftbh_mnt = mnt_var_leftbh(view(LeftbhData, LeftbhData[:chosen] .== 1, :),
							 	 :lnhinc_alts, :cage9, XTnames, XLnames, XFnames,
								 XMnames, XQJMnames, length(leftbh_mnt))
var_zcog_mnt = mnt_var_zcog(MigBootData, :clnhinc, :cog_adj, XQJMnames,
						    zshk_vars, XQnames, length(zcog_mnt))
dwt = 1.0 ./ vcat(var_leftbh_mnt, var_zcog_mnt)

##
## 4. search for initial values
##

lftvar = [XTnames; XFnames; XLnames; XQJMnames]
lft_form = @eval @formula(child_leftbh ~ $(Meta.parse(join(lftvar, " + "))) + lnhinc_alts)
lft_fit = glm(lft_form, view(LeftbhData, YL .== 1, :),
			  Binomial(), LogitLink(), wts = wgt)
lft_init = coef(lft_fit)

initset = load("$DTDIR/msl_indept_results/msl_indept_est_20190807.jld2")
initpar = initset["coefx"]
initdel = initset["deltas"]

lnq_form = @eval @formula(cog_adj ~ leftbh + clnhinc + $(Meta.parse(join(XQnames, " + "))) +
						  tstu2_ratio + sschool_per + leftbh&clnhinc + leftbh&tstu2_ratio +
						  leftbh&sschool_per)
lnq_fit = lm(lnq_form, MigBootData)
lnq_init = coef(lnq_fit)

##
## 5. Evaluate the moment conditions
##

nparm = size(XT, 1) + size(XL, 1) + size(XM, 1) + size(XFW, 1) + 3 +
		size(XQ, 1) + 2*size(XQJ_mig, 1) + size(ZSHK, 1) + 1

xt_init = lft_init[2:size(XT, 1)]
xl_init = lft_init[(size(XT, 1) + size(XF, 1) + 1):(size(XT, 1) + size(XF, 1) + size(XL, 1) - 1)]
xf_init = [0.0; lft_init[(size(XT, 1) + 2):(size(XT, 1) + size(XF, 1))]; -0.2]

# xqj_dif_init = lft_init[(end-size(XQJ_mig, 1)):(end-1)]
xm_init = initpar[11:25]
# xq_init = zeros(size(XQ, 1))

xq_init = [lnq_init[1]; lnq_init[4:end-5]]
xqj_mig_init = lnq_init[end-4:end-3]
xqj_dif_init = lnq_init[end-1:end]
bw = 0.19
blft = -0.124
bitr = -0.197
# initval = [initpar[1:4]; initpar[6:10]; 0.0; initpar[11:31]; initpar[33];
# 			blft; bw; bitr; xq_init; xqj_mig_init; xqj_dif_init;
# 			initpar[end-2:end-1]; -1.0]
initval = [xt_init; 4.2; -xl_init; xm_init; xf_init; -0.2; blft; bw; bitr;
		   xq_init; zeros(length(xqj_dif_init)); xqj_dif_init; zeros(2);
		   -1.0]

parm_names = [Symbol.("XT_" .* string.(XTnames)); :XL_cons; Symbol.("XL_" .* string.(XLnames));
			  Symbol.("XM_" .* string.(XMnames)); :XF_cons; Symbol.("XF_" .* string.(XFnames));
			  :XF_lnW; :blft; :bw; :bitr; :XQ_cons; Symbol.("XQ_" .* string.(XQnames));
			  Symbol.("XQJ_Mig_" .* string.(XQJMnames)); Symbol.("XQJ_Dif_" .* string.(XQJMnames));
			  zshk_vars; :lnsigq]

# --- iterative maximization ---
# msm_est_iter(initval, data_mnts, dwt, lnDataShare, alpha, lnW, lnP,
#			   XQJ_mig, XT, XL, XM, XF, XQ, ZSHK, USHK, QSHK,
# 			   pr_lft, pr_lft_alt, Delta_init, dgvec, htvec, dage9vec,
#  			   wgt, sgwgt, nind, nalt, nsim, dgvec; biter = 3)

# --- evaluate moment conditions ---
# 6s
@time mnt_serial = get_moments(initval, alpha, lnW, lnP, XQJ_mig, XT, XL,
				  		XM, XFW, XQ, ZSHK, USHK, QSHK, pr_lft, initdel,
				  		dgvec, htvec, dage9vec, wgt, shtwgt, swgt9, nind,
						nalt, nsim; xdim = 1)

# 1.2s for 8 threads
@time mnt_thread = get_moments_thread(initval, alpha, lnW, lnP, XQJ_mig,
						XT, XL, XM, XFW, XQ, ZSHK, USHK, QSHK, pr_lft,
				  		initdel, dgvec, htvec, dage9vec, wgt, shtwgt, swgt9,
				  		nind, nalt, nsim; xdim = 1)

dwt_iden = ones(length(data_mnts_all))
@time msm_obj(initval, data_mnts_all, dwt_iden, alpha, lnW, lnP, XQJ_mig,
			  XT, XL, XM, XF, XQ, ZSHK, USHK, QSHK, pr_lft,
			  initdel, dgvec, htvec, dage9vec, wgt, shtwgt, swgt9,
			  nind, nalt, nsim; xdim = 1)

ret_msm = msm_est_iter(initval, data_mnts_all, dwt,	lnDataShare, alpha, lnW,
						lnP, XQJ_mig, XT, XL, XM, XF, XQ, ZSHK, USHK, QSHK,
						pr_lft, initdel, dgvec, htvec, dage9vec, wgt, sgwgt,
				 		shtwgt, swgt9, nind, nalt, nsim; biter = 2)

# # --- test for ForwardDiff ---
# using Optim, ForwardDiff, Calculus
# msm_opt = parm -> msm_obj(parm, data_mnts_all, dwt, alpha, lnW, lnP, XQJ_mig,
# 			  			  XT, XL, XM, XF, XQ, ZSHK, USHK, QSHK, pr_lft,
# 			  			  Delta_init, dgvec, htvec, dage9vec, wgt, sgwgt, swgt9,
# 			  			  nind, nalt, nsim; xdim = 1)
# @time msm_opt(initval)
# @time gr = ForwardDiff.gradient(msm_opt, initval)
