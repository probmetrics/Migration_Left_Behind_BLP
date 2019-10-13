using FileIO, DataFrames, CSV, Statistics, StatsBase, LinearAlgebra, GLM
DTDIR = "E:/NutStore/Research/mig_leftbh_enrollment"
WKDIR = "E:/Dropbox/GitHub/Migration_Left_Behind_BLP"
# WKDIR = "/Users/probmetrics/Dropbox/GitHub/Migration_Left_Behind_BLP"
# DTDIR = "/Users/probmetrics/NutStore/Research/mig_leftbh_enrollment"

include("$WKDIR/slii/obj_helper_funs.jl")
include("$WKDIR/slii/data_prepare_slii.jl")
# include("$WKDIR/slii/get_msm_moments_rev2.jl")
include("$WKDIR/slii/slii_llk_loop.jl")
include("$WKDIR/slii/squarem_helper_funs.jl")
include("$WKDIR/slii/blp_squarem.jl")
include("$WKDIR/slii/slii_est_iter.jl")

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
# YL = Vector{Float64}(LeftbhData[:chosen])
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

MigBootData = CSV.read("$DTDIR/mig_leftbh_indboot.csv")
lnq_form = @eval @formula(cog_adj ~ leftbh + clnhinc + $(Meta.parse(join(XQnames, " + "))) +
						  tstu2_ratio + sschool_per + leftbh&clnhinc +
						  leftbh&tstu2_ratio + leftbh&sschool_per)
lnq_fit = lm(lnq_form, MigBootData)
lnq_init = coef(lnq_fit)
lnqrsd = sqrt(deviance(lnq_fit) / (dof_residual(lnq_fit) + length(lnq_init)))

XQparm = [lnq_init[1]; lnq_init[4:(length(XQnames) + 3)]]
XQJMparm = lnq_init[(length(XQnames) + 4):(length(XQnames) + 3 + length(XQJMnames))]
XQJMparm = [XQJMparm; lnq_init[(end - length(XQJMnames) + 1):end]]
LWIparm = [lnq_init[2:3]; lnq_init[end - length(XQJMnames)]]

lnDataShare, Delta_init, YL, YM, lnW, lnP, lnQbar,
XQJ_mig, wgt, sgwgt, shtwgt, swgt9, XT, XM, XL,
XF, XQ, nalt, nind, dgvec, htvec, dage9vec =
	data_prepare(LeftbhData, :lnhinc_alts, XQJMnames, XTnames, XLnames, XFnames,
	             XMnames, XQnames, XQparm, XQJMparm, LWIparm; trs = true)

##
## 2. get random draw
##

# --- bootstrap random shock ---
nsim = 20
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
## 4. search for initial values
##

# lftvar = [XTnames; XFnames; XLnames; XQJMnames]
# lft_form = @eval @formula(child_leftbh ~ $(Meta.parse(join(lftvar, " + "))) + lnhinc_alts)
# lft_fit = glm(lft_form, view(LeftbhData, YL .== 1, :),
# 			  Binomial(), LogitLink(), wts = wgt)
# lft_init = coef(lft_fit)

initset = load("$DTDIR/msl_indept_results/msl_indept_est_20190807.jld2")
initpar = initset["coefx"]
initdel = initset["deltas"]


##
## 5. Evaluate the moment conditions
##

nparm = size(XT, 1) + size(XL, 1) + size(XM, 1) + size(XF, 1) + 3 +
		size(XQ, 1) + 2*size(XQJ_mig, 1) + size(ZSHK, 1) + 2

# xt_init = [3.0; lft_init[2:size(XT, 1)]]
# xf_init = lft_init[(size(XT, 1) + 1):(size(XT, 1) + size(XF, 1) - 1)]
# xl_init = lft_init[(size(XT, 1) + size(XF, 1)):(size(XT, 1) + size(XF, 1) + size(XL, 1) - 1)]
# xqj_dif_init = lft_init[(end-size(XQJ_mig, 1)):(end-1)]
# xm_init = zeros(size(XM, 1))
# xq_init = zeros(size(XQ, 1))

xq_init = [lnq_init[1]; lnq_init[4:end-5]]
bw = 0.19
blft = -0.124
bitr = -0.197
initval = [initpar[1:31]; initpar[33]; blft; bw; bitr; xq_init; XQJMparm;
			initpar[end-2:end-1]; -2.5; log(lnqrsd)]
# initval = [xt_init; xl_init; xm_init; 0.0; xf_init; blft; bw; bitr;
# 		   xq_init; zeros(length(xqj_dif_init)); xqj_dif_init; zeros(2);
# 		   -2.5; 0.0; -1.0]

# --- iterative maximization ---
ret_msl = slii_est_iter(initval, lnDataShare, initdel, YL, YM, lnW, lnP,
			 			lnQbar, XQJ_mig, XT, XL, XM, XF, XQ, ZSHK, USHK, QSHK,
						wgt, sgwgt, nind, nalt, nsim, dgvec, dage9vec; biter = 1)

# --- evaluate log-likelihood ---
@time mig_leftbh_llk(initval, initdel, YL, YM, lnW, lnP, lnQbar, XQJ_mig,
			 		 XT, XL, XM, XF, XQ, ZSHK, USHK, QSHK, wgt, nind, nalt,
			   		 nsim, dgvec, dage9vec, lnqrsd, 0.12, 1)
# 444022.5091; 5.1s

@time mig_leftbh_llk_thread(initval, initdel, YL, YM, lnW, lnP, lnQbar,
			 		 	XQJ_mig, XT, XL, XM, XF, XQ, ZSHK, USHK, QSHK, wgt,
			   		 	nind, nalt, nsim, dgvec, dage9vec, lnqrsd, 0.12, 1)
# 444022.5091; 0.9s for 8 threads
#
# --- test for ForwardDiff ---
using Optim, ForwardDiff
# llk_opt = parm -> mig_leftbh_llk(parm, Delta_init, YL, YM, lnW, lnP, XQJ_mig,
# 			 		 			 XT, XL, XM, XF, XQ, ZSHK, USHK, wgt,
# 			   					 nind, nalt, nsim, dgvec, 0.12, 1)
# @time llk_opt(initval)
# @time gr = ForwardDiff.gradient(llk_opt, initval)

llk_opt_thread = parm -> mig_leftbh_llk_thread(parm, initdel, YL, YM, lnW, lnP,
										lnQbar, XQJ_mig, XT, XL, XM, XF, XQ, ZSHK,
										USHK, QSHK, wgt, nind, nalt, nsim, dgvec,
										dage9vec, lnqrsd, 0.12, 1)
@time llk_opt_thread(initval)
@time gr = ForwardDiff.gradient(llk_opt_thread, initval)
