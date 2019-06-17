using FileIO, DataFrames, CSV, Statistics, StatsBase
using SharedArrays, LinearAlgebra
include("msl_helper_funs.jl")
include("msl_data_prepare.jl")
# include("msl_llk.jl")
include("msl_llk_loop.jl")

DTDIR = "E:/NutStore/Research/mig_leftbh_enrollment"

##
## 1. load choice data from MPS
##

# LeftbhData = readcsv("$DTDIR/mig_leftbh_enroll_fit.csv")
LeftbhData = CSV.read("$DTDIR/mig_leftbh_enroll_fit.csv"; type = Float64)
LeftbhData[:cagey] = LeftbhData[:cagey] / 10
LeftbhData[:cageysq] = LeftbhData[:cagey] .^ 2 / 100
LeftbhData[:nchild_lnmw] = LeftbhData[:lnmnw_city].* LeftbhData[:nchild]
LeftbhData[:nchild_lnhp] = LeftbhData[:lnhprice].* LeftbhData[:nchild]

lnDataShare, Delta_init, lnW, lnP, wgt,
XT, XM, XL, XF, XQ, nalt, nind, ngvec = data_prepare(LeftbhData)
YL = Vector{Float64}(LeftbhData[:chosen])
YM = Vector{Float64}(LeftbhData[:child_leftbh])
dgvec = [locate_gidx(i, ngvec) for i = 1:nind]

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

USHK = dropdims(draw_shock(ndraw; dims = 1); dims = 2) # draw iid standard normal random shock

##
## 3. search for initial values
##
using GLM

XTnames = [:highsch_f, :highsch_m, :age_f, :age_m, :han]
XFnames = [:treat, :migscore_fcvx_city, :lnhprice, :migscore_treat, :lnhp_treat,
		   :lnmnw_city, :nchild_lnmw, :nchild_lnhp]
XLnames = [:cfemale, :nchild, :cagey, :cageysq]
lftvar = [XTnames; XFnames; XLnames]
lft_form = @eval @formula(child_leftbh ~ $(Meta.parse(join(lftvar, " + "))))
lft_fit = glm(lft_form, view(LeftbhData, YL .== 1, :),
			  Binomial(), LogitLink(), wts = wgt)
lft_init = coef(lft_fit)

##
## 4. Evaluate the likelihood
##

nparm = size(XT, 2) + size(XL, 2) + size(XM, 2) + size(XF, 2) + 3 +
		size(XQ, 2) + size(ZSHK, 2) + 1
xt_init = lft_init[1:6]
xf_init = lft_init[7:14]
xl_init = lft_init[15:end]
xm_init = zeros(size(XM, 2))
xq_init = zeros(size(XQ, 2))
bw = 0.194
blft = -0.148
bitr = -0.167
initval = [xt_init; xl_init; xm_init; 0; xf_init; blft; bw; bitr; xq_init; zeros(2); -1.5]

XTt = copy(XT')
XLt = copy(XL')
XMt = copy(XM')
XFt = copy(XF')
XQt = copy(XQ')
ZSt = copy(ZSHK')
@time mig_leftbh_llk(initval, Delta_init, YL, YM, lnW, lnP, XTt, XLt, XMt, XFt, XQt,
			   ZSt, USHK, wgt, nind, nalt, nsim, dgvec)
# 430352.6129; 6s

using Optim, ForwardDiff
llk_opt = parm -> mig_leftbh_llk(parm, Delta_init, YL, YM, lnW, lnP, XTt, XLt, XMt, XFt, XQt,
			   					 ZSt, USHK, wgt, nind, nalt, nsim, dgvec)
@time llk_opt(initval)
@time gr = ForwardDiff.gradient(llk_opt, initval)
