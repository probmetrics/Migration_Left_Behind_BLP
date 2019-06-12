using FileIO, DataFrames, CSV, Statistics, StatsBase
using SharedArrays, LinearAlgebra
include("msl_helper_funs.jl")
include("msl_data_prepare.jl")
DTDIR = "E:/NutStore/Research/mig_leftbh_enrollment"

##
## 1. load choice data from MPS
##

# LeftbhData = readcsv("$DTDIR/mig_leftbh_enroll_fit.csv")
LeftbhData = CSV.read("$DTDIR/mig_leftbh_enroll_fit.csv"; type = Float64)
LeftbhData[:cagey] = LeftbhData[:cagey] / 10

lnDataShare, Delta_init, lnW, lnP, wgt,
XT, XM, XL, XF, XQ, nalt, nind = data_prepare(LeftbhData)
yl = Vector{Float64}(LeftbhData[:chosen])
ym = Vector{Float64}(LeftbhData[:child_leftbh])

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
XFnames = [:treat, :migscore_fcvx_city, :lnhprice, :migscore_treat, :lnhp_treat, :lnmnw_city]
XLnames = [:cfemale, :nchild, :cagey]
lftvar = [XTnames; XFnames; XLnames]
lft_form = @eval @formula(child_leftbh ~ $(Meta.parse(join(lftvar, " + "))))
lft_fit = glm(lft_form, view(LeftbhData, yl .== 1, :),
			  Binomial(), LogitLink(), wts = wgt)
lft_init = coef(lft_fit)

##
## 4. Evaluate the likelihood
##

nparm = size(XT, 2) + size(XL, 2) + size(XM, 2) + size(XF, 2) + 3 +
		size(XQ, 2) + size(ZSHK, 2) + 1
xt_init = lft_init[1:6]
xf_init = lft_init[7:12]
xl_init = lft[13:end]
xm_init = zeros(size(XM, 2))
xq_init = zeros(size(XQ, 2))
bw = 0.194
blft = -0.148
bitq = -0.167
initval = [xt_init; xl_init; xm_init; 0; xf_init; blft; bw; bitr; xq_init; zeros(2); -1.5]
mig_leftbh_llk(initval, Delta, yl, ym, lnW, lnP, XT, XL, XM, XF, XQ,
			   ZSHK, USHK, wgt, nind, nalt, nsim, ngvec)
