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
