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
            [:hhtype, :year, :ID, :cline,
            :child_leftbh, :highsch_f, :highsch_m]]
znames = [:leftbh, :highsch_f, :highsch_m, :caring_study, :college_expect]
ZDF = boot_obsx(MigBootData, znames, nboot = ndraw)
rename!(ZDF, :leftbh => :child_leftbh)
sort!(ZDF, (:child_leftbh, :highsch_f, :highsch_m))

ZDF = join(DF_master, ZDF, on = [:child_leftbh, :highsch_f, :highsch_m])
sort!(ZDF, (:hhtype, :year, :ID, :cline))
ZSHK = ZSHK .- mean(ZSHK, dims = 1)

# udraws = halton(ndraw; dim = 2, normal = true)
USHK = Vector{Float64}(draw_shock(ndraw; dims = 1)) # draw iid standard normal random shock
