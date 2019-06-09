using FileIO, DataFrames, CSV, Statistics, StatsBase
using SharedArrays, LinearAlgebra
include("msl_helper_funs.jl")
include("msl_data_prepare.jl")
DTDIR = "E:/NutStore/Research/mig_leftbh_enrollment"

##
## 1. load choice data from MPS
##

LeftbhData = CSV.read("$DTDIR/mig_leftbh_enroll_fit.csv"; type = Float64)
#LeftbhData = readcsv("$DTDIR/mig_leftbh_enroll_fit.csv")
LeftbhData[:cagey] = LeftbhData[:cagey] / 10

lnDataShare, Delta_init, lnW, lnP, wgt,
XT, XM, XL, XF, XQ, nalt, nind = data_prepare(LeftbhData)

##
## 2. get random draw
##

MigBootData = CSV.read("$DTDIR/mig_leftbh_indboot.csv")
# --- bootstrap random shock ---
nsim = 10
ndraw = nind * nsim
alpha = 0.12

# bootstrap observed preference vars.
znames = [:caring_study, :college_expect]
ZSHK = boot_obsx(MigBootData, znames, nboot = ndraw)
ZSHK = ZSHK .- mean(ZSHK, dims = 1)

# udraws = halton(ndraw; dim = 2, normal = true)
USHK = Vector{Float64}(draw_shock(ndraw; dims = 1)) # draw iid standard normal random shock
