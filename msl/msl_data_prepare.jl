using DataFrames
function data_prepare(df::AbstractDataFrame)
    ndt = nrow(df)
    city_alts = Vector{Int64}(df[:city_alts])
    nalt = length(unique(city_alts))
    ngvec = by(view(df, df[:chosen] .== 1, :), [:treat, :year], nrow, sort = true)[3]
    ngvec = cumsum(ngvec)
    nind = Int(ndt / nalt)

    lnW = Vector{Float64}(df[:lnhinc_alts])
    lnW = lnW .- mean(lnW, weights(Vector{Float64}(df[:w_l])))
    lnP = Vector{Float64}(df[:lnhprice])
    wgt = Vector{Float64}(df[df[:chosen] .== 1, :w_l])


    # --- initial value of delta ---
    cfreq = by(view(df, df[:chosen] .== 1, :), [:treat, :year, :city_alts], d -> sum(d[:w_l]))
    cfreq = reshape(sort(cfreq, [:treat, :year, :city_alts])[:x1], nalt, length(ngvec))
    cprop = cfreq ./ sum(cfreq, dims = 1)
    lnDataShare = log.(cprop)
    Delta_init = lnDataShare .- lnDataShare[1, :]'

    # --- household preference ---
    XTnames = [:highsch_f, :highsch_m, :age_f, :age_m, :han]
    XT = [ones(nind) convert(Array{Float64, 2}, df[df[:chosen] .== 1, XTnames])]

    # --- migration cost ---
    XMnames = names(df)[(end - 17):(end - 4)]
    XM = convert(Array{Float64, 2}, df[XMnames])

    # --- left-behind utility loss ---
    cageysq = df[:cagey].^2
    XLnames = [:cfemale, :nchild, :cagey]
    XL = [convert(Array{Float64, 2}, df[df[:chosen] .== 1, XLnames]) cageysq]

    # --- fixed cost ---
    # NOTE: the most critical part of the model!!
    nchild_lnmw = df[:lnmnw_city].* df[:nchild]
    nchild_lnhp = df[:lnhprice].* df[:nchild]
    XFnames = [:treat, :migscore_fcvx_city, :lnhprice, :migscore_treat, :lnhp_treat,
               :lnmnw_city]
    XF = [ones(ndt) convert(Array{Float64, 2}, df[XFnames]) nchild_lnmw nchild_lnhp]

    # --- cognitive ability ---
    # TODO: how to incorporate regional eduation quality?
    #       we have identification problem.

    XQnames = [:cfemale, :cagey, :nchild]
    XQ = [ones(nind) convert(Array{Float64, 2}, df[df[:chosen] .== 1, XQnames])]

    return(lnDataShare, Delta_init, lnW, lnP, wgt, XT, XM, XL, XF, XQ, nalt, nind)
end

#=
XQCnames = [:tstu1_ratio, :tstu2_ratio]
XQC = convert(Array{Float64, 2}, LeftbhData[XQCnames])
XQC = XQC .- mean(XQC, weights(Vector{Float64}(LeftbhData[:cw_age9])), 1)
XQC = SharedArray(copy(XQC'))
=#
