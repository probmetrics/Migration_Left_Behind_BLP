using DataFrames
function data_prepare(df::AbstractDataFrame; trs::Bool = false)
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
    XLnames = [:cfemale, :nchild, :cagey, :cageysq]
    XL = convert(Array{Float64, 2}, df[df[:chosen] .== 1, XLnames])

    # --- fixed cost ---
    # NOTE: the most critical part of the model!!
    XFnames = [:treat, :migscore_fcvx_city, :lnhprice, :migscore_treat, :lnhp_treat,
               :lnmnw_city, :nchild_lnmw, :nchild_lnhp]
    XF = [ones(ndt) convert(Array{Float64, 2}, df[XFnames])]

    # --- cognitive ability ---
    # TODO: how to incorporate regional eduation quality?
    #       we have identification problem.

    XQnames = [:cfemale, :cagey, :nchild]
    XQ = convert(Array{Float64, 2}, df[XQnames])

    if trs == true
        XT = copy(XT')
        XL = copy(XL')
        XM = copy(XM')
        XF = copy(XF')
        XQ = copy(XQ')
    end

    return(lnDataShare, Delta_init, lnW, lnP, wgt, XT, XM, XL, XF, XQ, nalt, nind, ngvec)
end

#=
XQCnames = [:tstu1_ratio, :tstu2_ratio]
XQC = convert(Array{Float64, 2}, LeftbhData[XQCnames])
XQC = XQC .- mean(XQC, weights(Vector{Float64}(LeftbhData[:cw_age9])), 1)
XQC = SharedArray(copy(XQC'))
=#
