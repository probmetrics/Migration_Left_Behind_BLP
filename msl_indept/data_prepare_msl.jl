using DataFrames
function data_prepare(df::AbstractDataFrame, XQJMnames::AbstractVector{Symbol},
                      XTnames::AbstractVector{Symbol}, XLnames::AbstractVector{Symbol},
                      XFnames::AbstractVector{Symbol}, XMnames::AbstractVector{Symbol},
                      XQnames::AbstractVector{Symbol}; trs::Bool = false)
    ndt = nrow(df)
    city_alts = Vector{Int64}(df[:, :city_alts])
    nalt = length(unique(city_alts))
    ngvec = by(view(df, df[:, :chosen] .== 1, :), :year, nrow, sort = true)[:, 2]
    ngvec = cumsum(ngvec)
    nind = Int(ndt / nalt)
    dgvec = [locate_gidx(i, ngvec) for i = 1:nind]

    lnW = Vector{Float64}(df[:, :lnhinc_alts])
    wgtvec = Vector{Float64}(df[:w_l])
    cage9d = Vector{Int}(df[:cage9])
    lnW = lnW .- mean(view(lnW, cage9d .== 1), weights(view(wgtvec, cage9d .== 1)))

    lnP = Vector{Float64}(df[:, :lnhprice])

    wgt = Vector{Float64}(df[df[:, :chosen] .== 1, :w_l])
    sgwgt = countmap(dgvec, weights(wgt))
    sgwgt = [sgwgt[i] for i = 1:length(sgwgt)]

    # --- initial value of delta ---
    cfreq = by(view(df, df[:, :chosen] .== 1, :), [:year, :city_alts], d -> sum(d[:, :w_l]))
    cfreq = reshape(sort(cfreq, [:year, :city_alts])[:, :x1], nalt, length(ngvec))
    cprop = cfreq ./ sum(cfreq, dims = 1)
    lnDataShare = log.(cprop)
    Delta_init = lnDataShare .- lnDataShare[1, :]'

    # --- household preference ---
    XT = convert(Array{Float64, 2}, df[df[:, :chosen] .== 1, XTnames])

    # --- migration cost ---
    XM = convert(Array{Float64, 2}, df[:, XMnames])

    # --- left-behind utility loss ---
    XL = [ones(nind) convert(Array{Float64, 2}, df[df[:, :chosen] .== 1, XLnames])]

    # --- fixed cost ---
    # NOTE: the most critical part of the model!!
    XF = [ones(ndt) convert(Array{Float64, 2}, df[:, XFnames])]

    # --- cognitive ability ---
    # TODO: how to incorporate regional eduation quality?
    #       we have identification problem.

    XQ = [ones(nind) convert(Array{Float64, 2}, df[df[:, :chosen] .== 1, XQnames])]

    XQJ_mig = convert(Array{Float64, 2}, df[:, XQJMnames])
	XQJ_mig = XQJ_mig .- mean(view(XQJ_mig, cage9d .== 1, :), weights(view(wgtvec, cage9d .== 1)), dims = 1)

    if trs == true
        XT = copy(XT')
        XL = copy(XL')
        XM = copy(XM')
        XF = copy(XF')
        XQ = copy(XQ')
        XQJ_mig = copy(XQJ_mig')
    end

    return(lnDataShare, Delta_init, lnW, lnP, XQJ_mig,
           wgt, sgwgt, XT, XM, XL, XF, XQ, nalt, nind, dgvec)
end

#=
XQCnames = [:tstu1_ratio, :tstu2_ratio]
XQC = convert(Array{Float64, 2}, LeftbhData[XQCnames])
XQC = XQC .- mean(XQC, weights(Vector{Float64}(LeftbhData[:cw_age9])), 1)
XQC = SharedArray(copy(XQC'))
=#
